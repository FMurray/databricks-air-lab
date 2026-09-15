# Databricks notebook source
# MAGIC %md
# MAGIC # AIR wheelhouse builder — implementation
# MAGIC
# MAGIC This helper runs only on classic compute. `build_wheelhouse` resolves a developer
# MAGIC `requirements.txt` against a checked-in AIR environment profile, publishes the exact wheel
# MAGIC delta to a UC Volume, and emits a serverless `environment.yaml` tied to the resolved lock.

# COMMAND ----------
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    from pip._vendor.packaging.tags import compatible_tags, cpython_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import InvalidVersion, Version


def _load_pins(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(stripped)
        except InvalidRequirement as exc:
            raise ValueError(f"unsupported lock line: {stripped!r}") from exc
        exact = [item.version for item in requirement.specifier if item.operator == "=="]
        if len(exact) != 1 or requirement.url:
            raise ValueError(f"expected one exact version for {requirement.name!r}")
        name = canonicalize_name(requirement.name)
        if name in pins and pins[name] != exact[0]:
            raise ValueError(f"multiple target versions found for {name!r}")
        pins[name] = exact[0]
    return pins


def _same_version(left, right):
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right


def _version_delta(closure, baseline):
    return {
        name: version
        for name, version in closure.items()
        if name not in baseline or not _same_version(version, baseline[name])
    }


def _direct_resolution(requirement_lines, closure, baseline, marker_environment):
    requested = []
    issues = []
    for number, line in enumerate(requirement_lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(stripped)
        except InvalidRequirement as exc:
            issues.append({"line": number, "requirement": stripped, "error": str(exc)})
            continue
        if requirement.marker and not requirement.marker.evaluate(marker_environment):
            continue
        name = canonicalize_name(requirement.name)
        requested.append(str(requirement))
        resolved = closure.get(name)
        if resolved is None:
            issues.append({
                "line": number,
                "requirement": str(requirement),
                "error": "package is absent from resolved.lock",
            })
            continue
        if requirement.specifier and not requirement.specifier.contains(resolved, prereleases=True):
            issues.append({
                "line": number,
                "requirement": str(requirement),
                "resolved": resolved,
                "error": "resolved version does not satisfy the direct requirement",
            })
            continue
        if name not in baseline or not _same_version(resolved, baseline[name]):
            requested[-1] += f" -> delta {name}=={resolved}"
        else:
            requested[-1] += f" -> reused {name}=={resolved}"
    return requested, issues


def _target_args(target):
    args = [
        "--implementation", target.get("implementation", "cp"),
        "--python-version", target["python_version"],
    ]
    abis = target.get("abis") or [target["abi"], "abi3", "none"]
    for abi in dict.fromkeys(abis):
        args += ["--abi", abi]
    for platform in target.get("platforms") or []:
        args += ["--platform", platform]
    return args


def _python_tuple(target):
    compact = str(target["python_version"])
    return int(compact[0]), int(compact[1:])


def _supported_target_tags(target):
    configured = target.get("supported_tags") or []
    if configured:
        return set(configured)
    python_version = _python_tuple(target)
    platforms = target.get("platforms") or []
    abis = target.get("abis") or [target["abi"], "abi3", "none"]
    interpreter = f"{target.get('implementation', 'cp')}{target['python_version']}"
    tags = set(map(str, cpython_tags(python_version, abis, platforms)))
    tags.update(map(str, compatible_tags(python_version, interpreter, platforms)))
    return tags


def _uv_platform(target):
    for platform in target.get("platforms") or []:
        match = re.match(r"manylinux_(\d+)_(\d+)_(x86_64|aarch64)$", platform)
        if match:
            major, minor, architecture = match.groups()
            return f"{architecture}-manylinux_{major}_{minor}"
        match = re.match(r"manylinux2014_(x86_64|aarch64)$", platform)
        if match:
            return f"{match.group(1)}-manylinux2014"
    markers = target.get("marker_environment") or {}
    machine = markers.get("platform_machine", "")
    system = markers.get("sys_platform", "")
    if system == "darwin":
        return "aarch64-apple-darwin" if machine in {"arm64", "aarch64"} \
            else "x86_64-apple-darwin"
    if system in {"linux", "linux2"}:
        return "aarch64-unknown-linux-gnu" if machine in {"arm64", "aarch64"} \
            else "x86_64-unknown-linux-gnu"
    raise ValueError(f"cannot map target platforms for uv: {target.get('platforms')}")


def _configured_index_url():
    for name in ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "PIP_INDEX_URL"):
        if os.environ.get(name):
            return os.environ[name]
    for key in ("global.index-url", "site.index-url"):
        configured = subprocess.run(
            [sys.executable, "-m", "pip", "config", "get", key],
            capture_output=True, text=True,
        )
        if configured.returncode == 0 and configured.stdout.strip():
            return configured.stdout.strip()
    return ""


def _index_args(index_url="", find_links=""):
    if find_links:
        return ["--no-index", "--find-links", str(find_links)]
    effective = index_url or _configured_index_url()
    return ["--index-url", effective] if effective else []


def _ensure_uv(index_args):
    local_uv = Path(sys.executable).parent / "uv"
    uv = shutil.which("uv") or (str(local_uv) if local_uv.exists() else "")
    if uv:
        return uv
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "uv"] + index_args,
        capture_output=True, text=True,
    )
    uv = shutil.which("uv") or (str(local_uv) if local_uv.exists() else "")
    if installed.returncode != 0 or not uv:
        detail = installed.stderr[-2000:]
        raise RuntimeError(f"could not install uv from the configured index: {detail}")
    return uv


def _resolve(requirements, constraints, output, target, index_args, cache_dir):
    uv = _ensure_uv(index_args)
    shutil.copy2(constraints, output)
    python_version = (target.get("marker_environment") or {}).get("python_version")
    if not python_version:
        major, minor = _python_tuple(target)
        python_version = f"{major}.{minor}"
    command = [
        uv, "pip", "compile", str(requirements),
        "--output-file", str(output),
        "--python-version", python_version,
        "--python-platform", _uv_platform(target),
        "--only-binary=:all:",
        "--cache-dir", str(cache_dir),
        "--no-header", "--no-annotate", "--no-progress", "--color", "never",
    ]
    return subprocess.run(command + index_args, capture_output=True, text=True)


def _compatible_wheel_tags(filename, supported_tags):
    _distribution, _version, _build, wheel_tags = parse_wheel_filename(Path(filename).name)
    return sorted(str(tag) for tag in wheel_tags if str(tag) in supported_tags)


def _wheel_record(path, supported_tags):
    distribution, version, _build, _wheel_tags = parse_wheel_filename(Path(path).name)
    return {
        "name": canonicalize_name(distribution),
        "version": str(version),
        "file": Path(path).name,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "compatible_tags": _compatible_wheel_tags(path, supported_tags),
    }


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _environment_yaml(target, wheelhouse, delta_lock, has_delta):
    base_environment = target["air_environment"]
    environment_version = target["environment_version"]
    if not has_delta:
        return (
            f"base_environment: {json.dumps(base_environment)}\n"
            f"environment_version: {json.dumps(environment_version)}\n"
            "dependencies: []\n"
        )
    dependencies = [
        "--no-index",
        "--no-deps",
        "--require-hashes",
        f"--find-links {wheelhouse}",
        f"-r {delta_lock}",
    ]
    return (
        f"base_environment: {json.dumps(base_environment)}\n"
        f"environment_version: {json.dumps(environment_version)}\n"
        "dependencies:\n"
        + "".join(f"  - {json.dumps(item)}\n" for item in dependencies)
    )


def _failure(volume, request_id, stage, error, **details):
    manifest = {
        "ok": False,
        "engine": "uv",
        "request_id": request_id,
        "stage": stage,
        "error": error,
        **details,
    }
    _write_json(Path(volume) / "requests" / f"{request_id}.json", manifest)
    return manifest


def build_wheelhouse(requirements_file, wheelhouse_volume, profile_dir, index_url="",
                     work_root="", find_links=""):
    requirements = Path(requirements_file)
    volume = Path(wheelhouse_volume)
    profile = Path(profile_dir)
    constraints = profile / "constraints.txt"
    target_path = profile / "target_env.json"
    for required in (requirements, constraints, target_path):
        if not required.is_file():
            raise FileNotFoundError(f"required input does not exist: {required}")
    target = json.loads(target_path.read_text())
    air_environment = target["air_environment"]
    request_hash = hashlib.sha256()
    for path in (requirements, constraints, target_path):
        request_hash.update(path.name.encode())
        request_hash.update(b"\0")
        request_hash.update(path.read_bytes())
        request_hash.update(b"\0")
    request_id = request_hash.hexdigest()[:16]
    volume.mkdir(parents=True, exist_ok=True)

    scratch = Path(work_root) if work_root else Path("/local_disk0")
    local_root = scratch / f"air_wheelhouse_{request_id}"
    shutil.rmtree(local_root, ignore_errors=True)
    local_root.mkdir(parents=True)
    local_wheelhouse = local_root / "wheelhouse"
    local_wheelhouse.mkdir()
    resolved_path = local_root / "resolved.lock"
    index_args = _index_args(index_url, find_links)

    try:
        baseline = _load_pins(constraints)
        result = _resolve(
            requirements, constraints, resolved_path, target, index_args, local_root / "uv-cache",
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return _failure(volume, request_id, "resolve", str(exc))
    if result.returncode != 0:
        return _failure(
            volume, request_id, "resolve",
            "uv could not resolve the requirements while preferring the AIR baseline",
            stderr_tail=result.stderr[-5000:],
        )
    version_result = subprocess.run(
        [str(result.args[0]), "--version"], capture_output=True, text=True,
    )
    uv_version = version_result.stdout.strip() or "unknown"

    try:
        closure = _load_pins(resolved_path)
        supported_tags = _supported_target_tags(target)
    except (OSError, ValueError) as exc:
        return _failure(volume, request_id, "lock", str(exc))
    delta = _version_delta(closure, baseline)
    direct_resolution, direct_issues = _direct_resolution(
        requirements.read_text().splitlines(),
        closure,
        baseline,
        target.get("marker_environment") or {},
    )
    if direct_issues:
        return _failure(
            volume,
            request_id,
            "validate-resolution",
            "uv reported success but its lock omitted or contradicted a direct requirement",
            requirements_file=str(requirements),
            requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
            uv_version=uv_version,
            direct_resolution=direct_resolution,
            issues=direct_issues,
            resolved_lock_text=resolved_path.read_text(),
        )
    print(f"AIR profile {air_environment} | closure {len(closure)} | "
          f"reused {len(closure) - len(delta)} | wheel delta {len(delta)}")
    print("direct requirements:")
    for item in direct_resolution:
        print(f"  {item}")

    failures = []
    target_args = _target_args(target)
    for name, version in sorted(delta.items()):
        spec = f"{name}=={version}"
        downloaded = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
             "--dest", str(local_wheelhouse), spec] + target_args + index_args,
            capture_output=True, text=True,
        )
        if downloaded.returncode != 0:
            failures.append({"spec": spec, "stderr_tail": downloaded.stderr[-1500:]})

    records = []
    for wheel in sorted(local_wheelhouse.glob("*.whl")):
        record = _wheel_record(wheel, supported_tags)
        if not record["compatible_tags"]:
            failures.append({
                "spec": f"{record['name']}=={record['version']}",
                "error": f"{record['file']} has no wheel tag accepted by {air_environment}",
            })
        records.append(record)
    downloaded_keys = {(item["name"], Version(item["version"])) for item in records}
    for name, version in delta.items():
        if (name, Version(version)) not in downloaded_keys:
            failures.append({"spec": f"{name}=={version}", "error": "no wheel was downloaded"})
    if failures:
        return _failure(
            volume, request_id, "download",
            "one or more exact wheels could not be downloaded for the AIR target",
            failures=failures,
        )

    lock_text = "".join(f"{name}=={version}\n" for name, version in sorted(closure.items()))
    lock_id = hashlib.sha256(
        air_environment.encode() + b"\0" + lock_text.encode()
    ).hexdigest()[:16]
    build = volume / "builds" / lock_id
    shutil.rmtree(build, ignore_errors=True)
    wheelhouse = build / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    for wheel in local_wheelhouse.glob("*.whl"):
        shutil.copy2(wheel, wheelhouse / wheel.name)

    resolved_lock = build / "resolved.lock"
    delta_lock = build / "delta.lock"
    environment_file = build / "environment.yaml"
    resolved_lock.write_text(lock_text)
    records_by_key = defaultdict(list)
    for record in records:
        records_by_key[(record["name"], Version(record["version"]))].append(record)
    delta_lines = []
    for name, version in sorted(delta.items()):
        hashes = " ".join(
            f"--hash=sha256:{item['sha256']}"
            for item in records_by_key[(name, Version(version))]
        )
        delta_lines.append(f"{name}=={version} {hashes}".rstrip())
    delta_lock.write_text("\n".join(delta_lines) + ("\n" if delta_lines else ""))
    environment_file.write_text(
        _environment_yaml(target, wheelhouse, delta_lock, bool(delta))
    )
    shutil.copy2(requirements, build / "requirements.txt")

    manifest = {
        "ok": True,
        "engine": "uv",
        "request_id": request_id,
        "lock_id": lock_id,
        "requirements_file": str(requirements),
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "uv_version": uv_version,
        "air_environment": air_environment,
        "environment_version": target["environment_version"],
        "baseline_count": len(baseline),
        "closure_count": len(closure),
        "direct_resolution": direct_resolution,
        "reused_from_air": len(closure) - len(delta),
        "delta": [f"{name}=={version}" for name, version in sorted(delta.items())],
        "resolved_lock": str(resolved_lock),
        "delta_lock": str(delta_lock),
        "wheelhouse": str(wheelhouse),
        "environment_file": str(environment_file),
        "wheel_files": records,
        "target_top_tag": target["top_tag"],
    }
    _write_json(build / "manifest.json", manifest)
    _write_json(volume / "requests" / f"{request_id}.json", manifest)
    return manifest
