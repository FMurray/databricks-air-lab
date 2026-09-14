# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — WORKER (%run helper)
# MAGIC
# MAGIC Run this on a classic cluster with Artifactory access. The AIR-side freeze supplies four
# MAGIC inputs: `requirements.txt`, the complete AIR `constraints.txt`, `overrides.txt` (normally
# MAGIC empty), and `target_env.json` with AIR's supported wheel tags.
# MAGIC
# MAGIC The worker follows one rule: an AIR package is reused when its installed version satisfies
# MAGIC everything the requested package graph asks of it. If the requested graph requires another
# MAGIC version, that exact version becomes an override and enters the delta. The final delta is
# MAGIC therefore computed by **name and version**, never by name alone.

# COMMAND ----------
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import InvalidVersion, Version
except ModuleNotFoundError:
    # pip vendors packaging, so the worker can still run in a clean Python environment where
    # packaging is not installed as a top-level package. MLR normally takes the first branch.
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import InvalidVersion, Version


def _cfg(name, default=""):
    """Prefer a variable set by a %run driver; fall back to a widget."""
    if name in globals() and globals()[name]:
        return globals()[name]
    try:
        value = dbutils.widgets.get(name)
        if value:
            return value
    except Exception:
        pass
    return default


_JOB_MODE = not (("stage_dir" in globals()) and globals().get("stage_dir"))


def _load_pins(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if "==" in stripped and not stripped.startswith("#"):
            name, version = stripped.split("==", 1)
            pins[canonicalize_name(name.strip())] = version.strip()
    return pins


def _load_override_names(path):
    path = Path(path)
    if not path.exists():
        return set()
    overrides = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            requirement = Requirement(stripped)
        except InvalidRequirement as exc:
            raise ValueError(f"invalid override name {stripped!r}") from exc
        if requirement.specifier or requirement.url or requirement.extras or requirement.marker:
            raise ValueError(f"override entries are package names only, got {stripped!r}")
        overrides.add(canonicalize_name(requirement.name))
    return overrides


def _request_fingerprint(stage):
    digest = hashlib.sha256()
    for name in ("constraints.txt", "requirements.txt", "overrides.txt", "target_env.json"):
        path = Path(stage) / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.exists() else b"")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _target_args(target_env):
    args = [
        "--implementation", target_env.get("implementation", "cp"),
        "--python-version", target_env["python_version"],
    ]
    for abi in target_env.get("abis") or [target_env["abi"]]:
        args += ["--abi", abi]
    for platform in target_env.get("platforms", []):
        args += ["--platform", platform]
    return args


def _pip_resolve(requirements, constraints, report, target_args, index_args):
    """Resolve for AIR's interpreter and platform without installing anything."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--dry-run", "--ignore-installed", "--only-binary=:all:",
        "--report", str(report), "-r", str(requirements),
    ]
    if constraints:
        cmd += ["-c", str(constraints)]
    return subprocess.run(cmd + target_args + index_args, capture_output=True, text=True)


def _marker_applies(requirement, marker_environment, active_extras):
    if requirement.marker is None:
        return True
    extras = active_extras or {""}
    for extra in extras | {""}:
        environment = dict(marker_environment or {})
        environment["extra"] = extra
        if requirement.marker.evaluate(environment=environment):
            return True
    return False


def _parse_top_level_requirements(lines, marker_environment):
    parsed, skipped = [], []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("-", "--")):
            skipped.append(stripped)
            continue
        try:
            requirement = Requirement(stripped)
        except InvalidRequirement:
            skipped.append(stripped)
            continue
        if _marker_applies(requirement, marker_environment, {""}):
            parsed.append(requirement)
    return parsed, skipped


def _version_satisfies(version, requirements):
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    for requirement, _parent in requirements:
        if requirement.url or (
            requirement.specifier
            and not requirement.specifier.contains(parsed, prereleases=True)
        ):
            return False
    return True


def infer_required_overrides(requirement_lines, wants_install, baseline,
                             marker_environment=None, explicit_overrides=()):
    """Find AIR pins that the requested graph cannot reuse.

    The unconstrained report supplies candidate metadata, but a mere version difference is not a
    conflict. Starting at the requested packages, walk only candidates that must be added/replaced.
    A baseline version remains pinned whenever it satisfies every incoming requirement.
    """
    selected = {
        canonicalize_name(item["metadata"]["name"]): item
        for item in wants_install
    }
    explicit = {canonicalize_name(name) for name in explicit_overrides}
    incoming = defaultdict(list)
    active_extras = defaultdict(set)
    queued = set()
    queue = deque()

    def enqueue(requirement, parent):
        name = canonicalize_name(requirement.name)
        key = (str(requirement), parent)
        added = key not in {(str(req), source) for req, source in incoming[name]}
        if added:
            incoming[name].append((requirement, parent))
        before = len(active_extras[name])
        active_extras[name].update(requirement.extras)
        extras_changed = len(active_extras[name]) != before
        if (added or extras_changed) and name not in queued:
            queue.append(name)
            queued.add(name)

    direct, skipped = _parse_top_level_requirements(requirement_lines, marker_environment or {})
    for requirement in direct:
        enqueue(requirement, "<target requirements>")

    overrides = set(explicit)
    reasons = defaultdict(list)
    expanded_for_extras = {}
    missing_candidates = set()

    while queue:
        name = queue.popleft()
        queued.discard(name)
        requirements = incoming[name]
        reuse_baseline = (
            name in baseline
            and name not in explicit
            and _version_satisfies(baseline[name], requirements)
        )
        if reuse_baseline:
            continue

        if name in baseline:
            overrides.add(name)
            reasons[name] = sorted({
                f"{parent} requires {requirement}"
                for requirement, parent in requirements
                if requirement.url
                or not _version_satisfies(baseline[name], [(requirement, parent)])
            }) or ["listed explicitly in overrides.txt"]

        candidate = selected.get(name)
        if candidate is None:
            missing_candidates.add(name)
            continue

        extras_key = frozenset(active_extras[name])
        if expanded_for_extras.get(name) == extras_key:
            continue
        expanded_for_extras[name] = extras_key
        for raw_requirement in candidate["metadata"].get("requires_dist") or []:
            try:
                dependency = Requirement(raw_requirement)
            except InvalidRequirement:
                continue
            if _marker_applies(dependency, marker_environment or {}, set(extras_key)):
                enqueue(dependency, f"{name}=={candidate['metadata']['version']}")

    return {
        "overrides": overrides,
        "reasons": dict(reasons),
        "skipped_requirement_lines": skipped,
        "missing_candidates": sorted(missing_candidates),
    }


def _write_effective_constraints(source, destination, overrides):
    kept = []
    for line in Path(source).read_text().splitlines():
        stripped = line.strip()
        name = canonicalize_name(stripped.split("==", 1)[0]) if "==" in stripped else None
        if name not in overrides:
            kept.append(line)
    Path(destination).write_text("\n".join(kept) + "\n")


def _version_delta(closure, baseline):
    delta = {}
    for name, version in closure.items():
        try:
            same = name in baseline and Version(baseline[name]) == Version(version)
        except InvalidVersion:
            same = name in baseline and baseline[name] == version
        if not same:
            delta[name] = version
    return delta


def restore_compatible_pins(relaxed_overrides, fixed_overrides, can_resolve):
    """Restore AIR pins in groups while keeping every intermediate constraint set resolvable.

    ``relaxed_overrides`` is a known-good ceiling derived from the unconstrained target graph.
    Explicit and directly proven overrides stay fixed. The remaining pins are restored in groups;
    failed groups are bisected until the individual pins that break resolution are isolated.
    """
    overrides = set(relaxed_overrides)
    fixed = set(fixed_overrides)
    restored = set()
    probes = 0

    def restore_group(names):
        nonlocal overrides, probes
        if not names:
            return
        trial = overrides - set(names)
        probes += 1
        if can_resolve(trial):
            overrides = trial
            restored.update(names)
            return
        if len(names) == 1:
            return
        middle = len(names) // 2
        restore_group(names[:middle])
        restore_group(names[middle:])

    restore_group(sorted(overrides - fixed))
    return overrides, restored, probes


def _compatible_wheel_tags(filename, supported_tags):
    _distribution, _version, _build, wheel_tags = parse_wheel_filename(Path(filename).name)
    return sorted(str(tag) for tag in wheel_tags if str(tag) in supported_tags)


def _wheel_record(path, supported_tags):
    distribution, version, _build, _wheel_tags = parse_wheel_filename(Path(path).name)
    compatible = _compatible_wheel_tags(path, supported_tags)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "name": canonicalize_name(distribution),
        "version": str(version),
        "file": Path(path).name,
        "sha256": digest,
        "compatible_tags": compatible,
    }


def _write_manifest(stage, manifest):
    stage = Path(stage)
    temporary = stage / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, stage / "manifest.json")


def resolve_and_download(stage, index_url="", work_root=""):
    stage = Path(stage)
    requirements = stage / "requirements.txt"
    constraints = stage / "constraints.txt"
    overrides_path = stage / "overrides.txt"
    target_path = stage / "target_env.json"
    request_id = _request_fingerprint(stage)
    for required in (requirements, constraints, overrides_path, target_path):
        if not required.exists():
            manifest = {"ok": False, "stage": "inputs", "request_id": request_id,
                        "error": f"missing required input: {required}"}
            _write_manifest(stage, manifest)
            return manifest

    try:
        target_env = json.loads(target_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        manifest = {"ok": False, "stage": "inputs", "request_id": request_id,
                    "error": f"invalid target_env.json: {exc}"}
        _write_manifest(stage, manifest)
        return manifest
    supported_tags = set(target_env.get("supported_tags") or [])
    if not supported_tags:
        manifest = {
            "ok": False, "stage": "inputs", "request_id": request_id,
            "error": "target_env.json has no supported_tags; rerun the rewritten AIR freeze",
        }
        _write_manifest(stage, manifest)
        return manifest
    host_python = f"{sys.version_info.major}{sys.version_info.minor}"
    if host_python != target_env["python_version"]:
        manifest = {
            "ok": False, "stage": "inputs",
            "request_id": request_id,
            "error": f"classic Python cp{host_python} does not match AIR cp"
                     f"{target_env['python_version']}; pip's wheel flags cross-target downloads, "
                     "but dependency markers still require a matching resolver interpreter",
        }
        _write_manifest(stage, manifest)
        return manifest
    host_markers = default_environment()
    air_markers = target_env.get("marker_environment") or {}
    marker_mismatches = {
        key: {"classic": host_markers.get(key), "air": air_markers.get(key)}
        for key in ("implementation_name", "platform_machine", "sys_platform")
        if air_markers.get(key) and host_markers.get(key) != air_markers.get(key)
    }
    if marker_mismatches:
        manifest = {
            "ok": False, "stage": "inputs", "request_id": request_id,
            "error": f"classic and AIR dependency-marker environments differ: {marker_mismatches}",
        }
        _write_manifest(stage, manifest)
        return manifest

    _write_manifest(stage, {"ok": False, "stage": "running", "request_id": request_id,
                            "error": "classic worker has not finished"})
    scratch_root = Path(work_root) if work_root else Path("/local_disk0")
    local_root = scratch_root / f"vendor_wheelhouse_{request_id}"
    shutil.rmtree(local_root, ignore_errors=True)
    local_root.mkdir(parents=True)
    local_wheelhouse = local_root / "wheelhouse"
    local_wheelhouse.mkdir()
    effective = local_root / "constraints.effective.txt"
    report = local_root / "resolved.json"
    wants_report = local_root / "wants.json"

    try:
        baseline = _load_pins(constraints)
        explicit = _load_override_names(overrides_path)
    except (OSError, ValueError) as exc:
        manifest = {"ok": False, "stage": "inputs", "request_id": request_id,
                    "error": str(exc)}
        _write_manifest(stage, manifest)
        return manifest
    target_args = _target_args(target_env)
    index_args = ["--index-url", index_url] if index_url else []
    _write_effective_constraints(constraints, effective, explicit)

    print(f"request {request_id} | AIR target {target_env['top_tag']} | "
          f"explicit overrides {sorted(explicit) or 'none'}")
    first = _pip_resolve(requirements, effective, report, target_args, index_args)
    detected = {"overrides": set(), "reasons": {}, "skipped_requirement_lines": [],
                "missing_candidates": []}
    all_overrides = set(explicit)
    reconciled = set()
    reconciliation_probes = 0
    override_reasons = {}

    if first.returncode != 0:
        wants = _pip_resolve(requirements, None, wants_report, target_args, index_args)
        if wants.returncode != 0:
            manifest = {
                "ok": False, "stage": "resolve-wants", "request_id": request_id,
                "error": "target requirements do not resolve from Artifactory even without AIR constraints",
                "stderr_tail": wants.stderr[-4000:],
                "constrained_stderr_tail": first.stderr[-2000:],
            }
            _write_manifest(stage, manifest)
            return manifest

        wants_install = json.loads(wants_report.read_text()).get("install", [])
        detected = infer_required_overrides(
            requirements.read_text().splitlines(), wants_install, baseline,
            target_env.get("marker_environment") or {}, explicit,
        )
        all_overrides |= detected["overrides"]
        override_reasons.update(detected["reasons"])
        _write_effective_constraints(constraints, effective, all_overrides)
        print(f"detected overrides: {sorted(all_overrides) or 'none'}")
        final = _pip_resolve(requirements, effective, report, target_args, index_args)

        if final.returncode != 0:
            wants_closure = {
                canonicalize_name(item["metadata"]["name"]): item["metadata"]["version"]
                for item in wants_install
            }
            relaxation_candidates = {
                name for name in _version_delta(wants_closure, baseline)
                if name in baseline
            }
            fixed_overrides = set(explicit) | detected["overrides"]
            relaxed_overrides = fixed_overrides | relaxation_candidates
            relaxed_effective = local_root / "constraints.relaxed.txt"
            relaxed_report = local_root / "relaxed.json"
            _write_effective_constraints(constraints, relaxed_effective, relaxed_overrides)
            relaxed = _pip_resolve(
                requirements, relaxed_effective, relaxed_report, target_args, index_args,
            )
            if relaxed.returncode != 0:
                manifest = {
                    "ok": False, "stage": "resolve-relaxed", "request_id": request_id,
                    "error": "the unconstrained target resolved, but automatically relaxing every "
                             "different AIR pin did not reproduce that resolution",
                    "explicit_overrides": sorted(explicit),
                    "detected_overrides": sorted(detected["overrides"] - explicit),
                    "override_reasons": override_reasons,
                    "skipped_requirement_lines": detected["skipped_requirement_lines"],
                    "missing_candidates": detected["missing_candidates"],
                    "stderr_tail": relaxed.stderr[-4000:],
                }
                _write_manifest(stage, manifest)
                return manifest

            probe_number = 0

            def can_resolve(trial_overrides):
                nonlocal probe_number
                probe_number += 1
                trial_constraints = local_root / f"constraints.probe-{probe_number}.txt"
                trial_report = local_root / f"probe-{probe_number}.json"
                _write_effective_constraints(constraints, trial_constraints, trial_overrides)
                trial = _pip_resolve(
                    requirements, trial_constraints, trial_report, target_args, index_args,
                )
                return trial.returncode == 0

            print("direct conflict walk was insufficient; restoring compatible AIR pins by probe")
            all_overrides, _restored, reconciliation_probes = restore_compatible_pins(
                relaxed_overrides, fixed_overrides, can_resolve,
            )
            reconciled = all_overrides - fixed_overrides
            for name in sorted(reconciled):
                override_reasons[name] = [
                    f"restoring the AIR pin {name}=={baseline[name]} makes resolution fail"
                ]
            print(f"automatically reconciled overrides: {sorted(reconciled) or 'none'} "
                  f"({reconciliation_probes} probes)")
            _write_effective_constraints(constraints, effective, all_overrides)
            final = _pip_resolve(requirements, effective, report, target_args, index_args)
    else:
        final = first

    if final.returncode != 0:
        manifest = {
            "ok": False, "stage": "resolve-effective", "request_id": request_id,
            "error": "automatic AIR-pin reconciliation produced a constraint set that no longer "
                     "resolves",
            "explicit_overrides": sorted(explicit),
            "detected_overrides": sorted(all_overrides - explicit),
            "reconciled_overrides": sorted(reconciled),
            "override_reasons": override_reasons,
            "skipped_requirement_lines": detected["skipped_requirement_lines"],
            "missing_candidates": detected["missing_candidates"],
            "stderr_tail": final.stderr[-4000:],
        }
        _write_manifest(stage, manifest)
        return manifest

    install = json.loads(report.read_text()).get("install", [])
    closure = {
        canonicalize_name(item["metadata"]["name"]): item["metadata"]["version"]
        for item in install
    }
    delta = _version_delta(closure, baseline)
    print(f"closure {len(closure)} | reused from AIR {len(closure) - len(delta)} | delta {len(delta)}")
    for name, version in sorted(delta.items()):
        previous = f"; AIR has {baseline[name]}" if name in baseline else "; new"
        print(f"  {name}=={version}{previous}")

    failures = []
    for name, version in sorted(delta.items()):
        spec = f"{name}=={version}"
        download = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
             "--dest", str(local_wheelhouse), spec] + target_args + index_args,
            capture_output=True, text=True,
        )
        if download.returncode != 0:
            failures.append({"spec": spec, "stderr_tail": download.stderr[-1200:]})

    wheel_records = []
    for wheel in sorted(local_wheelhouse.glob("*.whl")):
        record = _wheel_record(wheel, supported_tags)
        if not record["compatible_tags"]:
            failures.append({
                "spec": f"{record['name']}=={record['version']}",
                "error": f"downloaded wheel {record['file']} has no tag accepted by AIR",
            })
        wheel_records.append(record)

    downloaded_keys = {(record["name"], Version(record["version"])) for record in wheel_records}
    for name, version in delta.items():
        if (name, Version(version)) not in downloaded_keys:
            failures.append({"spec": f"{name}=={version}", "error": "no matching wheel was staged"})

    if failures:
        manifest = {
            "ok": False, "stage": "download", "request_id": request_id,
            "error": "one or more exact delta wheels could not be downloaded for AIR's tags",
            "failures": failures,
            "explicit_overrides": sorted(explicit),
            "detected_overrides": sorted(all_overrides - explicit),
        }
        _write_manifest(stage, manifest)
        return manifest

    build = stage / "builds" / request_id
    shutil.rmtree(build, ignore_errors=True)
    wheelhouse = build / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    for wheel in local_wheelhouse.glob("*.whl"):
        shutil.copy2(wheel, wheelhouse / wheel.name)
    shutil.copy2(effective, build / "constraints.effective.txt")

    (build / "resolved.lock").write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(closure.items()))
    )
    records_by_key = defaultdict(list)
    for record in wheel_records:
        records_by_key[(record["name"], Version(record["version"]))].append(record)
    delta_lines = []
    for name, version in sorted(delta.items()):
        hashes = " ".join(
            f"--hash=sha256:{record['sha256']}"
            for record in records_by_key[(name, Version(version))]
        )
        delta_lines.append(f"{name}=={version} {hashes}".rstrip())
    (build / "delta.lock").write_text("\n".join(delta_lines) + ("\n" if delta_lines else ""))

    manifest = {
        "ok": True,
        "engine": "pip",
        "request_id": request_id,
        "closure_count": len(closure),
        "delta": [f"{name}=={version}" for name, version in sorted(delta.items())],
        "reused_from_air": len(closure) - len(delta),
        "explicit_overrides": sorted(explicit),
        "detected_overrides": sorted(all_overrides - explicit),
        "reconciled_overrides": sorted(reconciled),
        "reconciliation_probes": reconciliation_probes,
        "override_reasons": override_reasons,
        "constraints_for_install": str(build / "constraints.effective.txt"),
        "resolved_lock": str(build / "resolved.lock"),
        "delta_lock": str(build / "delta.lock"),
        "wheelhouse": str(wheelhouse),
        "wheel_files": wheel_records,
        "target_top_tag": target_env["top_tag"],
    }
    _write_manifest(stage, manifest)
    return manifest


# COMMAND ----------
if "dbutils" in globals():
    STAGE = _cfg("stage_dir")
    INDEX_URL = _cfg("index_url")
    assert STAGE, "stage_dir is required (set it in the driver or as a notebook parameter)"

    manifest = resolve_and_download(STAGE, INDEX_URL)
    print(json.dumps({k: v for k, v in manifest.items() if k != "wheel_files"},
                     indent=2, ensure_ascii=False))

    if _JOB_MODE:
        dbutils.notebook.exit(json.dumps(manifest, ensure_ascii=False))
    # else (%run): the driver reads `manifest` from the shared namespace.
