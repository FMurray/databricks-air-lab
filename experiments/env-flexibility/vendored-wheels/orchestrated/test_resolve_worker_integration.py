"""End-to-end resolver test using only locally built wheels."""

import base64
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from io import StringIO
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "air_wheelhouse_integration_resolver", HERE / "resolve_worker.py"
)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def _wheel_hash(data):
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _record(rows):
    output = StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue().encode()


def _build_wheel(destination, name, version, requires=()):
    """Build a valid minimal pure-Python wheel without build tools or network access."""
    distribution = re.sub(r"[-_.]+", "_", name)
    dist_info = f"{distribution}-{version}.dist-info"
    filename = f"{distribution}-{version}-py3-none-any.whl"
    metadata = (
        "Metadata-Version: 2.1\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
        + "\n"
    ).encode()
    files = {
        f"{distribution.lower()}/__init__.py": f'__version__ = "{version}"\n'.encode(),
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: air-wheelhouse-integration-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
    }
    record_path = f"{dist_info}/RECORD"
    rows = [(path, _wheel_hash(data), str(len(data))) for path, data in sorted(files.items())]
    rows.append((record_path, "", ""))
    files[record_path] = _record(rows)

    wheel = destination / filename
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel


def _build_package_source(directory):
    directory.mkdir(parents=True)
    definitions = [
        ("airlab-openai", "1.0.0", ("airlab-jiter>=0.8,<0.9", "airlab-array>=2")),
        ("airlab-openai", "2.0.0", ("airlab-jiter>=0.10,<1", "airlab-array>=2")),
        ("airlab-jiter", "0.8.0", ()),
        ("airlab-jiter", "0.11.0", ()),
        ("airlab-array", "2.1.3", ()),
        ("airlab-array", "2.3.0", ()),
        ("airlab-app", "1.0.0", ("airlab-parent>=1", "airlab-shared>=2")),
        ("airlab-parent", "1.0.0", ("airlab-shared<2",)),
        ("airlab-parent", "2.0.0", ("airlab-shared>=2",)),
        ("airlab-shared", "1.0.0", ()),
        ("airlab-shared", "2.0.0", ()),
    ]
    for name, version, requires in definitions:
        _build_wheel(directory, name, version, requires)


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=True)


def _target_environment():
    return {
        "profile_schema": 1,
        "air_environment": "databricks_ai_test",
        "python_full": "3.12.3",
        "python_version": "312",
        "implementation": "cp",
        "abi": "cp312",
        "abis": ["cp312", "abi3", "none"],
        "top_tag": "cp312-cp312-manylinux_2_39_x86_64",
        "platforms": [
            "manylinux_2_39_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
            "linux_x86_64",
        ],
        "marker_environment": {
            "implementation_name": "cpython",
            "platform_machine": "x86_64",
            "platform_python_implementation": "CPython",
            "python_full_version": "3.12.3",
            "python_version": "3.12",
            "sys_platform": "linux",
        },
    }


@contextmanager
def _isolated_package_configuration():
    names = (
        "PIP_CONFIG_FILE",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_NO_INDEX",
        "PIP_FIND_LINKS",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
    )
    previous = {name: os.environ.get(name) for name in names}
    previous_disable_check = os.environ.get("PIP_DISABLE_PIP_VERSION_CHECK")
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ["PIP_CONFIG_FILE"] = os.devnull
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if previous_disable_check is None:
            os.environ.pop("PIP_DISABLE_PIP_VERSION_CHECK", None)
        else:
            os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = previous_disable_check


@unittest.skipUnless(resolver.shutil.which("uv"), "uv is not installed")
class OfflineWheelhouseIntegrationTest(unittest.TestCase):
    def test_static_profile_resolves_conflicts_and_installs_hashed_delta(self):
        with tempfile.TemporaryDirectory(prefix="air-wheelhouse-test-") as temporary:
            root = Path(temporary)
            packages = root / "packages"
            _build_package_source(packages)

            baseline = root / "baseline"
            _run([sys.executable, "-m", "venv", str(baseline)])
            baseline_python = baseline / "bin" / "python"
            _run([
                str(baseline_python), "-m", "pip", "install", "--no-index",
                "--find-links", str(packages),
                "airlab-openai==1.0.0", "airlab-jiter==0.8.0", "airlab-array==2.1.3",
                "airlab-parent==1.0.0", "airlab-shared==1.0.0",
            ])

            profile = root / "profile"
            profile.mkdir()
            freeze = _run([str(baseline_python), "-m", "pip", "freeze", "--all"]).stdout
            (profile / "constraints.txt").write_text(freeze)
            (profile / "target_env.json").write_text(
                json.dumps(_target_environment(), indent=2) + "\n"
            )
            requirements = root / "requirements.txt"
            requirements.write_text("airlab-openai==2.0.0\nairlab-app==1.0.0\n")
            volume = root / "volume"

            with _isolated_package_configuration():
                manifest = resolver.build_wheelhouse(
                    requirements,
                    volume,
                    profile,
                    work_root=root / "scratch",
                    find_links=packages,
                )

            self.assertTrue(manifest["ok"], json.dumps(manifest, indent=2))
            self.assertEqual(
                manifest["delta"],
                [
                    "airlab-app==1.0.0",
                    "airlab-jiter==0.11.0",
                    "airlab-openai==2.0.0",
                    "airlab-parent==2.0.0",
                    "airlab-shared==2.0.0",
                ],
            )
            self.assertNotIn("airlab-array==2.3.0", Path(manifest["resolved_lock"]).read_text())
            self.assertEqual(len(manifest["wheel_files"]), 5)

            build = volume / "builds" / manifest["lock_id"]
            expected_lock_id = hashlib.sha256(
                b"databricks_ai_test\0" + Path(manifest["resolved_lock"]).read_bytes()
            ).hexdigest()[:16]
            self.assertEqual(manifest["lock_id"], expected_lock_id)
            self.assertEqual(Path(manifest["environment_file"]), build / "environment.yaml")
            built_manifest = json.loads((build / "manifest.json").read_text())
            self.assertEqual(built_manifest["lock_id"], manifest["lock_id"])
            environment = (build / "environment.yaml").read_text()
            self.assertIn('version: "databricks_ai_test"', environment)
            self.assertNotIn("-r ", environment)
            for wheel in manifest["wheel_files"]:
                self.assertIn(str(build / "wheelhouse" / wheel["file"]), environment)

            install = _run([
                str(baseline_python), "-m", "pip", "install",
                "--no-index", "--require-hashes", "--find-links", manifest["wheelhouse"],
                "-r", manifest["delta_lock"],
            ])
            self.assertIn("Successfully installed", install.stdout)
            _run([str(baseline_python), "-m", "pip", "check"])
            installed = _run([str(baseline_python), "-m", "pip", "freeze"]).stdout
            self.assertIn("airlab-array==2.1.3", installed)
            self.assertIn("airlab-jiter==0.11.0", installed)
            self.assertNotIn("airlab-array==2.3.0", installed)


if __name__ == "__main__":
    unittest.main()
