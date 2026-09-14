"""Offline integration test for the AIR-aware wheelhouse worker.

The test builds a private file:// package index containing tiny pure-Python wheels. It installs an
old dependency graph into a temporary environment, captures that environment with ``pip freeze``,
then requests a new parent whose metadata requires a newer transitive dependency. No package in
this test is read from public PyPI.
"""

import base64
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

try:
    from packaging.markers import default_environment
    from packaging.tags import sys_tags
except ModuleNotFoundError:
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.tags import sys_tags


HERE = Path(__file__).resolve().parent
WORKER = HERE / "resolve_worker.py"
SPEC = importlib.util.spec_from_file_location("air_wheelhouse_integration_resolver", WORKER)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


def _wheel_hash(data):
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _record(rows):
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode()


def _build_wheel(destination, name, version, requires=()):
    """Build a valid minimal wheel directly, without setuptools, build, or network access."""
    distribution = re.sub(r"[-_.]+", "_", name)
    module = distribution.lower()
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
        f"{module}/__init__.py": f'__version__ = "{version}"\n'.encode(),
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: air-wheelhouse-integration-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode(),
    }
    record_path = f"{dist_info}/RECORD"
    record_rows = [
        (path, _wheel_hash(data), str(len(data)))
        for path, data in sorted(files.items())
    ]
    record_rows.append((record_path, "", ""))
    files[record_path] = _record(record_rows)

    wheel = destination / filename
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel


def _normalized_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _build_index(root):
    packages = root / "packages"
    simple = root / "simple"
    packages.mkdir(parents=True)
    simple.mkdir()
    wheels = {
        "airlab-openai": [
            _build_wheel(
                packages,
                "airlab-openai",
                "1.0.0",
                ("airlab-jiter>=0.8,<0.9", "airlab-array>=2"),
            ),
            _build_wheel(
                packages,
                "airlab-openai",
                "2.0.0",
                ("airlab-jiter>=0.10,<1", "airlab-array>=2"),
            ),
        ],
        "airlab-jiter": [
            _build_wheel(packages, "airlab-jiter", "0.8.0"),
            _build_wheel(packages, "airlab-jiter", "0.11.0"),
        ],
        "airlab-array": [
            _build_wheel(packages, "airlab-array", "2.1.3"),
            _build_wheel(packages, "airlab-array", "2.3.0"),
        ],
        "airlab-app": [
            _build_wheel(
                packages,
                "airlab-app",
                "1.0.0",
                ("airlab-parent>=1", "airlab-shared>=2"),
            ),
        ],
        "airlab-parent": [
            _build_wheel(packages, "airlab-parent", "1.0.0", ("airlab-shared<2",)),
            _build_wheel(packages, "airlab-parent", "2.0.0", ("airlab-shared>=2",)),
        ],
        "airlab-shared": [
            _build_wheel(packages, "airlab-shared", "1.0.0"),
            _build_wheel(packages, "airlab-shared", "2.0.0"),
        ],
    }
    root_links = []
    for name, project_wheels in wheels.items():
        normalized = _normalized_name(name)
        project = simple / normalized
        project.mkdir()
        links = []
        for wheel in project_wheels:
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            links.append(
                f'<a href="../../packages/{wheel.name}#sha256={digest}">{wheel.name}</a>'
            )
        (project / "index.html").write_text("\n".join(links) + "\n")
        root_links.append(f'<a href="{normalized}/">{name}</a>')
    (simple / "index.html").write_text("\n".join(root_links) + "\n")
    return packages, simple


def _run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, check=True, **kwargs)


def _freeze(python):
    return _run([str(python), "-m", "pip", "freeze", "--all"]).stdout


def _target_environment():
    tags = list(sys_tags())
    return {
        "python_full": sys.version.split()[0],
        "python_version": f"{sys.version_info.major}{sys.version_info.minor}",
        "implementation": tags[0].interpreter[:2],
        "abi": tags[0].abi,
        "abis": list(dict.fromkeys(tag.abi for tag in tags)),
        "top_tag": str(tags[0]),
        "platforms": list(dict.fromkeys(tag.platform for tag in tags)),
        "supported_tags": [str(tag) for tag in tags],
        "marker_environment": default_environment(),
    }


@contextmanager
def _isolated_pip_configuration():
    names = (
        "PIP_CONFIG_FILE",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_NO_INDEX",
        "PIP_TRUSTED_HOST",
        "PIP_FIND_LINKS",
        "PIP_DISABLE_PIP_VERSION_CHECK",
    )
    previous = {name: os.environ.get(name) for name in names}
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


class OfflineWheelhouseIntegrationTest(unittest.TestCase):
    def test_freeze_conflict_resolution_download_and_offline_install(self):
        with tempfile.TemporaryDirectory(prefix="air-wheelhouse-test-") as temporary:
            root = Path(temporary)
            packages, simple = _build_index(root / "index")
            baseline = root / "baseline"
            _run([sys.executable, "-m", "venv", str(baseline)])
            baseline_python = baseline / "bin" / "python"

            _run([
                str(baseline_python), "-m", "pip", "install", "--no-index",
                "--find-links", str(packages),
                "airlab-openai==1.0.0", "airlab-jiter==0.8.0", "airlab-array==2.1.3",
            ])

            stage = root / "stage"
            stage.mkdir()
            (stage / "constraints.txt").write_text(_freeze(baseline_python))
            (stage / "requirements.txt").write_text("airlab-openai==2.0.0\n")
            (stage / "overrides.txt").write_text("")
            (stage / "target_env.json").write_text(
                json.dumps(_target_environment(), indent=2) + "\n"
            )

            with _isolated_pip_configuration():
                manifest = resolver.resolve_and_download(
                    stage,
                    simple.as_uri(),
                    root / "worker-scratch",
                )

            self.assertTrue(manifest["ok"], json.dumps(manifest, indent=2))
            self.assertEqual(
                manifest["detected_overrides"],
                ["airlab-jiter", "airlab-openai"],
            )
            self.assertEqual(
                manifest["delta"],
                ["airlab-jiter==0.11.0", "airlab-openai==2.0.0"],
            )
            self.assertEqual(manifest["reused_from_air"], 1)

            effective = Path(manifest["constraints_for_install"]).read_text()
            self.assertIn("airlab-array==2.1.3", effective)
            self.assertNotIn("airlab-jiter==0.8.0", effective)
            self.assertNotIn("airlab-openai==1.0.0", effective)
            self.assertEqual(
                {item["file"] for item in manifest["wheel_files"]},
                {
                    "airlab_jiter-0.11.0-py3-none-any.whl",
                    "airlab_openai-2.0.0-py3-none-any.whl",
                },
            )

            offline_resolve = _run([
                str(baseline_python), "-m", "pip", "install", "--dry-run", "--no-index",
                "--find-links", manifest["wheelhouse"], "-r", str(stage / "requirements.txt"),
                "-c", manifest["constraints_for_install"],
            ])
            self.assertIn("airlab-jiter-0.11.0", offline_resolve.stdout)
            self.assertIn("airlab-openai-2.0.0", offline_resolve.stdout)

            install = _run([
                str(baseline_python), "-m", "pip", "install", "--no-index", "--no-deps",
                "--require-hashes", "--find-links", manifest["wheelhouse"],
                "-r", manifest["delta_lock"],
            ])
            self.assertIn("Successfully installed", install.stdout)
            _run([str(baseline_python), "-m", "pip", "check"])
            installed = _freeze(baseline_python)
            self.assertIn("airlab-array==2.1.3", installed)
            self.assertIn("airlab-jiter==0.11.0", installed)
            self.assertIn("airlab-openai==2.0.0", installed)
            self.assertNotIn("airlab-array==2.3.0", installed)

    def test_automatically_relaxes_hidden_incompatible_parent(self):
        with tempfile.TemporaryDirectory(prefix="air-wheelhouse-parent-test-") as temporary:
            root = Path(temporary)
            packages, simple = _build_index(root / "index")
            baseline = root / "baseline"
            _run([sys.executable, "-m", "venv", str(baseline)])
            baseline_python = baseline / "bin" / "python"

            _run([
                str(baseline_python), "-m", "pip", "install", "--no-index",
                "--find-links", str(packages),
                "airlab-parent==1.0.0", "airlab-shared==1.0.0",
            ])

            stage = root / "stage"
            stage.mkdir()
            (stage / "constraints.txt").write_text(_freeze(baseline_python))
            (stage / "requirements.txt").write_text("airlab-app==1.0.0\n")
            (stage / "overrides.txt").write_text("")
            (stage / "target_env.json").write_text(
                json.dumps(_target_environment(), indent=2) + "\n"
            )

            with _isolated_pip_configuration():
                manifest = resolver.resolve_and_download(
                    stage,
                    simple.as_uri(),
                    root / "worker-scratch",
                )

            self.assertTrue(manifest["ok"], json.dumps(manifest, indent=2))
            self.assertEqual(
                manifest["detected_overrides"],
                ["airlab-parent", "airlab-shared"],
            )
            self.assertEqual(manifest["reconciled_overrides"], ["airlab-parent"])
            self.assertGreater(manifest["reconciliation_probes"], 0)
            self.assertEqual(
                manifest["delta"],
                [
                    "airlab-app==1.0.0",
                    "airlab-parent==2.0.0",
                    "airlab-shared==2.0.0",
                ],
            )
            self.assertIn(
                "restoring the AIR pin airlab-parent==1.0.0 makes resolution fail",
                manifest["override_reasons"]["airlab-parent"],
            )

            _run([
                str(baseline_python), "-m", "pip", "install", "--dry-run", "--no-index",
                "--find-links", manifest["wheelhouse"], "-r", str(stage / "requirements.txt"),
                "-c", manifest["constraints_for_install"],
            ])

    @unittest.skipUnless(shutil.which("uv"), "uv is not installed")
    def test_uv_preferences_resolve_both_conflict_shapes_in_one_pass(self):
        with tempfile.TemporaryDirectory(prefix="air-wheelhouse-uv-test-") as temporary:
            root = Path(temporary)
            packages, _simple = _build_index(root / "index")
            baseline = root / "baseline"
            _run([sys.executable, "-m", "venv", str(baseline)])
            baseline_python = baseline / "bin" / "python"

            _run([
                str(baseline_python), "-m", "pip", "install", "--no-index",
                "--find-links", str(packages),
                "airlab-openai==1.0.0", "airlab-jiter==0.8.0", "airlab-array==2.1.3",
                "airlab-parent==1.0.0", "airlab-shared==1.0.0",
            ])

            stage = root / "stage"
            stage.mkdir()
            (stage / "constraints.txt").write_text(_freeze(baseline_python))
            (stage / "requirements.txt").write_text(
                "airlab-openai==2.0.0\nairlab-app==1.0.0\n"
            )
            (stage / "overrides.txt").write_text("")
            (stage / "target_env.json").write_text(
                json.dumps(_target_environment(), indent=2) + "\n"
            )

            with _isolated_pip_configuration():
                manifest = resolver.resolve_and_download(
                    stage,
                    work_root=root / "worker-scratch",
                    resolver_engine="uv",
                    find_links=packages,
                )

            self.assertTrue(manifest["ok"], json.dumps(manifest, indent=2))
            self.assertEqual(manifest["engine"], "uv")
            self.assertEqual(
                manifest["detected_overrides"],
                ["airlab-jiter", "airlab-openai", "airlab-parent", "airlab-shared"],
            )
            self.assertEqual(manifest["reconciliation_probes"], 0)
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
            effective = Path(manifest["constraints_for_install"]).read_text()
            self.assertIn("airlab-array==2.1.3", effective)
            self.assertNotIn("airlab-array==2.3.0", Path(manifest["resolved_lock"]).read_text())
            self.assertEqual(len(manifest["wheel_files"]), 5)

            _run([
                str(baseline_python), "-m", "pip", "install", "--dry-run", "--no-index",
                "--find-links", manifest["wheelhouse"], "-r", str(stage / "requirements.txt"),
                "-c", manifest["constraints_for_install"],
            ])


if __name__ == "__main__":
    unittest.main()
