"""Offline checks of the workspace-notebooks bundle (notebooks + AIR GPU usage dashboard).

    uv run --with databricks-sdk --with pyyaml python bundles/workspace-notebooks/tests/test_deploy_offline.py
    DATABRICKS_CLI=/path/to/databricks uv run --with databricks-sdk --with pyyaml python .../test_deploy_offline.py

- Dashboard: the committed dashboard JSON matches a fresh build from utils/billing/air_gpu_usage.sql, its parameters
  match the SQL's, and every widget field is a dataset column. Needs nothing installed.
- DeployScript: runs deploy.py (validate, deploy twice, remove) against an in-memory fake of the workspace and
  Lakeview APIs, then checks what landed where. Needs databricks-sdk and pyyaml (skipped without them).
- BundleCLI: runs a real `databricks bundle validate` / `deploy` from the same databricks.yml against the same fake
  (skipped without a CLI on PATH or DATABRICKS_CLI).

The fake implements only what these deploys touch: SCIM Me, workspace get-status / mkdirs / list / export / delete /
import, workspace-files import-file, Lakeview create / get / update / publish / trash, telemetry. Like the real
service, it stores a source file that starts with a notebook header as a notebook, without the extension. Requests it
doesn't implement are recorded and fail the test. Everything runs with an empty config file, so no real profile is
ever used.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[1]
USER = "someone@example.com"
WAREHOUSE = "fakewarehouse123"
HOME = f"/Users/{USER}/databricks-air-lab"
DASH_PATH = f"/Users/{USER}/AIR GPU usage.lvdash.json"
HEADERS = {".py": ("# Databricks notebook source", "PYTHON"), ".sql": ("-- Databricks notebook source", "SQL"),
           ".scala": ("// Databricks notebook source", "SCALA"), ".r": ("# Databricks notebook source", "R")}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("build_dashboard", BUNDLE / "build_dashboard.py")


class FakeWorkspace:
    def __init__(self) -> None:
        # path (without /Workspace) -> {"type": DIRECTORY|FILE|NOTEBOOK|DASHBOARD, "language"?, "content"?}
        self.objects: dict[str, dict] = {}
        self.dashboards: dict[str, dict] = {}   # dashboard_id -> dashboard
        self.published: dict[str, dict] = {}    # dashboard_id -> published state
        self.unhandled: list[str] = []
        self.lock = threading.Lock()
        self.mkdirs(f"/Users/{USER}")

    @staticmethod
    def norm(path: str) -> str:
        path = "/" + unquote(path).lstrip("/")
        if path == "/Workspace" or path.startswith("/Workspace/"):
            path = path[len("/Workspace"):] or "/"
        return path.rstrip("/") or "/"

    def mkdirs(self, path: str) -> None:
        parts = self.norm(path).strip("/").split("/")
        for i in range(1, len(parts) + 1):
            self.objects.setdefault("/" + "/".join(parts[:i]), {"type": "DIRECTORY"})

    def store(self, key: str, kind: str, lang: str | None, content: bytes, overwrite: bool) -> tuple[int, dict]:
        if key in self.objects and not overwrite:
            return 409, {"error_code": "RESOURCE_ALREADY_EXISTS", "message": f"{key} already exists"}
        self.mkdirs(os.path.dirname(key))
        self.objects[key] = {"type": kind, "language": lang, "content": content}
        return 200, {}

    def put(self, path: str, content: bytes, overwrite: bool) -> tuple[int, dict]:
        """workspace-files/import-file (the CLI) and import format AUTO: a header makes it a notebook."""
        path = self.norm(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in HEADERS and content.decode("utf-8", "replace").splitlines()[:1] == [HEADERS[ext][0]]:
            return self.store(path[: -len(ext)], "NOTEBOOK", HEADERS[ext][1], content, overwrite)
        return self.store(path, "FILE", None, content, overwrite)

    def import_(self, js: dict) -> tuple[int, dict]:
        """workspace/import (the SDK): SOURCE + language -> notebook at the path given; RAW -> file."""
        path, fmt = self.norm(js["path"]), js.get("format", "SOURCE")
        content = base64.b64decode(js.get("content", ""))
        overwrite = bool(js.get("overwrite"))
        if fmt == "SOURCE":
            if not js.get("language"):
                return 400, {"error_code": "INVALID_PARAMETER_VALUE", "message": "language required"}
            return self.store(path, "NOTEBOOK", js["language"], content, overwrite)
        if fmt == "RAW":
            return self.store(path, "FILE", None, content, overwrite)
        if fmt == "AUTO":
            return self.put(path, content, overwrite)
        return 400, {"error_code": "INVALID_PARAMETER_VALUE", "message": f"fake: format {fmt}"}

    def children(self, path: str) -> list[str]:
        prefix = self.norm(path).rstrip("/") + "/"
        return sorted(p for p in self.objects if p.startswith(prefix) and "/" not in p[len(prefix):])

    def delete(self, path: str, recursive: bool) -> tuple[int, dict]:
        path = self.norm(path)
        if path not in self.objects:
            return 404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": path}
        doomed = [p for p in self.objects if p == path or p.startswith(path + "/")]
        if len(doomed) > 1 and not recursive:
            return 400, {"error_code": "DIRECTORY_NOT_EMPTY", "message": path}
        for p in doomed:
            del self.objects[p]
        return 200, {}

    def status(self, path: str) -> dict:
        path = self.norm(path)
        o = self.objects[path]
        out = {"path": "/Workspace" + path, "object_type": o["type"], "object_id": abs(hash(path)) % 10**12}
        if o.get("language"):
            out["language"] = o["language"]
        if o["type"] == "DASHBOARD":
            out["resource_id"] = next((k for k, d in self.dashboards.items() if self.norm(d["path"]) == path), None)
        return out

    def lakeview(self, method: str, path: str, js: dict) -> tuple[int, dict] | None:
        m = re.fullmatch(r"/api/2\.0/lakeview/dashboards(?:/([^/]+))?(/published)?", path)
        if not m:
            return None
        did, pub = m.group(1), m.group(2)
        if did is None and method == "POST":
            did = uuid.uuid4().hex[:32]
            d = {"dashboard_id": did, "display_name": js["display_name"], "parent_path": js["parent_path"],
                 "path": f"{js['parent_path']}/{js['display_name']}.lvdash.json",
                 "serialized_dashboard": js.get("serialized_dashboard", ""), "warehouse_id": js.get("warehouse_id"),
                 "etag": "1", "lifecycle_state": "ACTIVE"}
            self.dashboards[did] = d
            self.objects[self.norm(d["path"])] = {"type": "DASHBOARD", "content": b""}
            return 200, d
        d = self.dashboards.get(did or "")
        if d is None or d["lifecycle_state"] != "ACTIVE":
            return 404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": f"dashboard {did}"}
        if pub and method == "POST":
            self.published[did] = {"display_name": d["display_name"], "warehouse_id": js.get("warehouse_id"),
                                   "embed_credentials": js.get("embed_credentials", False), "etag": d["etag"]}
            return 200, self.published[did]
        if pub and method == "GET":
            return (200, self.published[did]) if did in self.published else (404, {"error_code": "NOT_FOUND"})
        if not pub and method == "GET":
            return 200, d
        if not pub and method == "PATCH":
            for k in ("display_name", "serialized_dashboard", "warehouse_id"):
                if k in js:
                    d[k] = js[k]
            d["etag"] = str(int(d["etag"]) + 1)
            return 200, d
        if not pub and method == "DELETE":
            d["lifecycle_state"] = "TRASHED"
            self.objects.pop(self.norm(d["path"]), None)
            self.published.pop(did, None)
            return 200, {}
        return None

    def active(self) -> list[dict]:
        return [d for d in self.dashboards.values() if d["lifecycle_state"] == "ACTIVE"]

    def files_under(self, root: str) -> dict[str, tuple[str, str | None]]:
        return {p[len(root) + 1:]: (o["type"], o.get("language")) for p, o in self.objects.items()
                if p.startswith(root + "/") and o["type"] != "DIRECTORY"}


def make_handler(ws: FakeWorkspace):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, obj=None, raw: bytes | None = None) -> None:
            data = raw if raw is not None else json.dumps(obj if obj is not None else {}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _go(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            u = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            p = u.path
            js = {}
            if body and (self.headers.get("Content-Type") or "").startswith("application/json"):
                try:
                    js = json.loads(body)
                except ValueError:
                    js = {}
            with ws.lock:
                if p.endswith("/scim/v2/Me"):
                    return self._send(200, {"userName": USER, "id": "1", "displayName": "Someone"})
                if p == "/telemetry-ext":
                    return self._send(200, {})
                if p.startswith("/.well-known/") or p.startswith("/oidc/"):  # optional discovery
                    return self._send(404, {"error_code": "NOT_FOUND", "message": p})
                out = ws.lakeview(self.command, p, js) if p.startswith("/api/2.0/lakeview/") else None
                if out:
                    return self._send(*out)
                if p == "/api/2.0/workspace/get-status":
                    path = ws.norm(q.get("path", ""))
                    if path not in ws.objects:
                        return self._send(404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": path})
                    return self._send(200, ws.status(path))
                if p == "/api/2.0/workspace/mkdirs":
                    ws.mkdirs(js["path"])
                    return self._send(200, {})
                if p == "/api/2.0/workspace/import":
                    return self._send(*ws.import_(js))
                if p.startswith("/api/2.0/workspace-files/import-file/"):
                    return self._send(*ws.put(p[len("/api/2.0/workspace-files/import-file/"):], body,
                                              q.get("overwrite", "false") == "true"))
                if p == "/api/2.0/workspace/list":
                    path = ws.norm(q.get("path", ""))
                    if path not in ws.objects:
                        return self._send(404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": path})
                    return self._send(200, {"objects": [ws.status(c) for c in ws.children(path)]})
                if p == "/api/2.0/workspace/export":
                    path = ws.norm(q.get("path", ""))
                    o = ws.objects.get(path)
                    if not o or o["type"] == "DIRECTORY":
                        return self._send(404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": path})
                    if q.get("direct_download") == "true":
                        return self._send(200, raw=o["content"])
                    return self._send(200, {"content": base64.b64encode(o["content"]).decode()})
                if p == "/api/2.0/workspace/delete":
                    return self._send(*ws.delete(js["path"], bool(js.get("recursive"))))
                ws.unhandled.append(f"{self.command} {self.path}")
                return self._send(501, {"error_code": "NOT_IMPLEMENTED", "message": f"fake: {self.command} {p}"})

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _go

        def log_message(self, *args) -> None:
            pass

    return H


class Dashboard(unittest.TestCase):
    """Checks that need nothing installed."""

    def test_json_is_fresh(self) -> None:
        committed = (BUNDLE / "src" / "air_gpu_usage.lvdash.json").read_text()
        self.assertEqual(committed, build.render(), "dashboard JSON is stale: run build_dashboard.py")

    def test_parameters_match_sql(self) -> None:
        sql = build.SQL.read_text()
        used = set(re.findall(r"(?<![:\w]):([A-Za-z_]\w*)", sql))
        declared = {k for k, *_ in build.PARAMS}
        self.assertEqual(used, declared)

    def test_widgets_reference_dataset_columns(self) -> None:
        d = build.build()
        # the final SELECT's output columns: `AS name` aliases plus bare column names
        final = build.SQL.read_text().rsplit("\nSELECT\n", 1)[1].split("\nFROM unit_days")[0]
        cols = set(re.findall(r"\bAS (\w+)\s*,?\s*$", final, re.M)) | set(re.findall(r"^\s+(\w+),$", final, re.M))
        for page in d["pages"]:
            for item in page["layout"]:
                for q in item["widget"].get("queries", []):
                    for f in q["query"].get("fields", []):
                        for ref in re.findall(r"`(\w+)`", f["expression"]):
                            if ref != "associative_filter_predicate_group":
                                self.assertIn(ref, cols, f"{item['widget']['name']}: no column {ref}")
                pos = item["position"]   # widgets stay on the 6-column grid
                self.assertLessEqual(pos["x"] + pos["width"], 6, item["widget"]["name"])


class Offline(unittest.TestCase):
    """A fresh fake workspace per test, and an environment that can't reach a real profile."""

    def setUp(self) -> None:
        self.ws = FakeWorkspace()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.ws))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.tmp = Path(tempfile.mkdtemp(prefix="bundle-offline-"))
        (self.tmp / "databrickscfg").write_text("[DEFAULT]\n")
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("DATABRICKS_", "BUNDLE_VAR_")) and k != "BUNDLE_ROOT"}
        self.env.update(DATABRICKS_CONFIG_FILE=str(self.tmp / "databrickscfg"),
                        DATABRICKS_HOST=f"http://127.0.0.1:{self.server.server_address[1]}",
                        DATABRICKS_TOKEN="fake-token")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(BUNDLE / ".databricks", ignore_errors=True)   # the CLI's local sync state

    def check_deployed(self) -> None:
        files = self.ws.files_under(HOME)
        self.assertEqual(files.get("utils/billing/reserved_gpu_attribution"), ("NOTEBOOK", "SQL"))
        self.assertEqual(files.get("utils/billing/air_sessions_probe"), ("NOTEBOOK", "PYTHON"))
        self.assertEqual(files.get("utils/billing/air_gpu_usage.sql"), ("FILE", None))
        self.assertEqual(files.get("utils/billing/queries.py"), ("FILE", None))
        # Only the listed folder, nothing gitignored, no bundle config.
        self.assertTrue(all(p.startswith("utils/billing/") for p in files), sorted(files))
        self.assertFalse(any("__pycache__" in p or p.startswith("docs/") for p in files), sorted(files))
        dashboards = self.ws.active()
        self.assertEqual(len(dashboards), 1, dashboards)
        d = dashboards[0]
        self.assertEqual(self.ws.norm(d["path"]), DASH_PATH)
        self.assertEqual(d["warehouse_id"], WAREHOUSE)
        self.assertEqual(json.loads(d["serialized_dashboard"]), build.build())
        self.assertEqual(self.ws.published[d["dashboard_id"]]["embed_credentials"], False)


class DeployScript(Offline):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import databricks.sdk  # noqa: F401
            import yaml  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("needs databricks-sdk and pyyaml (uv run --with databricks-sdk --with pyyaml)")

    def run_script(self, *args: str, stdin: str = "") -> subprocess.CompletedProcess:
        r = subprocess.run([sys.executable, str(BUNDLE / "deploy.py"), *args], env=self.env, input=stdin,
                           capture_output=True, text=True, timeout=120, cwd=self.tmp)
        self.assertEqual(self.ws.unhandled, [], f"deploy.py called endpoints the fake lacks:\n{r.stderr}")
        return r

    def ok(self, *args: str, stdin: str = "") -> str:
        r = self.run_script(*args, stdin=stdin)
        self.assertEqual(r.returncode, 0, f"deploy.py {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    def test_validate_writes_nothing(self) -> None:
        before = dict(self.ws.objects)
        out = self.ok("validate")
        self.assertIn(f"notebooks: /Workspace{HOME}", out)
        self.assertIn("NOT SET", out)                       # no warehouse: says deploy would skip the dashboard
        self.assertIn("utils/billing/reserved_gpu_attribution", out)
        self.assertEqual(self.ws.objects, before)

    def test_deploy_twice_then_remove(self) -> None:
        out = self.ok("deploy", "--var", f"warehouse_id={WAREHOUSE}")
        self.assertIn(f"Notebooks are in /Workspace{HOME}", out)
        self.assertIn("Dashboard published:", out)
        self.check_deployed()
        first = self.ws.active()[0]["dashboard_id"]
        # A file deleted from the checkout is removed from the workspace on the next deploy.
        self.ws.store(f"{HOME}/utils/billing/gone_from_repo", "NOTEBOOK", "SQL", b"", True)
        self.env["BUNDLE_VAR_warehouse_id"] = WAREHOUSE   # the environment works instead of --var
        self.ok("deploy")
        self.assertNotIn(f"{HOME}/utils/billing/gone_from_repo", self.ws.objects)
        self.assertEqual([d["dashboard_id"] for d in self.ws.active()], [first])   # updated in place
        self.check_deployed()
        # Without --yes and a "n" answer, nothing is deleted.
        r = self.run_script("remove", stdin="n\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn(HOME, self.ws.objects)
        self.ok("remove", "--yes")
        self.assertEqual(self.ws.files_under(HOME), {})
        self.assertNotIn(HOME, self.ws.objects)
        self.assertEqual(self.ws.active(), [])

    def test_deploy_without_warehouse_skips_dashboard(self) -> None:
        out = self.ok("deploy")
        self.assertIn("skipped", out)
        self.assertEqual(self.ws.active(), [])
        self.assertEqual(self.ws.files_under(HOME).get("utils/billing/reserved_gpu_attribution"), ("NOTEBOOK", "SQL"))

    def test_updates_a_dashboard_deployed_by_the_cli(self) -> None:
        # One created earlier (e.g. by `databricks bundle deploy`) is found by its path and updated, not duplicated.
        with self.ws.lock:
            code, d = self.ws.lakeview("POST", "/api/2.0/lakeview/dashboards",
                                       {"display_name": "AIR GPU usage", "parent_path": f"/Users/{USER}"})
        self.assertEqual(code, 200)
        self.ok("deploy", "--var", f"warehouse_id={WAREHOUSE}")
        self.assertEqual([x["dashboard_id"] for x in self.ws.active()], [d["dashboard_id"]])

    def test_bad_variable(self) -> None:
        r = self.run_script("validate", "--var", "nope=1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown variable", r.stderr)


class BundleCLI(Offline):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = os.environ.get("DATABRICKS_CLI") or shutil.which("databricks")
        if not cls.cli:
            raise unittest.SkipTest("no databricks CLI (set DATABRICKS_CLI)")

    def bundle(self, *args: str) -> subprocess.CompletedProcess:
        r = subprocess.run([self.cli, "bundle", *args], cwd=BUNDLE, env=self.env,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(self.ws.unhandled, [], f"CLI called endpoints the fake lacks:\n{r.stderr}")
        return r

    def test_cli_needs_warehouse(self) -> None:
        r = self.bundle("validate")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("warehouse_id", r.stdout + r.stderr)

    def test_cli_deploy(self) -> None:
        for args in (("validate",), ("deploy",), ("deploy",)):   # a second deploy is a no-op, not an error
            r = self.bundle(*args, "--var", f"warehouse_id={WAREHOUSE}")
            self.assertEqual(r.returncode, 0, f"bundle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        self.check_deployed()


if __name__ == "__main__":
    unittest.main(verbosity=2)
