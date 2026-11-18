"""Offline check of the air-usage-dashboard bundle: a real `databricks bundle deploy` against an in-memory fake of
the workspace and Lakeview APIs, then assertions on the dashboard that was created and published.

    python3 bundles/air-usage-dashboard/tests/test_deploy_offline.py          # CLI on PATH
    DATABRICKS_CLI=/path/to/databricks python3 .../test_deploy_offline.py     # a specific CLI

Reuses the workspace fake from bundles/workspace-notebooks/tests and adds the Lakeview calls a dashboard deploy makes
(create, get, update, publish, trash). Requests neither implements are recorded and fail the test. The CLI runs with
an empty config file, so no real profile is used. It also checks the committed dashboard JSON matches a fresh
build from utils/billing/air_gpu_usage.sql, and that the dataset's parameters match the SQL's. Stdlib only.
"""
from __future__ import annotations

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
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

BUNDLE = Path(__file__).resolve().parents[1]
REPO = BUNDLE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load("nb_bundle_offline", REPO / "bundles" / "workspace-notebooks" / "tests" / "test_deploy_offline.py")
build = _load("build_dashboard", BUNDLE / "build_dashboard.py")
USER = base.USER
WAREHOUSE = "fakewarehouse123"


class FakeLakeview(base.FakeWorkspace):
    def __init__(self) -> None:
        super().__init__()
        self.dashboards: dict[str, dict] = {}   # dashboard_id -> dashboard
        self.published: dict[str, dict] = {}    # dashboard_id -> published state

    def route(self, method: str, path: str, js: dict) -> tuple[int, dict] | None:
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

    def status(self, path: str) -> dict:
        out = super().status(path)
        if out["object_type"] == "DASHBOARD":
            did = next((k for k, d in self.dashboards.items() if self.norm(d["path"]) == self.norm(path)), None)
            out["resource_id"] = did
        return out


def make_handler(ws: FakeLakeview):
    Base = base.make_handler(ws)

    class H(Base):
        def _go(self) -> None:
            u = urlsplit(self.path)
            if u.path.startswith("/api/2.0/lakeview/"):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                js = json.loads(body) if body else {}
                with ws.lock:
                    out = ws.route(self.command, u.path, js)
                    if out is None:
                        ws.unhandled.append(f"{self.command} {self.path}")
                        out = (501, {"error_code": "NOT_IMPLEMENTED", "message": f"fake: {self.command} {u.path}"})
                return self._send(*out)
            return Base._go(self)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _go

    return H


class Dashboard(unittest.TestCase):
    """Checks that need no CLI."""

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
                # widgets stay on the 6-column grid
                pos = item["position"]
                self.assertLessEqual(pos["x"] + pos["width"], 6, item["widget"]["name"])


class DeployOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = os.environ.get("DATABRICKS_CLI") or shutil.which("databricks")
        if not cls.cli:
            raise unittest.SkipTest("no databricks CLI (set DATABRICKS_CLI)")
        cls.ws = FakeLakeview()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.ws))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.tmp = Path(tempfile.mkdtemp(prefix="bundle-offline-"))
        (cls.tmp / "databrickscfg").write_text("[DEFAULT]\n")
        cls.env = {k: v for k, v in os.environ.items()
                   if not k.startswith("DATABRICKS_") and not k.startswith("BUNDLE_")}
        cls.env.update(DATABRICKS_CONFIG_FILE=str(cls.tmp / "databrickscfg"),
                       DATABRICKS_HOST=f"http://127.0.0.1:{cls.server.server_address[1]}",
                       DATABRICKS_TOKEN="fake-token")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(BUNDLE / ".databricks", ignore_errors=True)

    def run_cli(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([self.cli, "bundle", *args], cwd=BUNDLE, env=env or self.env,
                              capture_output=True, text=True, timeout=180)

    def bundle(self, *args: str) -> subprocess.CompletedProcess:
        r = self.run_cli(*args, "--var", f"warehouse_id={WAREHOUSE}")
        self.assertEqual(self.ws.unhandled, [], f"CLI called endpoints the fake lacks:\n{r.stderr}")
        self.assertEqual(r.returncode, 0, f"bundle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r

    def active(self) -> list[dict]:
        return [d for d in self.ws.dashboards.values() if d["lifecycle_state"] == "ACTIVE"]

    def test_1_needs_warehouse(self) -> None:
        r = self.run_cli("validate")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("warehouse_id", r.stdout + r.stderr)

    def test_2_deploy_publish_remove(self) -> None:
        self.bundle("validate")
        self.bundle("deploy")
        self.bundle("deploy")   # a second deploy updates in place, not a second dashboard

        dashboards = self.active()
        self.assertEqual(len(dashboards), 1, dashboards)
        d = dashboards[0]
        self.assertEqual(d["display_name"], "AIR GPU usage")
        self.assertEqual(base.FakeWorkspace.norm(d["parent_path"]), f"/Users/{USER}")
        self.assertEqual(d["warehouse_id"], WAREHOUSE)
        self.assertEqual(json.loads(d["serialized_dashboard"]),
                         json.loads((BUNDLE / "src" / "air_gpu_usage.lvdash.json").read_text()))
        pub = self.ws.published.get(d["dashboard_id"])
        self.assertIsNotNone(pub, "dashboard was not published")
        self.assertFalse(pub["embed_credentials"])
        # Nothing but the dashboard is uploaded; deploy state lives in the hidden bundle folder.
        home = f"/Users/{USER}/"
        files = [p for p, o in self.ws.objects.items() if o["type"] in ("FILE", "NOTEBOOK")
                 and p.startswith(home) and "/.bundle/air-lab-usage-dashboard/user/state/" not in p]
        self.assertFalse([p for p in files if "/.bundle/air-lab-usage-dashboard/user/files/" in p], files)
        print(f"\ndeployed {d['path']} (dashboard {d['dashboard_id']}), published")

        env = dict(self.env, DATABRICKS_CLI=self.cli, BUNDLE_VAR_warehouse_id=WAREHOUSE)
        manage = [str(BUNDLE / "manage.sh")]
        r = subprocess.run(manage + ["remove"], env=env, input="n\n", capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.active()), 1)
        r = subprocess.run(manage + ["remove", "--yes"], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.ws.unhandled, [])
        self.assertEqual(self.active(), [])

    def test_3_manage_deploy(self) -> None:
        env = dict(self.env, DATABRICKS_CLI=self.cli)
        r = subprocess.run([str(BUNDLE / "manage.sh"), "deploy", "--var", f"warehouse_id={WAREHOUSE}"],
                           env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("Dashboard deployed:", r.stdout)
        self.assertEqual(self.ws.unhandled, [])
        self.assertEqual(len(self.active()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
