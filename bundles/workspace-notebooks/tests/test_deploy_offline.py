"""Offline check of the workspace-notebooks bundle: run a real `databricks bundle deploy` against an
in-memory fake of the workspace API, then assert what landed where.

    python3 bundles/workspace-notebooks/tests/test_deploy_offline.py          # CLI on PATH
    DATABRICKS_CLI=/path/to/databricks python3 .../test_deploy_offline.py     # a specific CLI

The fake implements only what a files-only deploy touches (SCIM Me, workspace get-status / mkdirs /
list / export / delete, workspace-files import-file, telemetry). Like the real service, it stores an
uploaded source file that starts with a notebook header as a notebook, without the extension.
Requests it doesn't implement are recorded and fail the test, so a CLI that needs more shows up
here rather than in a customer workspace. The CLI runs with an empty config file, so no real
profile (or `default_profile`) is ever used. Stdlib only.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

BUNDLE = Path(__file__).resolve().parents[1]
USER = "someone@example.com"
HEADERS = {".py": ("# Databricks notebook source", "PYTHON"), ".sql": ("-- Databricks notebook source", "SQL"),
           ".scala": ("// Databricks notebook source", "SCALA"), ".r": ("# Databricks notebook source", "R")}


class FakeWorkspace:
    def __init__(self) -> None:
        # path (without /Workspace) -> {"type": DIRECTORY|FILE|NOTEBOOK, "language"?, "content"?}
        self.objects: dict[str, dict] = {}
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

    def put(self, path: str, content: bytes, overwrite: bool) -> tuple[int, dict]:
        path = self.norm(path)
        ext = os.path.splitext(path)[1].lower()
        kind, lang, key = "FILE", None, path
        if ext in HEADERS and content.decode("utf-8", "replace").splitlines()[:1] == [HEADERS[ext][0]]:
            kind, lang, key = "NOTEBOOK", HEADERS[ext][1], path[: -len(ext)]
        if key in self.objects and not overwrite:
            return 409, {"error_code": "RESOURCE_ALREADY_EXISTS", "message": f"{key} already exists"}
        self.mkdirs(os.path.dirname(key))
        self.objects[key] = {"type": kind, "language": lang, "content": content}
        return 200, {}

    def children(self, path: str) -> list[str]:
        path = self.norm(path)
        prefix = path.rstrip("/") + "/"
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
        return out


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
                if p.startswith("/.well-known/"):   # optional host metadata; the CLI carries on without it
                    return self._send(404, {"error_code": "NOT_FOUND", "message": p})
                if p == "/api/2.0/workspace/get-status":
                    path = ws.norm(q.get("path", ""))
                    if path not in ws.objects:
                        return self._send(404, {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": path})
                    return self._send(200, ws.status(path))
                if p == "/api/2.0/workspace/mkdirs":
                    ws.mkdirs(js["path"])
                    return self._send(200, {})
                if p.startswith("/api/2.0/workspace-files/import-file/"):
                    code, out = ws.put(p[len("/api/2.0/workspace-files/import-file/"):], body,
                                       q.get("overwrite", "false") == "true")
                    return self._send(code, out)
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
                    code, out = ws.delete(js["path"], bool(js.get("recursive")))
                    return self._send(code, out)
                ws.unhandled.append(f"{self.command} {self.path}")
                return self._send(501, {"error_code": "NOT_IMPLEMENTED", "message": f"fake: {self.command} {p}"})

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _go

        def log_message(self, *args) -> None:
            pass

    return H


class DeployOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = os.environ.get("DATABRICKS_CLI") or shutil.which("databricks")
        if not cls.cli:
            raise unittest.SkipTest("no databricks CLI (set DATABRICKS_CLI)")
        cls.ws = FakeWorkspace()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.ws))
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.tmp = Path(tempfile.mkdtemp(prefix="bundle-offline-"))
        (cls.tmp / "databrickscfg").write_text("[DEFAULT]\n")
        cls.env = {k: v for k, v in os.environ.items()
                   if not k.startswith("DATABRICKS_") and k not in ("BUNDLE_ROOT",)}
        cls.env.update(DATABRICKS_CONFIG_FILE=str(cls.tmp / "databrickscfg"),
                       DATABRICKS_HOST=f"http://127.0.0.1:{cls.server.server_address[1]}",
                       DATABRICKS_TOKEN="fake-token")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        # The CLI keeps local sync state next to the bundle; don't leave it in the checkout.
        shutil.rmtree(BUNDLE / ".databricks", ignore_errors=True)

    def bundle(self, *args: str) -> subprocess.CompletedProcess:
        r = subprocess.run([self.cli, "bundle", *args], cwd=BUNDLE, env=self.env,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(self.ws.unhandled, [], f"CLI called endpoints the fake lacks:\n{r.stderr}")
        self.assertEqual(r.returncode, 0, f"bundle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
        return r

    def test_deploy(self) -> None:
        self.bundle("validate")
        self.bundle("deploy")
        self.bundle("deploy")   # a second deploy is a no-op, not an error

        home = f"/Users/{USER}/databricks-air-lab"
        objs = {p[len(home) + 1:]: o for p, o in self.ws.objects.items() if p.startswith(home + "/")}
        files = {p: (o["type"], o.get("language")) for p, o in objs.items() if o["type"] != "DIRECTORY"}

        self.assertEqual(files.get("utils/billing/reserved_gpu_attribution"), ("NOTEBOOK", "SQL"))
        self.assertEqual(files.get("utils/billing/reserved_gpu_attribution_pyspark"), ("NOTEBOOK", "PYTHON"))
        self.assertEqual(files.get("utils/billing/air_usage.sql"), ("FILE", None))
        self.assertEqual(files.get("utils/billing/queries.py"), ("FILE", None))
        # Only the listed folder, nothing gitignored, no bundle config.
        self.assertTrue(all(p.startswith("utils/billing/") for p in files), sorted(files))
        self.assertFalse(any("__pycache__" in p or p.startswith("docs/") for p in files), sorted(files))
        # Deploy state lives in the hidden bundle folder, not next to the notebooks.
        self.assertTrue(any(p.startswith(f"/Users/{USER}/.bundle/air-lab-notebooks/user/")
                            for p in self.ws.objects))
        print("\ndeployed under /Workspace" + home + ":")
        for p, (kind, lang) in sorted(files.items()):
            print(f"  {kind:8} {lang or '':6} {p}")

    def test_manage_deploy_then_remove(self) -> None:
        env = dict(self.env, DATABRICKS_CLI=self.cli)
        manage = [str(BUNDLE / "manage.sh")]
        r = subprocess.run(manage + ["deploy"], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"Notebooks are in /Workspace/Users/{USER}/databricks-air-lab", r.stdout)
        # Without --yes and a "n" answer, nothing is deleted.
        r = subprocess.run(manage + ["remove"], env=env, input="n\n", capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 1)
        self.assertIn(f"/Users/{USER}/databricks-air-lab", self.ws.objects)
        r = subprocess.run(manage + ["remove", "--yes"], env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.ws.unhandled, [])
        left = [p for p in self.ws.objects if p.startswith(f"/Users/{USER}/databricks-air-lab")
                or p.startswith(f"/Users/{USER}/.bundle/air-lab-notebooks")]
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
