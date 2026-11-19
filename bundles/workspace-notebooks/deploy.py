#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["databricks-sdk>=0.50", "pyyaml>=6"]
# ///
"""Deploy this repo's workspace notebooks and the AIR GPU usage dashboard with the Databricks SDK.

    uv run bundles/workspace-notebooks/deploy.py deploy   -p <profile> --var warehouse_id=<id>
    uv run bundles/workspace-notebooks/deploy.py validate -p <profile> --var warehouse_id=<id>
    uv run bundles/workspace-notebooks/deploy.py remove   -p <profile> [--yes]

The same commands and flags as the bundle (`--var name=value` or BUNDLE_VAR_<name>; `destroy` = `remove`), read
from the same databricks.yml, so `databricks bundle deploy` and this script deploy the same thing. No Databricks CLI
needed: auth is the SDK's (the profile's host and credentials). Without -p the SDK falls back to DATABRICKS_* env
vars, then [DEFAULT] in ~/.databrickscfg: pass -p to be sure which workspace you hit.

- deploy: uploads every file under `sync.paths` (git-tracked or untracked-but-not-ignored, so .gitignore applies)
  to `workspace.file_path`, keeping repo-relative paths. Source-format notebooks become notebooks (extension
  dropped); everything else becomes a workspace file. Then deletes files under those folders that are no longer in
  the checkout. Creates or updates each dashboard resource (found by its path, so one deployed by
  `databricks bundle deploy` is updated in place) and publishes it. Without a warehouse ID it skips the dashboard.
- validate: shows what deploy would do. Reads the workspace (current user, dashboard lookup); writes nothing.
- remove: deletes the notebooks folder and trashes the dashboard, plus any `databricks bundle` deploy state left
  under ~/.bundle/ by earlier CLI deploys (asks first; --yes skips the prompt).
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "databricks.yml"
# Extension -> (first line that marks a source-format notebook, notebook language)
HEADERS = {".py": ("# Databricks notebook source", "PYTHON"), ".sql": ("-- Databricks notebook source", "SQL"),
           ".scala": ("// Databricks notebook source", "SCALA"), ".r": ("# Databricks notebook source", "R")}
SKIP_DIRS = {"__pycache__", ".databricks", ".git"}
# deploy state of the two bundles this one replaced, from earlier `databricks bundle` / manage.sh deploys
LEGACY_STATE = ("air-lab-notebooks", "air-lab-usage-dashboard")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Upload:
    local: Path
    remote: str             # workspace path, without /Workspace
    kind: str               # NOTEBOOK | FILE
    language: str | None


@dataclass(frozen=True)
class DashboardSpec:
    key: str
    display_name: str
    file: Path
    parent_path: str        # without /Workspace
    warehouse_id: str | None
    embed_credentials: bool

    @property
    def path(self) -> str:
        return f"{self.parent_path}/{self.display_name}.lvdash.json"


@dataclass(frozen=True)
class Plan:
    user: str
    root: str               # notebooks folder, without /Workspace
    sync_dirs: list[str]    # remote folders mirrored from sync.paths (stale files are deleted under these)
    uploads: list[Upload]
    dashboards: list[DashboardSpec]


def ws_path(path: str) -> str:
    """Workspace path without the /Workspace prefix, no trailing slash (the form the API returns)."""
    path = "/" + path.strip().lstrip("/")
    if path == "/Workspace" or path.startswith("/Workspace/"):
        path = path[len("/Workspace"):] or "/"
    return path.rstrip("/") or "/"


def parse_vars(pairs: list[str]) -> dict[str, str]:
    out = {k[len("BUNDLE_VAR_"):]: v for k, v in os.environ.items() if k.startswith("BUNDLE_VAR_")}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise ConfigError(f"--var takes name=value, got {pair!r}")
        out[name] = value
    return out


def interpolate(value, ctx: dict[str, str]):
    """Substitute ${...} references (the subset this bundle uses: workspace.current_user.userName and var.*)."""
    if not isinstance(value, str):
        return value

    def sub(m: re.Match) -> str:
        ref = m.group(1)
        if ref not in ctx:
            raise ConfigError(f"databricks.yml: no value for ${{{ref}}}"
                              + (f" (pass --var {ref[4:]}=<value>)" if ref.startswith("var.") else ""))
        return ctx[ref]

    return re.sub(r"\$\{([^}]+)\}", sub, value)


def load_config(path: Path = CONFIG) -> dict:
    import yaml  # here, so --help works without the dependency

    with open(path) as f:
        return yaml.safe_load(f)


def resolve_context(cfg: dict, user: str, cli_vars: dict[str, str], target: str) -> dict[str, str]:
    targets = cfg.get("targets") or {}
    if targets and target not in targets:
        raise ConfigError(f"no target {target!r} in databricks.yml (have: {', '.join(targets)})")
    ctx = {"workspace.current_user.userName": user}
    for name, spec in (cfg.get("variables") or {}).items():
        value = cli_vars.get(name, (spec or {}).get("default"))
        if value not in (None, ""):
            ctx[f"var.{name}"] = str(value)
    unknown = set(cli_vars) - set(cfg.get("variables") or {})
    if unknown:
        raise ConfigError(f"unknown variable(s): {', '.join(sorted(unknown))}")
    return ctx


def sync_root(bundle_dir: Path, paths: list[Path]) -> Path:
    """Like the bundle: the common ancestor of the bundle directory and every sync path."""
    return Path(os.path.commonpath([str(bundle_dir), *(str(p) for p in paths)]))


def list_files(path: Path) -> list[Path]:
    """Files under `path` that git tracks or would add (.gitignore applies); a plain walk outside a git checkout."""
    if path.is_file():
        return [path]
    try:
        out = subprocess.run(["git", "-C", str(path), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard", "--", "."], capture_output=True, check=True).stdout
        files = [path / f for f in out.decode().split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for d, dirs, fs in os.walk(path):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            files += [Path(d) / f for f in fs]
    return sorted(f for f in files if f.is_file() and not SKIP_DIRS.intersection(f.relative_to(path).parts))


def classify(local: Path, remote: str) -> Upload:
    ext = local.suffix.lower()
    if ext in HEADERS:
        with open(local, encoding="utf-8", errors="replace") as f:
            if f.readline().rstrip("\r\n") == HEADERS[ext][0]:
                return Upload(local, remote[: -len(local.suffix)], "NOTEBOOK", HEADERS[ext][1])
    return Upload(local, remote, "FILE", None)


def build_plan(cfg: dict, user: str, cli_vars: dict[str, str], target: str = "user",
               bundle_dir: Path = HERE) -> Plan:
    ctx = resolve_context(cfg, user, cli_vars, target)
    root = ws_path(interpolate((cfg.get("workspace") or {}).get("file_path")
                               or f"/Workspace/Users/{user}/.bundle/{cfg['bundle']['name']}/{target}/files", ctx))
    paths = [(bundle_dir / p).resolve() for p in ((cfg.get("sync") or {}).get("paths") or ["."])]
    for p in paths:
        if not p.exists():
            raise ConfigError(f"sync path {p} doesn't exist")
    base = sync_root(bundle_dir, paths)
    uploads, sync_dirs = [], []
    for p in paths:
        rel = p.relative_to(base).as_posix()
        sync_dirs.append(root if rel == "." else f"{root}/{rel}")
        for f in list_files(p):
            uploads.append(classify(f, f"{root}/{f.relative_to(base).as_posix()}"))
    dashboards = []
    for key, d in ((cfg.get("resources") or {}).get("dashboards") or {}).items():
        try:
            warehouse = interpolate(d.get("warehouse_id"), ctx)
        except ConfigError:
            warehouse = None   # deploy skips the dashboard; validate says why
        dashboards.append(DashboardSpec(
            key=key, display_name=interpolate(d["display_name"], ctx), file=(bundle_dir / d["file_path"]).resolve(),
            parent_path=ws_path(interpolate(d.get("parent_path") or f"/Workspace/Users/{user}", ctx)),
            warehouse_id=warehouse or None, embed_credentials=bool(d.get("embed_credentials", False))))
    return Plan(user, root, sorted(set(sync_dirs)), uploads, dashboards)


# ---------------------------------------------------------------------------------------------- workspace calls


def client(profile: str | None):
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient(profile=profile) if profile else WorkspaceClient()


def not_found():
    from databricks.sdk.errors import NotFound

    return NotFound


def walk_remote(w, path: str):
    """Every object under `path` (recursive), as (path without /Workspace, object type). Nothing if it's absent."""
    try:
        objs = list(w.workspace.list(path))
    except not_found():
        return
    for o in objs:
        kind = o.object_type.value if hasattr(o.object_type, "value") else str(o.object_type)
        p = ws_path(o.path)
        yield p, kind
        if kind == "DIRECTORY":
            yield from walk_remote(w, p)


def dashboard_id(w, d: DashboardSpec) -> str | None:
    try:
        return w.workspace.get_status(d.path).resource_id
    except not_found():
        return None


def deploy(w, plan: Plan) -> list[str]:
    from databricks.sdk.service.dashboards import Dashboard
    from databricks.sdk.service.workspace import ImportFormat, Language

    for d in sorted({u.remote.rsplit("/", 1)[0] for u in plan.uploads}):
        w.workspace.mkdirs(d)
    for u in plan.uploads:
        content = base64.b64encode(u.local.read_bytes()).decode()
        if u.kind == "NOTEBOOK":
            w.workspace.import_(u.remote, content=content, format=ImportFormat.SOURCE,
                                language=Language(u.language), overwrite=True)
        else:
            w.workspace.import_(u.remote, content=content, format=ImportFormat.RAW, overwrite=True)
    wanted = {u.remote for u in plan.uploads}
    for top in plan.sync_dirs:
        for p, kind in list(walk_remote(w, top)):
            if kind != "DIRECTORY" and p not in wanted:
                w.workspace.delete(p)
    out = [f"Notebooks are in /Workspace{plan.root} ({len(plan.uploads)} files)"]
    for d in plan.dashboards:
        if not d.warehouse_id:
            out.append(f"Dashboard {d.display_name!r} skipped: pass --var warehouse_id=<id>")
            continue
        body = Dashboard(display_name=d.display_name, serialized_dashboard=d.file.read_text(),
                         warehouse_id=d.warehouse_id)
        did = dashboard_id(w, d)
        if did:
            w.lakeview.update(did, body)
        else:
            body.parent_path = d.parent_path
            did = w.lakeview.create(body).dashboard_id
        w.lakeview.publish(did, embed_credentials=d.embed_credentials, warehouse_id=d.warehouse_id)
        out.append(f"Dashboard published: {w.config.host.rstrip('/')}/dashboardsv3/{did}/published")
    return out


def validate(w, plan: Plan) -> list[str]:
    notebooks = sum(u.kind == "NOTEBOOK" for u in plan.uploads)
    out = [f"user:      {plan.user}", f"host:      {w.config.host}",
           f"notebooks: /Workspace{plan.root} ({notebooks} notebooks, {len(plan.uploads) - notebooks} files)"]
    out += [f"  {u.kind:8} {u.language or '':6} {u.remote[len(plan.root) + 1:]}" for u in plan.uploads]
    for d in plan.dashboards:
        state = "updated" if dashboard_id(w, d) else "created"
        warehouse = d.warehouse_id or "NOT SET (deploy skips it: pass --var warehouse_id=<id>)"
        out += [f"dashboard: /Workspace{d.path} (would be {state} and published)", f"  warehouse: {warehouse}",
                f"  embed_credentials: {str(d.embed_credentials).lower()}"]
    return out


def remove(w, plan: Plan, yes: bool) -> tuple[int, list[str]]:
    legacy = [f"/Users/{plan.user}/.bundle/{name}" for name in LEGACY_STATE]
    targets = [plan.root] + legacy
    dashes = [(d, dashboard_id(w, d)) for d in plan.dashboards]
    if not yes:
        what = [f"/Workspace{plan.root} and everything in it"]
        what += [f"the dashboard /Workspace{d.path}" for d, did in dashes if did]
        print("Delete " + ", ".join(what) + ", plus any old bundle deploy state? [y/N] ", end="", flush=True)
        if sys.stdin.readline().strip().lower() != "y":
            return 1, ["Nothing deleted."]
    out = []
    for path in targets:
        try:
            w.workspace.delete(path, recursive=True)
            out.append(f"Deleted /Workspace{path}")
        except not_found():
            pass
    for d, did in dashes:
        if did:
            w.lakeview.trash(did)
            out.append(f"Trashed dashboard /Workspace{d.path}")
    return 0, out or ["Nothing to delete."]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deploy.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["deploy", "validate", "remove", "destroy"])
    ap.add_argument("-p", "--profile", help="~/.databrickscfg profile (host and credentials)")
    ap.add_argument("-t", "--target", default="user", help="bundle target (only 'user' exists)")
    ap.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="bundle variable, repeatable")
    ap.add_argument("--yes", action="store_true", help="remove: don't ask")
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
        cli_vars = parse_vars(args.var)
        w = client(args.profile)
        plan = build_plan(cfg, w.current_user.me().user_name, cli_vars, args.target)
        if args.command == "deploy":
            code, out = 0, deploy(w, plan)
        elif args.command == "validate":
            code, out = 0, validate(w, plan)
        else:
            code, out = remove(w, plan, args.yes)
    except ConfigError as e:
        print(f"deploy.py: {e}", file=sys.stderr)
        return 2
    print("\n".join(out))
    return code


if __name__ == "__main__":
    sys.exit(main())
