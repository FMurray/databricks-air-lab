"""Persisted UAT prefs — wheels root + per-workload wheel lists.

Stored at ~/.air-lab/config.json (override with $AIR_LAB_CONFIG). Stdlib-only so
uat_min / uat_core can use it on no-PyPI workspaces. Wheel injection into AIR YAML
only happens when both wheels_root and a non-empty workload list are set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def config_path() -> Path:
    override = os.environ.get("AIR_LAB_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".air-lab" / "config.json"


def load() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save(data: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def get_wheels_root(data: dict[str, Any] | None = None) -> str | None:
    cfg = load() if data is None else data
    root = cfg.get("wheels_root")
    if not root or not isinstance(root, str):
        return None
    root = root.strip()
    return root or None


def set_wheels_root(path: str | None) -> Path:
    cfg = load()
    if path is None or not str(path).strip():
        cfg.pop("wheels_root", None)
    else:
        cfg["wheels_root"] = str(path).strip().rstrip("/")
    return save(cfg)


def get_workload_wheels(name: str, data: dict[str, Any] | None = None) -> list[str]:
    cfg = load() if data is None else data
    raw = (cfg.get("workload_wheels") or {}).get(name) or []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def set_workload_wheels(name: str, filenames: list[str] | None) -> Path:
    cfg = load()
    wheels = cfg.setdefault("workload_wheels", {})
    if not isinstance(wheels, dict):
        wheels = {}
        cfg["workload_wheels"] = wheels
    cleaned = [f.strip() for f in (filenames or []) if isinstance(f, str) and f.strip()]
    if cleaned:
        wheels[name] = cleaned
    else:
        wheels.pop(name, None)
    if not wheels:
        cfg.pop("workload_wheels", None)
    return save(cfg)


def resolve_dep_paths(name: str, data: dict[str, Any] | None = None) -> list[str] | None:
    """Absolute .whl paths for a suite item, or None if wheels are not configured.

    Requires both wheels_root and a non-empty filename list for `name`.
    """
    cfg = load() if data is None else data
    root = get_wheels_root(cfg)
    files = get_workload_wheels(name, cfg)
    if not root or not files:
        return None
    return [f"{root}/{fn.lstrip('/')}" for fn in files]


def list_wheels_via_fs(root: str, profile: str | None = None,
                       timeout: int = 60) -> tuple[list[str], str | None]:
    """List *.whl basenames under a /Volumes or /Workspace path via `databricks fs ls`.

    Returns (filenames, error). On success error is None.
    """
    import subprocess

    if not root:
        return [], "wheels_root is not set"
    # databricks fs wants dbfs:/Volumes/... or /Volumes/...; Workspace paths use file:/ or bare.
    ls_path = root
    if root.startswith("/Volumes/"):
        ls_path = "dbfs:" + root
    elif root.startswith("/Workspace/"):
        ls_path = root  # workspace files API accepts /Workspace
    cmd = ["databricks", "fs", "ls", ls_path]
    if profile:
        cmd += ["-p", profile]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return [], "databricks CLI not on PATH"
    except subprocess.TimeoutExpired:
        return [], "databricks fs ls timed out"
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "fs ls failed").strip().splitlines()
        return [], (err[-1] if err else "fs ls failed")
    names: list[str] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # `databricks fs ls` prints one path/name per line (sometimes with size prefix).
        basenames = line.replace("\\", "/").rstrip("/").split("/")
        name = basenames[-1]
        if name.endswith(".whl"):
            names.append(name)
    return sorted(set(names)), None
