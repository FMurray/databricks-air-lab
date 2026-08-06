"""Compose an AIR workload YAML from a base recipe plus options.

Each option encodes one receipt-backed recipe (docs/06, docs/cookbook, skills).
An option is a transform: env variables, include paths, a command prelude, secrets,
and field overrides. Both dispatch paths use this module: the local CLI path and the
hosted notebook hop. Keep it dependency-light: PyYAML only.

Usage:
    python -m utils.composition.compose <base.yaml> --options ai-env-torch,deps-uc-volume \
        --param deps-uc-volume.vendor_path=/Volumes/x/y/deps/vendor --out composed.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

AI_ENV = "/opt/databricks-environments/databricks-ai"

# Every option: prelude lines run before the base command; other keys merge into the YAML.
OPTIONS: dict[str, dict] = {
    "ai-env-torch": {
        "description": "Use torch from the baked-in AI environment (env v5 default python has none).",
        "prelude": [
            f'export PATH="{AI_ENV}/bin:$PATH"',
            f'export PYTHONPATH="{AI_ENV}/lib/python3.12/site-packages:$PYTHONPATH"',
        ],
    },
    "deps-snapshot": {
        "description": "Vendored packages shipped in the code snapshot (small package sets).",
        "params": {"vendor_dir": "experiments/env-flexibility/vendored-wheels/vendor"},
        "include_paths": ["{vendor_dir}"],
        "prelude": ['export PYTHONPATH="$CODE_SOURCE_PATH/{vendor_dir}:$PYTHONPATH"'],
    },
    "deps-uc-volume": {
        "description": "Vendored packages staged in a UC volume (large package sets).",
        "params": {"vendor_path": "/Volumes/<catalog>/<schema>/deps/vendor"},
        "prelude": ['export PYTHONPATH="{vendor_path}:$PYTHONPATH"'],
    },
    "airtel-telemetry": {
        "description": "OTEL logs/metrics/GPU telemetry to Delta via Zerobus (needs SP secrets).",
        "params": {
            "workspace_id": "",
            "workspace_url": "",
            "zerobus_region": "us-east-1",
            "logs_table": "",
            "metrics_table": "",
            "secret_scope": "air_lab",
        },
        "include_paths": ["utils/telemetry/"],
        "env_variables": {
            "WORKSPACE_ID": "{workspace_id}",
            "WORKSPACE_URL": "{workspace_url}",
            "ZEROBUS_REGION": "{zerobus_region}",
            "OTEL_LOGS_TABLE": "{logs_table}",
            "OTEL_METRICS_TABLE": "{metrics_table}",
        },
        "secrets": {
            "DATABRICKS_CLIENT_ID": "{secret_scope}/zerobus_sp_client_id",
            "DATABRICKS_CLIENT_SECRET": "{secret_scope}/zerobus_sp_client_secret",
        },
        "prelude": ['export PYTHONPATH="$CODE_SOURCE_PATH:$PYTHONPATH"'],
    },
    "receipts": {
        "description": "Mark the run for MLflow param receipts (verification-skill conventions).",
        "env_variables": {"AIR_LAB_RECEIPTS": "1"},
    },
}


def compose(base: dict, options: list[str], params: dict[str, str] | None = None) -> dict:
    """Return a new workload dict: base plus the selected option transforms."""
    out = copy.deepcopy(base)
    params = params or {}
    prelude: list[str] = []
    for name in options:
        opt = OPTIONS.get(name)
        if opt is None:
            raise KeyError(f"unknown option: {name}")
        fill = {**opt.get("params", {}),
                **{k.split(".", 1)[1]: v for k, v in params.items()
                   if k.startswith(name + ".")}}

        def f(s: str) -> str:
            return s.format(**fill) if fill else s

        for line in opt.get("prelude", []):
            prelude.append(f(line))
        for key, val in opt.get("env_variables", {}).items():
            out.setdefault("env_variables", {})[key] = f(val)
        for key, val in opt.get("secrets", {}).items():
            out.setdefault("secrets", {})[key] = f(val)
        for ip in opt.get("include_paths", []):
            snap = out.setdefault("code_source", {}).setdefault("snapshot", {})
            snap.setdefault("root_path", "..")
            paths = snap.setdefault("include_paths", [])
            if f(ip) not in paths:
                paths.append(f(ip))
    if prelude:
        out["command"] = "\n".join([*prelude, str(out.get("command", "")).rstrip()])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--options", default="")
    ap.add_argument("--param", action="append", default=[],
                    help="option-scoped value: <option>.<key>=<value>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="",
                    help="absolute snapshot root; use when the composed file is not next to the base")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.base).read_text())
    options = [o for o in args.options.split(",") if o]
    params = dict(p.split("=", 1) for p in args.param)
    composed = compose(base, options, params)
    if args.root and "code_source" in composed:
        composed["code_source"]["snapshot"]["root_path"] = args.root
    Path(args.out).write_text(yaml.safe_dump(composed, sort_keys=False))
    print(json.dumps({"out": args.out, "options": options}))


if __name__ == "__main__":
    main()
