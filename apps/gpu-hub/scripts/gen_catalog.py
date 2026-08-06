#!/usr/bin/env python3
"""Generate plugins/broker/catalog.json from the repo's workloads/ directory.
Run whenever workloads change (predeploy). Reuses the training-hub catalog logic."""
import json, sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP.parent / "training-hub"))
from hub.catalog import repo_workloads  # noqa: E402

team = json.loads((APP / "config" / "broker.json").read_text())["catalog_team"]
catalog = repo_workloads(team)

VARIANT_SUFFIXES = ("-nodeps", "-debug", "-fevm", "-mkazia", "-v5")
for c in catalog:
    stem = c["name"].split("/")[-1]
    for suf in VARIANT_SUFFIXES:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    c["workload_key"] = stem

out = APP / "plugins" / "broker" / "catalog.json"
out.write_text(json.dumps(catalog, indent=1))
print(f"{len(catalog)} configurations -> {out}")

sys.path.insert(0, str(APP.parents[1]))
from utils.composition.compose import OPTIONS  # noqa: E402
opts = [{"name": k, "description": v.get("description", ""),
         "params": v.get("params", {})} for k, v in OPTIONS.items()]
(APP / "plugins" / "broker" / "options.json").write_text(json.dumps(opts, indent=1))
print(f"{len(opts)} options -> options.json")
