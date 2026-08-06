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
out = APP / "plugins" / "broker" / "catalog.json"
out.write_text(json.dumps(catalog, indent=1))
print(f"{len(catalog)} workloads -> {out}")
