# Databricks notebook source
# MAGIC %md
# MAGIC # Build an AIR wheelhouse
# MAGIC
# MAGIC Run this notebook on a classic cluster with Artifactory access. Provide a developer
# MAGIC `requirements.txt` and an output UC Volume directory. The notebook uses a baked AIR profile,
# MAGIC resolves with uv, and prints the generated serverless `environment.yaml` path.

# COMMAND ----------
dbutils.widgets.text("requirements_file", "", "Developer requirements.txt path")
dbutils.widgets.text("wheelhouse_volume", "", "UC Volume output directory")
dbutils.widgets.dropdown(
    "air_environment", "databricks_ai_v5", ["databricks_ai_v5"], "AIR environment",
)
dbutils.widgets.text("index_url", "", "Artifactory index URL (blank = cluster configuration)")

# COMMAND ----------
import json
from pathlib import Path

REQUIREMENTS_FILE = dbutils.widgets.get("requirements_file").strip()
WHEELHOUSE_VOLUME = dbutils.widgets.get("wheelhouse_volume").rstrip("/")
AIR_ENVIRONMENT = dbutils.widgets.get("air_environment").strip()
INDEX_URL = dbutils.widgets.get("index_url").strip()

assert REQUIREMENTS_FILE, "requirements_file is required"
assert WHEELHOUSE_VOLUME, "wheelhouse_volume is required"
assert WHEELHOUSE_VOLUME.startswith("/Volumes/"), "wheelhouse_volume must be a /Volumes path"

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_path = context.notebookPath().get()
notebook_dir = Path("/Workspace") / Path(notebook_path.lstrip("/")).parent
PROFILE_DIR = notebook_dir / "profiles" / AIR_ENVIRONMENT

print(f"requirements: {REQUIREMENTS_FILE}")
print(f"output:       {WHEELHOUSE_VOLUME}")
print(f"AIR profile:  {AIR_ENVIRONMENT}")

# COMMAND ----------
# MAGIC %run ./resolve_worker

# COMMAND ----------
manifest = build_wheelhouse(
    REQUIREMENTS_FILE,
    WHEELHOUSE_VOLUME,
    PROFILE_DIR,
    index_url=INDEX_URL,
)
print(json.dumps({key: value for key, value in manifest.items() if key != "wheel_files"},
                 indent=2, ensure_ascii=False))
assert manifest.get("ok"), f"wheelhouse build failed at {manifest['stage']}: {manifest['error']}"

print("\nUse this file as the AIR workload dependency spec:")
print(f"  environment:\n    dependencies: {manifest['environment_file']}")
