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

# COMMAND ----------
# MAGIC %run ./resolve_worker

# COMMAND ----------
requirements_file = dbutils.widgets.get("requirements_file").strip()
wheelhouse_volume = dbutils.widgets.get("wheelhouse_volume").rstrip("/")
air_environment = dbutils.widgets.get("air_environment").strip()
index_url = dbutils.widgets.get("index_url").strip()

assert requirements_file, "requirements_file is required"
assert wheelhouse_volume, "wheelhouse_volume is required"
assert wheelhouse_volume.startswith("/Volumes/"), "wheelhouse_volume must be a /Volumes path"

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_path = context.notebookPath().get()
notebook_dir = Path("/Workspace") / Path(notebook_path.lstrip("/")).parent
profile_dir = notebook_dir / "profiles" / air_environment

print(f"requirements: {requirements_file}")
print(f"output:       {wheelhouse_volume}")
print(f"AIR profile:  {air_environment}")

manifest = build_wheelhouse(
    requirements_file,
    wheelhouse_volume,
    profile_dir,
    index_url=index_url,
)
print(json.dumps({key: value for key, value in manifest.items() if key != "wheel_files"},
                 indent=2, ensure_ascii=False))
assert manifest.get("ok"), f"wheelhouse build failed at {manifest['stage']}: {manifest['error']}"

print("\nApply this custom serverless environment file:")
print(f"  {manifest['environment_file']}")
