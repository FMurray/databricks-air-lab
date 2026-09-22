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
    "air_environment", "databricks_ai_v5",
    ["databricks_ai_v4", "databricks_ai_v5", "databricks_ai_v6", "standard_v5"],
    "Base environment",
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

# Only databricks_ai_v5 ships with a captured profile; the others (databricks_ai_v4/v6,
# standard_v5) exist as dropdown options but need a one-time capture via capture_profile,
# run on that environment. Fail fast with a clear message instead of a bare FileNotFoundError.
profiles_root = notebook_dir / "profiles"
if not profile_dir.is_dir():
    available = sorted(p.name for p in profiles_root.iterdir() if p.is_dir()) \
        if profiles_root.is_dir() else []
    raise AssertionError(
        f"no captured profile for {air_environment!r} at {profile_dir}. "
        f"Capture it first by running capture_profile on the {air_environment} environment. "
        f"Available profiles: {available}"
    )

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
print("For a native Jobs ai_runtime_task, embed this JSON object as environments[].spec:")
print(f"  {manifest['jobs_environment_file']}")
