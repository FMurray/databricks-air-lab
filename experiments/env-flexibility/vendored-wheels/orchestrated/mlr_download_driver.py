# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — MLR DOWNLOAD DRIVER (run on CLASSIC 17.3 + Artifactory)
# MAGIC
# MAGIC Step 2 of the two-step flow. Attach this notebook to a **classic MLR 17.3 cluster** that can
# MAGIC reach Artifactory (Python must match the AIR target — cp312 today). It sets the stage dir,
# MAGIC then `%run`s the worker, which does the delta resolve + download entirely on this cluster.
# MAGIC (`%run` runs the worker in THIS cluster's context — that's exactly why the download happens
# MAGIC on the MLR side.)
# MAGIC
# MAGIC Prereq: `air_freeze` Phase 1 has already written `constraints.txt` / `requirements.txt` /
# MAGIC `target_env.json` into the stage dir below.

# COMMAND ----------
dbutils.widgets.text("stage_dir", "/Volumes/<catalog>/<schema>/<vol>/vendor-stage",
                     "UC Volume stage dir (same one air_freeze wrote to)")
dbutils.widgets.text("index_url", "", "Artifactory index URL (blank = this cluster's pip.conf)")

# these variables are read by the worker via %run (namespace is shared)
stage_dir = dbutils.widgets.get("stage_dir").rstrip("/")
index_url = dbutils.widgets.get("index_url").strip()
print(f"stage_dir = {stage_dir}\nindex_url = {index_url or '(cluster pip.conf)'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## The orchestration: %run the worker on this (MLR) cluster
# MAGIC The worker resolves the closure bounded by the AIR env's constraints, subtracts what AIR
# MAGIC already has, and downloads only the delta into `stage_dir/wheelhouse`. It leaves a `manifest`
# MAGIC in the namespace (it does not exit, so this driver keeps running).

# COMMAND ----------
# MAGIC %run ./resolve_worker

# COMMAND ----------
# MAGIC %md
# MAGIC ## Result

# COMMAND ----------
import json
print(json.dumps({k: v for k, v in manifest.items() if k != "wheel_files"}, indent=2, ensure_ascii=False))
if manifest.get("ok"):
    print(f"\n✓ {len(manifest.get('downloaded', []))} wheels staged to {manifest['wheelhouse']}")
    print("➡  Go back to air_freeze and run Phase 2 to verify offline resolution.")
else:
    # echo the resolver's message verbatim — json.dumps escapes its box-drawing chars/newlines
    print("\n✗ resolve failed — resolver output below, then fix requirements before verifying on AIR:")
    print(manifest.get("error", ""))
    print(manifest.get("stderr_tail", ""))
    for f in manifest.get("failures", []):
        print(f)
