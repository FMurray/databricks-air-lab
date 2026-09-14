# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — MLR DOWNLOAD DRIVER (run on CLASSIC 17.3 + Artifactory)
# MAGIC
# MAGIC Step 2 of the two-step flow. Attach this notebook to a **classic MLR 17.3 cluster** that can
# MAGIC reach Artifactory. It sets the stage dir, then `%run`s the worker, which resolves for the
# MAGIC AIR interpreter/platform and downloads the exact name+version delta on this cluster.
# MAGIC (`%run` runs the worker in THIS cluster's context — that's exactly why the download happens
# MAGIC on the MLR side.)
# MAGIC
# MAGIC Prereq: `air_freeze` Phase 1 has written `constraints.txt`, `requirements.txt`,
# MAGIC `overrides.txt`, and `target_env.json` into the stage dir below.

# COMMAND ----------
dbutils.widgets.text("stage_dir", "/Volumes/<catalog>/<schema>/<vol>/vendor-stage",
                     "UC Volume stage dir (same one air_freeze wrote to)")
dbutils.widgets.text("index_url", "", "Artifactory index URL (blank = this cluster's pip.conf)")
dbutils.widgets.text("overrides", "",
                     "Optional replacement overrides; blank keeps the AIR-staged overrides.txt")

# these variables are read by the worker via %run (namespace is shared)
stage_dir = dbutils.widgets.get("stage_dir").rstrip("/")
index_url = dbutils.widgets.get("index_url").strip()
override_text = dbutils.widgets.get("overrides").strip()
if override_text:
    with open(f"{stage_dir}/overrides.txt", "w") as handle:
        handle.write(override_text + "\n")
print(f"stage_dir = {stage_dir}\nindex = {'explicit Artifactory URL' if index_url else 'cluster pip.conf'}\n"
      f"explicit overrides = {override_text.splitlines() if override_text else 'AIR-staged value'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## The orchestration: %run the worker on this (MLR) cluster
# MAGIC The worker retains every compatible AIR pin, detects required replacements by walking the
# MAGIC target package graph, and writes a request-specific wheelhouse plus hashed delta lock. It
# MAGIC leaves a `manifest` in the namespace (it does not exit, so this driver keeps running).

# COMMAND ----------
# MAGIC %run ./resolve_worker

# COMMAND ----------
# MAGIC %md
# MAGIC ## Result

# COMMAND ----------
import json
print(json.dumps({k: v for k, v in manifest.items() if k != "wheel_files"}, indent=2, ensure_ascii=False))
if manifest.get("ok"):
    print(f"\n✓ {len(manifest.get('delta', []))} wheels staged to {manifest['wheelhouse']}")
    print("➡  Go back to air_freeze and run Phase 2 to verify offline resolution.")
else:
    # echo the resolver's message verbatim — json.dumps escapes its box-drawing chars/newlines
    print("\n✗ resolve failed — review the detected overrides/error before verifying on AIR:")
    print(manifest.get("error", ""))
    print(manifest.get("stderr_tail", ""))
    for f in manifest.get("failures", []):
        print(f)
    raise RuntimeError(f"wheelhouse build failed at {manifest.get('stage')}: {manifest.get('error')}")
