# Databricks notebook source
# MAGIC %md
# MAGIC # Dispatch a repo workload via the vendored air CLI
# MAGIC The hosted gpu-hub broker's dispatch arm: a hosted Databricks App has no `air` binary
# MAGIC and no repo checkout, so it submits THIS notebook (CPU serverless) with a
# MAGIC `workload_ref` parameter; the notebook installs the CLI from the vendored wheel set,
# MAGIC runs `air run --file <mirror>/<workload_ref>`, and exits with JSON carrying the AIR
# MAGIC job run id. Generalized from the verified `checks/air-cli-from-notebook` pattern.

# COMMAND ----------

# MAGIC %pip install --quiet --no-index --find-links /Workspace/Shared/databricks-air-lab/uat/wheels databricks-air

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("workload_ref", "", "repo-relative workload YAML, e.g. workloads/exec-probe.yaml")

import json, os, re, subprocess

REPO = "/Workspace/Shared/databricks-air-lab"
ref = dbutils.widgets.get("workload_ref").strip()
out = {"workload_ref": ref, "air_run_id": "", "error": ""}

if not ref or ".." in ref or not ref.startswith("workloads/"):
    out["error"] = f"invalid workload_ref: {ref!r} (must be under workloads/)"
elif not os.path.exists(f"{REPO}/{ref}"):
    out["error"] = f"{ref} not found in the workspace mirror"
else:
    r = subprocess.run(["air", "run", "--file", f"{REPO}/{ref}"],
                       capture_output=True, text=True, timeout=600, cwd=REPO)
    text = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
    m = re.search(r"Job Run ID:\s*(\d+)", text)
    if m:
        out["air_run_id"] = m.group(1)
    else:
        out["error"] = text[-400:]
print(out)

# COMMAND ----------

dbutils.notebook.exit(json.dumps(out))
