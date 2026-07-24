# Databricks notebook source
# MAGIC %md
# MAGIC # UAT DRIVER — run the whole check suite
# MAGIC
# MAGIC Run-all on any serverless env. Each check lives in `checks/` as its own notebook with its
# MAGIC own environment (per-env pattern): `dbutils.notebook.run` executes each child as an
# MAGIC ephemeral job using the **child's stored environment** — so GPU checks bring their own
# MAGIC accelerator (see `environment.yml` + the header cell of each check) while this driver can
# MAGIC sit on cheap CPU serverless.
# MAGIC
# MAGIC First-time setup for GPU checks: open `checks/gpu-smoke`, Environment panel → Base
# MAGIC environment **AI**, Accelerator **A10**, save. (Imports don't always carry the
# MAGIC accelerator setting.)
# MAGIC
# MAGIC Expected-broken checks (until the network fix — `docs/06-uat-suite.md`):
# MAGIC `catalog_bucket_read`, `root_storage_tcp443`, `pypi_egress`, `mlflow_artifact_upload`.

# COMMAND ----------

import json

CHECKS = [
    # (relative notebook, timeout_seconds)
    ("checks/network-blockers", 900),
    ("checks/gpu-smoke", 1200),
]

EXPECTED_BROKEN = {"catalog_bucket_read", "root_storage_tcp443", "pypi_egress", "mlflow_artifact_upload"}

all_results = {}
for path, timeout in CHECKS:
    print(f"\n=== running {path} (timeout {timeout}s) ===")
    try:
        out = dbutils.notebook.run(path, timeout)
        payload = json.loads(out)
        all_results[payload["check"]] = payload["results"]
    except Exception as e:
        all_results[path] = {"_notebook_run": {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}}
        print("❌ notebook.run failed:", str(e)[:200])

# COMMAND ----------

# MAGIC %md ## Summary — got vs expected-now

# COMMAND ----------

rows = []
for check, res in all_results.items():
    for name, r in res.items():
        got = {True: "✅", False: "❌", None: "⏭️"}[r["ok"]]
        exp = "❌" if name in EXPECTED_BROKEN else "✅"
        if r["ok"] is None:
            drift = "(skipped)"
        elif got == exp:
            drift = ""
        elif got == "✅":
            drift = "← CHANGED (blocker fixed?)"
        else:
            drift = "← REGRESSION"
        rows.append((check, name, got, exp, drift, r["detail"]))
        print(f"{check:20} {name:28} {got} (expected-now {exp}) {drift}")

print("\nAll four expected-❌ checks flipping to ✅ = network fix applied, UAT unblocked.")
print("Receipts + context: /Workspace/Shared/databricks-air-lab/docs/06-uat-suite.md")

# COMMAND ----------

dbutils.notebook.exit(json.dumps(all_results))
