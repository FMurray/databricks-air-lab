# Databricks notebook source
# MAGIC %md
# MAGIC # UAT DRIVER — run the check suite per accelerator shape
# MAGIC
# MAGIC Run-all on plain serverless (CPU — the driver itself needs no GPU). It reads
# MAGIC `uat_config.py` (same folder; plain Python — the default env has no yaml and PyPI is
# MAGIC blocked here), then launches each check notebook as a **one-time serverless notebook
# MAGIC job per shape** with the accelerator pinned via the Jobs API
# MAGIC (`compute.hardware_accelerator`), polls them, and prints one aggregated matrix.
# MAGIC
# MAGIC **Cost control — the `shapes` widget** (top of notebook):
# MAGIC - `GPU_1xA10` (default) — cheap dry-run: CPU + A10 rows only.
# MAGIC - `all` — every shape in uat_config.py **including 8xH100. Real money; coordinate first.**
# MAGIC - or a comma list, e.g. `GPU_1xA10,GPU_1xH100`.
# MAGIC
# MAGIC Multi-node is NOT run from here: distributed multi-node goes through the **air CLI only**
# MAGIC (engagement rule) — `workloads/multinode-*.yaml`, `workloads/nccl-allreduce.example.yaml`.

# COMMAND ----------

dbutils.widgets.text("shapes", "GPU_1xA10", "shapes: comma list | all (8xH100 = real money)")

# COMMAND ----------

import json, os, time

HERE = os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext()
                       .notebookPath().get())
_ns = {}
exec(open(f"/Workspace{HERE}/uat_config.py").read(), _ns)
CFG = _ns["UAT_CONFIG"]

want = dbutils.widgets.get("shapes").strip()
ALLOWED = None if want.lower() == "all" else {s.strip() for s in want.split(",")} | {"CPU"}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# submit one job per (check, shape)
submitted = {}  # (check_path, shape) -> run_id
for check in CFG["checks"]:
    for shape in check["shapes"]:
        if ALLOWED is not None and shape not in ALLOWED:
            print(f"skip {check['path']} @ {shape} (not in shapes widget)")
            continue
        task = {
            "task_key": "check",
            "notebook_task": {
                "notebook_path": f"{HERE}/{check['path']}",
                "base_parameters": {"expect_gpus": str(check.get("expect_gpus", {}).get(shape, ""))},
            },
            "environment_key": "uat_env",
            "timeout_seconds": check["timeout_minutes"] * 60,
        }
        if shape != "CPU":
            task["compute"] = {"hardware_accelerator": shape}
        # raw API call: the preinstalled databricks-sdk may predate the `environments` kwarg
        run = w.api_client.do("POST", "/api/2.2/jobs/runs/submit", body={
            "run_name": f"uat-{os.path.basename(check['path'])}-{shape}",
            "tasks": [task],
            "environments": [{"environment_key": "uat_env",
                              "spec": {"environment_version": CFG["environment_version"],
                                       "dependencies": CFG["dependencies"]}}],
        })
        submitted[(check["path"], shape)] = run["run_id"]
        print(f"submitted {check['path']} @ {shape} -> run {run['run_id']}")

# COMMAND ----------

# poll all to terminal, collect notebook exit JSON
results = {}
pending = dict(submitted)
while pending:
    time.sleep(30)
    for key, run_id in list(pending.items()):
        r = w.api_client.do("GET", f"/api/2.2/jobs/runs/get?run_id={run_id}")
        state = r["state"]["life_cycle_state"]
        if state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            del pending[key]
            outcome = {"run_id": run_id,
                       "result_state": r["state"].get("result_state", state)}
            try:
                out = w.api_client.do(
                    "GET", f"/api/2.2/jobs/runs/get-output?run_id={r['tasks'][0]['run_id']}")
                payload = json.loads(out["notebook_output"]["result"])
                outcome["results"] = payload["results"]
            except Exception as e:
                outcome["results"] = {"_output": {"ok": False,
                                                  "detail": f"{type(e).__name__}: {e}"[:200]}}
            results[key] = outcome
            print(f"done {key[0]} @ {key[1]}: {outcome['result_state']}")
        else:
            print(f"... {key[0]} @ {key[1]}: {state}")

# COMMAND ----------

# MAGIC %md ## Summary matrix — got vs expected-now

# COMMAND ----------

EXPECTED_BROKEN = {"catalog_bucket_read", "pypi_egress", "mlflow_artifact_upload"}
# root_storage_tcp443: expected ✅ on CPU, ❌ on GPU shapes (the plane differential, docs/06)

print(f"{'check':24} {'shape':12} {'probe':28} got exp")
for (path, shape), outcome in sorted(results.items()):
    for name, r in outcome["results"].items():
        got = {True: "✅", False: "❌", None: "⏭️"}[r["ok"]]
        exp = "❌" if (name in EXPECTED_BROKEN
                      or (name == "root_storage_tcp443" and shape != "CPU")) else "✅"
        drift = ("" if got == exp or r["ok"] is None
                 else " ← CHANGED (fixed?)" if got == "✅" else " ← REGRESSION")
        print(f"{os.path.basename(path):24} {shape:12} {name:28} {got}  {exp}{drift}  {r['detail'][:80]}")

print("\nRe-run after the network fix: expected-❌ flipping to ✅ = UAT unblocked (docs/06-uat-suite.md).")
dbutils.notebook.exit(json.dumps({f"{p}@{s}": o for (p, s), o in results.items()}))
