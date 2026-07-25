# Databricks notebook source
# MAGIC %md
# MAGIC # REPRO: GPU-plane storage egress stall (fe-sandbox-mkazia-lw2)
# MAGIC
# MAGIC **The issue is specific to JOB-SUBMITTED GPU runs.** Same code, three outcomes
# MAGIC (all verified 2026-07-24/25):
# MAGIC
# MAGIC | How the GPU run is launched | Artifact upload to workspace root storage |
# MAGIC |---|---|
# MAGIC | Interactive notebook (Serverless GPU attached) | ✅ OK in 0.8s |
# MAGIC | Notebook **job** (`compute.hardware_accelerator`) | ❌ stalls, 60s timeout |
# MAGIC | **AIR CLI Gen-AI task — what multinode training uses** | ❌ stalls, 60s timeout |
# MAGIC
# MAGIC **Healthy control:** attach Serverless GPU (A10, AI v4) and **Run all** — everything
# MAGIC passes, including cell 3 (TCP ✅, upload ✅). That proves the bucket and the GPU env are fine.
# MAGIC
# MAGIC **Reproduce the failure:** run **cell 6** below — it submits THIS notebook as a GPU
# MAGIC *job* and prints the verdict (`upload=REPRODUCED after 60s`). Or use the CLI path (cell 5).
# MAGIC
# MAGIC **Why it matters:** the AIR launcher ships run logs to this same bucket → every job-
# MAGIC submitted GPU run (i.e. every CLI training run, incl. all multinode) has NO logs
# MAGIC (`air logs` empty, no MLflow log artifacts), and hung log-shipping can keep runs alive
# MAGIC past `timeout_minutes` (observed: 12-min probe ran ~6h until manual cancel — billing
# MAGIC risk). Full write-up: `docs/06-uat-suite.md` in this folder.

# COMMAND ----------

# Cell 1 — where am I running?
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU serverless)"
except ImportError:
    gpu = "none (CPU serverless, no torch)"
print(f"GPU: {gpu}")

# COMMAND ----------

# Cell 2 — TCP 443 to the workspace root-storage bucket: connects fine even on GPU
import socket, time
HOST = "mkazia-lw2-workspace-root-storage.s3-fips.us-east-1.amazonaws.com"
t0 = time.time()
socket.create_connection((HOST, 443), timeout=15).close()
print(f"TCP connect to {HOST}:443 OK in {time.time()-t0:.2f}s  → NOT a connection deny")

# COMMAND ----------

# Cell 3 — THE ISSUE: a tiny artifact upload to that bucket. <1s on CPU, times out on GPU.
import mlflow, signal, time

def _alarm(sig, frame):
    raise TimeoutError("upload still hanging after 60s")

with open("/tmp/receipt.txt", "w") as f:
    f.write("repro: gpu-plane egress stall")

signal.signal(signal.SIGALRM, _alarm)
signal.alarm(60)
t0 = time.time()
try:
    with mlflow.start_run(run_name="repro-gpu-egress"):
        mlflow.log_artifact("/tmp/receipt.txt")
    UPLOAD = f"OK in {time.time()-t0:.1f}s"
    print(f"✅ upload {UPLOAD} — egress from THIS plane is healthy")
except Exception as e:
    UPLOAD = f"REPRODUCED after {time.time()-t0:.0f}s ({type(e).__name__})"
    print(f"❌ {UPLOAD}: {str(e)[:200]}")
    print("   TCP connects (cell 2) but the upload stalls → stateful egress/proxy filtering")
    print("   on the GPU plane's path to the workspace's own S3 bucket.")
finally:
    signal.alarm(0)

# COMMAND ----------

# Cell 4 — related: PyPI DNS fails on BOTH planes (this is what breaks pip deps on AIR runs)
import urllib.request
try:
    with urllib.request.urlopen("https://pypi.org/simple/", timeout=15) as r:
        PYPI = f"HTTP {r.status}"
        print(f"pypi.org: {PYPI} — egress OK")
except Exception as e:
    PYPI = f"{type(e).__name__}"
    print(f"pypi.org: {PYPI}: {e}  → pip dependencies cannot install")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — reproduce on the **CLI path** (the one multinode training uses)
# MAGIC
# MAGIC This notebook covers the *notebook-job* GPU path. Distributed/multi-node training uses the
# MAGIC **AIR CLI** path (Gen AI compute task) — same stall, verified run `683173786603437`. To
# MAGIC reproduce it yourself (laptop, ~5 min):
# MAGIC
# MAGIC ```bash
# MAGIC uv tool install databricks-air
# MAGIC databricks auth login --host https://fe-sandbox-mkazia-lw2.cloud.databricks.com --profile mkazia-lw2
# MAGIC # from a copy of this folder (Workspace UI → ⋮ → Export → Source), repo root:
# MAGIC air run --file workloads/probes/cli-egress-probe.example.yaml -p mkazia-lw2
# MAGIC ```
# MAGIC
# MAGIC **Read the result in MLflow** (run logs are the very thing that's broken, so the probe
# MAGIC reports via params): Experiments → `air-lab-cli-egress-probe` → newest run → Params:
# MAGIC `tcp_root_storage = OK`, `artifact_upload = FAIL ... 60s`, `probe_done = yes`.
# MAGIC
# MAGIC ⚠️ **Then cancel the job run** (Job runs tab): the hung log-shipping can keep the run
# MAGIC alive past its `timeout_minutes` — one probe ran ~6 h before manual cancel. Params land
# MAGIC within ~3 min of start; nothing after that is useful.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — one-click repro: submit THIS notebook as a GPU **job** and read the verdict
# MAGIC Interactive runs are healthy (see above); the job path is what breaks. This cell
# MAGIC submits the notebook via the Jobs API with `hardware_accelerator: GPU_1xA10`, waits
# MAGIC (~4–6 min incl. provisioning), and prints the child's verdict. Expect
# MAGIC `upload=REPRODUCED after 60s` until the egress fix lands — then this flips to `upload=OK`.

# COMMAND ----------

import json as _json
import time as _t

_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_tags = _json.loads(_ctx.toJson()).get("tags", {})
if "jobId" in _tags or "runId" in _tags or "jobRunId" in _tags:
    # we ARE the job-submitted child — never self-submit again (would recurse)
    print("running as a job — skipping self-submission; verdict comes from the exit cell below")
else:
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
    _path = _ctx.notebookPath().get()
    _run = _w.api_client.do("POST", "/api/2.2/jobs/runs/submit", body={
        "run_name": "repro-gpu-egress-jobpath",
        "tasks": [{"task_key": "repro", "notebook_task": {"notebook_path": _path},
                   "compute": {"hardware_accelerator": "GPU_1xA10"},
                   "environment_key": "e", "timeout_seconds": 900}],
        "environments": [{"environment_key": "e",
                          "spec": {"environment_version": "4", "dependencies": []}}],
    })
    print(f"submitted job run {_run['run_id']} on GPU_1xA10 — polling (~4-6 min)...")
    while True:
        _t.sleep(30)
        _r = _w.api_client.do("GET", f"/api/2.2/jobs/runs/get?run_id={_run['run_id']}")
        _state = _r["state"]["life_cycle_state"]
        print("  ", _state)
        if _state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            break
    _out = _w.api_client.do("GET", f"/api/2.2/jobs/runs/get-output?run_id={_r['tasks'][0]['run_id']}")
    print("\nJOB-PATH VERDICT:", _out.get("notebook_output", {}).get("result", "(no output)"))
    print("(compare with your interactive run above: upload OK — same code, same GPU type)")

# COMMAND ----------

# machine-readable verdict (for scripted verification; humans read the cells above)
dbutils.notebook.exit(f"gpu={gpu} | upload={UPLOAD} | pypi={PYPI}")
