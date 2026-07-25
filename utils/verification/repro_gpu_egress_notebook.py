# Databricks notebook source
# MAGIC %md
# MAGIC # REPRO: GPU-plane storage egress stall (fe-sandbox-mkazia-lw2)
# MAGIC
# MAGIC **How to run:** compute dropdown → **Serverless GPU** → Environment panel → Accelerator
# MAGIC **A10**, Base environment **AI v4** → Apply. Then **Run all** (~2 min).
# MAGIC
# MAGIC **What you'll see on GPU:** cell 2 (TCP connect to the workspace root-storage bucket)
# MAGIC **succeeds**, but cell 3 (an actual ~30-byte MLflow artifact upload to that same bucket)
# MAGIC **hangs and times out after 60s**. Run the identical notebook on plain serverless (CPU)
# MAGIC and cell 3 completes in <1s. So the GPU plane's egress path stalls data transfer to the
# MAGIC workspace's own S3 — a connection is allowed, payload never lands.
# MAGIC
# MAGIC **Why it matters:** the AIR launcher ships run logs to this same bucket → every AIR GPU
# MAGIC run has NO logs (`air logs` empty, no MLflow log artifacts) and runs can hang to
# MAGIC TIMEDOUT after user code succeeds. Full write-up: `docs/06-uat-suite.md` in this folder.

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
# MAGIC air run --file workloads/cli-egress-probe.example.yaml -p mkazia-lw2
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

# machine-readable verdict (for scripted verification; humans read the cells above)
dbutils.notebook.exit(f"gpu={gpu} | upload={UPLOAD} | pypi={PYPI}")
