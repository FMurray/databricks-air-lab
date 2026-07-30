# Databricks notebook source
# MAGIC %md
# MAGIC # CAVEAT REPRO: AIR env v4 breaks job-submitted GPU egress (fe-sandbox-mkazia-lw2)
# MAGIC
# MAGIC **The caveat (verified 2026-07-24/25):** on this workspace, GPU runs submitted as *jobs*
# MAGIC cannot upload to the workspace root-storage bucket **when the environment is version 4**.
# MAGIC Version 5 works. Interactive GPU notebooks are unaffected either way.
# MAGIC
# MAGIC | GPU run | env v4 | env v5 |
# MAGIC |---|---|---|
# MAGIC | Interactive notebook | ✅ (0.8s) | ✅ |
# MAGIC | Notebook job (`hardware_accelerator`) | ❌ 60s stall ×4 runs | ✅ 11.5s (run 733701251559072) |
# MAGIC | AIR CLI Gen-AI task (multinode's path) | ❌ 60s stall | ✅ 0.4s (run 20867331866373) |
# MAGIC
# MAGIC **Action for all UAT workloads: pin `environment.version: "5"`** (workload YAMLs) /
# MAGIC `environment_version: "5"` (notebook-job env specs). Repo YAMLs are already pinned.
# MAGIC
# MAGIC **Why it matters:** the AIR launcher ships run logs to that same bucket — v4 runs have
# MAGIC no logs (`air logs` empty, no MLflow log artifacts), can hang past `timeout_minutes`
# MAGIC (observed ~6h on a 12-min run — billing risk), and TIMEDOUT/INTERNAL_ERROR states can
# MAGIC mask successful user code. Also: PyPI is unreachable from all planes (by design —
# MAGIC customer-realistic; vendor wheels). Full receipts: `docs/06-uat-suite.md` in this folder.
# MAGIC
# MAGIC **Run-all** on any serverless compute: cells 1–4 check the current plane; cell 6 submits
# MAGIC this notebook as TWO GPU jobs (env v4 + env v5) and prints both verdicts side by side —
# MAGIC that pair is the caveat. ⚠️ The v4 child may hang after finishing its work; cell 6
# MAGIC cancels it once verdicts are collected.

# COMMAND ----------

# Cell 1 — where am I running? (nvidia-smi = hardware truth; torch presence ≠ GPU presence —
# e.g. env v5 without the AI base env has a GPU but no torch)
import subprocess
try:
    _smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=20)
    gpu = ", ".join(_smi.stdout.split()) if _smi.returncode == 0 and _smi.stdout.strip() else "none (CPU serverless)"
except FileNotFoundError:
    gpu = "none (CPU serverless)"
print(f"GPU: {gpu}")

# COMMAND ----------

# Cell 2 — TCP 443 to the workspace root-storage bucket (connects on every plane, even
# where uploads stall — the block is in data transfer, not connection setup)
import socket, time
HOST = "mkazia-lw2-workspace-root-storage.s3-fips.us-east-1.amazonaws.com"
t0 = time.time()
socket.create_connection((HOST, 443), timeout=15).close()
print(f"TCP connect to {HOST}:443 OK in {time.time()-t0:.2f}s")

# COMMAND ----------

# Cell 3 — the probe: a tiny artifact upload to that bucket, 60s fail-fast guard.
import signal, time
try:
    import mlflow
    def _alarm(sig, frame):
        raise TimeoutError("upload still hanging after 60s")
    with open("/tmp/receipt.txt", "w") as f:
        f.write("repro: env-v4 gpu egress caveat")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(60)
    t0 = time.time()
    try:
        with mlflow.start_run(run_name="repro-gpu-egress"):
            mlflow.log_artifact("/tmp/receipt.txt")
        UPLOAD = f"OK in {time.time()-t0:.1f}s"
        print(f"✅ upload {UPLOAD} — egress from THIS plane/env is healthy")
    except Exception as e:
        UPLOAD = f"REPRODUCED after {time.time()-t0:.0f}s ({type(e).__name__})"
        print(f"❌ {UPLOAD}: {str(e)[:200]}")
    finally:
        signal.alarm(0)
except ImportError:
    UPLOAD = "SKIPPED (no mlflow in this env)"
    print(UPLOAD)

# COMMAND ----------

# Cell 4 — related: PyPI unreachable from all planes (by design — no-PyPI posture; vendor wheels)
import urllib.request
try:
    with urllib.request.urlopen("https://pypi.org/simple/", timeout=15) as r:
        PYPI = f"HTTP {r.status}"
        print(f"pypi.org: {PYPI} — egress OK")
except Exception as e:
    PYPI = f"{type(e).__name__}"
    print(f"pypi.org: {PYPI}: {e}  → pip dependencies cannot install; vendor wheels instead")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — reproduce on the **CLI path** (what multinode training uses)
# MAGIC
# MAGIC ```bash
# MAGIC # healthy (v5 default in the repo YAML — clean exit, logs work):
# MAGIC air run --file workloads/probes/cli-egress-probe.example.yaml -p mkazia-lw2
# MAGIC # broken (v4 — reproduces the stall):
# MAGIC air run --file workloads/probes/cli-egress-probe.example.yaml -p mkazia-lw2 --override environment.version=4
# MAGIC ```
# MAGIC
# MAGIC Results land as MLflow **params** (experiment `air-lab-cli-egress-probe`):
# MAGIC `tcp_root_storage`, `artifact_upload`, `pypi_egress`, `probe_done`.
# MAGIC ⚠️ **The v4 run must be cancelled manually** after `probe_done=yes` (~3 min) — hung
# MAGIC log-shipping defeated `timeout_minutes` once (~6h runaway, run 683173786603437).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — one-click A/B: submit this notebook as env-v4 AND env-v5 GPU jobs
# MAGIC Expect `v4 → upload=REPRODUCED after 60s` and `v5 → upload=OK` (~5–8 min total).
# MAGIC If v4 flips to OK, the platform/network fix has landed — update `docs/06-uat-suite.md`.

# COMMAND ----------

import time as _t

# job-context guard: children must never self-submit (recursion = runaway GPU jobs).
_is_job = False
try:
    _is_job = dbutils.widgets.get("is_job_child") == "1"
except Exception:
    pass
_ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
if not _is_job:
    try:
        _is_job = _ctx.jobId().isDefined()
    except Exception:
        pass

if _is_job:
    print("running as a job — skipping self-submission; verdict comes from the exit cell below")
else:
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
    _path = _ctx.notebookPath().get()

    def _submit(env_version):
        return _w.api_client.do("POST", "/api/2.2/jobs/runs/submit", body={
            "run_name": f"repro-gpu-egress-env-v{env_version}",
            "tasks": [{"task_key": "repro",
                       "notebook_task": {"notebook_path": _path,
                                         "base_parameters": {"is_job_child": "1"}},
                       "compute": {"hardware_accelerator": "GPU_1xA10"},
                       "environment_key": "e", "timeout_seconds": 900}],
            "environments": [{"environment_key": "e",
                              "spec": {"environment_version": env_version, "dependencies": []}}],
        })["run_id"]

    _runs = {"4": _submit("4"), "5": _submit("5")}
    print(f"submitted env-v4 run {_runs['4']} and env-v5 run {_runs['5']} — polling (~5-8 min)...")
    _verdicts, _deadline = {}, _t.time() + 20 * 60
    while _runs and _t.time() < _deadline:
        _t.sleep(30)
        for _v, _rid in list(_runs.items()):
            _r = _w.api_client.do("GET", f"/api/2.2/jobs/runs/get?run_id={_rid}")
            _state = _r["state"]["life_cycle_state"]
            if _state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
                del _runs[_v]
                try:
                    _out = _w.api_client.do(
                        "GET", f"/api/2.2/jobs/runs/get-output?run_id={_r['tasks'][0]['run_id']}")
                    _verdicts[_v] = _out.get("notebook_output", {}).get("result", "(no output)")
                except Exception as _e:
                    _verdicts[_v] = f"(output unreadable: {_e})"
            else:
                print(f"  env v{_v}: {_state}")
    for _v, _rid in _runs.items():  # v4 child may hang past its work — don't wait forever
        _w.api_client.do("POST", "/api/2.2/jobs/runs/cancel", body={"run_id": _rid})
        _verdicts[_v] = f"(hung — cancelled run {_rid}; the hang IS the v4 symptom)"
    print("\n=== A/B VERDICT ===")
    for _v in ("4", "5"):
        print(f"env v{_v}: {_verdicts.get(_v)}")

# COMMAND ----------

# machine-readable verdict (for scripted verification; humans read the cells above)
dbutils.notebook.exit(f"gpu={gpu} | upload={UPLOAD} | pypi={PYPI}")
