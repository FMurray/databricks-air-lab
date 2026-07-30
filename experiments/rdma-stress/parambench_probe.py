"""M5: parambench-train-comms (Meta's comms benchmark) — present in the v5 image's package
list; requires torch, which on v5 lives in the databricks-ai env (survey 96419244890099).
Run under: /opt/databricks-environments/databricks-ai/bin/python (or env v4's default).

Probe-grade: verify the module imports and its CLI answers; then attempt a small
single-node all-reduce sweep via its own runner. Report via stdout (v5 ships logs)
+ MLflow params.
"""
import importlib
import os
import signal
import subprocess
import sys


def log_receipt(params):
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        c = MlflowClient()
        for k, v in params.items():
            c.log_param(run_id, k, v)
    except Exception as e:
        print(f"receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


results = {"m5_python": sys.executable}
try:
    import torch
    results["m5_torch"] = torch.__version__
except ImportError:
    results["m5_torch"] = "MISSING"

mod = None
for name in ["param_bench.train.comms.pt.comms", "et_replay", "param_bench"]:
    try:
        mod = importlib.import_module(name)
        results["m5_module"] = name
        break
    except ImportError as e:
        print(f"M5 import {name}: {e}", flush=True)
results.setdefault("m5_module", "NOT_IMPORTABLE")

for exe in ["comms.py", "comm_replay"]:
    r = subprocess.run(["which", exe], capture_output=True, text=True)
    print(f"M5 which {exe}: {r.stdout.strip() or 'not found'}", flush=True)

if results["m5_module"] != "NOT_IMPORTABLE" and results["m5_torch"] != "MISSING":
    cmd = [sys.executable, "-m", "param_bench.train.comms.pt.comms",
           "--master-ip", os.environ.get("MASTER_ADDR", "127.0.0.1"),
           "--b", "64M", "--e", "256M", "--collective", "all_reduce",
           "--backend", "nccl", "--device", "cuda"]
    print("M5 attempting:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
    print("M5 output tail:", *tail, sep="\n", flush=True)
    results["m5_run_exit"] = r.returncode

log_receipt(results)
print(f"M5_PARAMBENCH_PROBE_DONE {results}", flush=True)
