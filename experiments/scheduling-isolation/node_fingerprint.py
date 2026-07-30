"""Node fingerprint + isolation probe (UAT scheduling/isolation family).

Logs everything needed to decide, across two concurrent runs, whether they shared a node
and what each could see of the other: host identity (boot_id survives containerization
tricks better than hostname — AIR hostnames are identical across nodes), the GPU UUIDs
actually visible, and coarse blast-radius signals (process count, /tmp writability).
Receipts via MLflow params (durable channel); zero pip deps (nvidia-smi CLI, not pynvml).

Env knobs: HOLD_SECONDS (default 180) — stay alive so concurrent probes provably overlap;
the overlap window [hold_start_utc, hold_end_utc] is logged for the comparison.
"""
import os
import signal
import subprocess
import time
from datetime import datetime, timezone

HOLD_SECONDS = int(os.environ.get("HOLD_SECONDS", "180"))


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return "(unreadable)"


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout.strip() or r.stderr.strip())[:400]
    except Exception as e:
        return f"(failed: {type(e).__name__})"


fp = {
    "boot_id": read("/proc/sys/kernel/random/boot_id"),
    "machine_id": read("/etc/machine-id"),
    "hostname": sh("hostname"),
    "gpu_uuids": sh("nvidia-smi --query-gpu=uuid --format=csv,noheader | sort | paste -sd, -"),
    "gpu_count_visible": sh("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l").strip(),
    "gpu_names": sh("nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd, -"),
    # blast-radius signals: what else can this workload observe on the host?
    "proc_count": sh("ls /proc | grep -c '^[0-9]'").strip(),
    "other_gpu_procs": sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l").strip(),
    "pod_rank": os.environ.get("POD_RANK", os.environ.get("NODE_RANK", "")),
    "num_nodes": os.environ.get("NUM_NODES", ""),
}
fp["hold_start_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

for k, v in fp.items():
    print(f"RESULT {k}={v}", flush=True)

run_id = os.environ.get("MLFLOW_RUN_ID")
if run_id:
    signal.alarm(120)
    try:
        import mlflow
        mlflow.start_run(run_id=run_id)
        mlflow.log_params({k: str(v)[:450] for k, v in fp.items()})
        mlflow.end_run()  # REQUIRED: metrics-monitor thread otherwise keeps python alive
    except Exception as e:
        print(f"receipt logging failed: {e}", flush=True)
    finally:
        signal.alarm(0)

print(f"RESULT holding for {HOLD_SECONDS}s so concurrent probes overlap...", flush=True)
time.sleep(HOLD_SECONDS)

# stamp the end of the overlap window (second mlflow touch, own alarm)
if run_id:
    signal.alarm(120)
    try:
        import mlflow
        mlflow.start_run(run_id=run_id)
        mlflow.log_param("hold_end_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        mlflow.end_run()
    except Exception:
        pass
    finally:
        signal.alarm(0)
print("RESULT fingerprint=DONE", flush=True)
