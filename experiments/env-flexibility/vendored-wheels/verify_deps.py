"""Vendored-deps verification (UAT W-vendor). Imports each vendored package — including a
compiled extension — from PYTHONPATH, prints RESULT receipts, and mirrors them to MLflow
params (the durable channel). Exit non-zero on any failure.

Env: VENDOR_PATH (informational — the launcher command already put it on PYTHONPATH).
"""
import os
import platform
import signal
import sys

results = {}
results["python_version"] = platform.python_version()
results["vendor_path"] = os.environ.get("VENDOR_PATH", "(not set)")
print(f"RESULT python={results['python_version']} vendor={results['vendor_path']}", flush=True)

failures = []

try:
    import emoji  # pure-python
    results["emoji"] = f"OK {emoji.__version__}"
except Exception as e:
    results["emoji"] = f"FAIL {type(e).__name__}: {e}"
    failures.append("emoji")

try:
    import xxhash  # compiled extension — proves platform/py tags match the node
    results["xxhash"] = f"OK {xxhash.VERSION} digest={xxhash.xxh64(b'air-lab').hexdigest()}"
except Exception as e:
    results["xxhash"] = f"FAIL {type(e).__name__}: {e}"
    failures.append("xxhash")

for k in ("emoji", "xxhash"):
    print(f"RESULT {k}={results[k]}", flush=True)

# MLflow param receipts; alarm so a blocked tracking call can't hang the run
run_id = os.environ.get("MLFLOW_RUN_ID")
if run_id:
    signal.alarm(120)
    try:
        import mlflow
        mlflow.start_run(run_id=run_id)
        mlflow.log_params({f"vendored_{k}": v[:450] for k, v in results.items()})
        mlflow.log_param("vendored_verdict", "FAIL: " + ",".join(failures) if failures else "PASS")
        mlflow.end_run()  # REQUIRED: metrics-monitor thread otherwise keeps python alive
    except Exception as e:
        print(f"receipt logging failed: {e}", flush=True)
    finally:
        signal.alarm(0)

assert not failures, f"vendored imports failed: {failures}"
print("RESULT vendored_deps=PASS", flush=True)
