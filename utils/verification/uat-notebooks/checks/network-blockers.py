# Databricks notebook source
# MAGIC %md
# MAGIC # UAT check: network blockers
# MAGIC Env: any serverless (CPU fine; run on serverless **GPU** to reproduce the AIR-node view).
# MAGIC Returns JSON via `dbutils.notebook.exit` for the DRIVER; also printable standalone.
# MAGIC Expected-broken list as of 2026-07-24 — see `docs/06-uat-suite.md`.
# MAGIC
# MAGIC Evidence design, for environments where the notebook output is all we get back:
# MAGIC - every probe line is **timestamped** with per-probe elapsed — a slow pass is visible
# MAGIC   signal, and the gap between lines locates a hang without any other logs;
# MAGIC - probes with no inner timeout (the Spark reads) get a **stall alarm** — a hang is a
# MAGIC   *finding*, not a wait: "stalls" and "refused" are different network postures;
# MAGIC - each result is mirrored to an MLflow **param** the moment it lands (control-plane
# MAGIC   write, works where the S3 artifact path is blocked), so partial receipts survive a
# MAGIC   job-timeout kill that would eat the exit JSON.

# COMMAND ----------

import contextlib, json, signal, socket, time, urllib.request

T0 = time.time()
results = {}

# best-effort receipt run — its URL prints up front so the output names where durable
# evidence lives even if the run is killed mid-check
try:
    import mlflow
    _receipt = mlflow.start_run(run_name="uat-network-blockers")
    print(f"receipt run_id: {_receipt.info.run_id}")
except Exception as e:  # no mlflow in env / tracking unreachable — output-only mode
    mlflow = None
    print(f"no receipt run ({type(e).__name__}: {e}) — notebook output is the only evidence")

_probe_t0 = T0

def record(name, ok, detail=""):
    detail = f"{time.time() - _probe_t0:.1f}s — {str(detail)[:200]}"
    results[name] = {"ok": (None if ok is None else bool(ok)), "detail": detail}
    icon = {True: "✅", False: "❌", None: "⏭️"}[results[name]["ok"]]
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.1f}s] {icon} {name} — {detail}")
    if mlflow:
        with contextlib.suppress(Exception):
            mlflow.log_param(f"probe.{name}", f"{icon} {detail}"[:490])

def run_probe(name, fn, stall_s=90):
    """fn() -> (ok, detail). Raises past stall_s become an explicit STALL failure."""
    global _probe_t0
    _probe_t0 = time.time()
    def _alarm(sig, frame):
        raise TimeoutError(f"STALL — no response in {stall_s}s (hang, not a refusal)")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(stall_s)
    try:
        record(name, *fn())
    except Exception as e:
        record(name, False, f"{type(e).__name__}: {e}")
    finally:
        signal.alarm(0)

# COMMAND ----------

# control: metastore + samples (should always pass)
run_probe("control_samples_read", lambda: (
    True, f"{spark.sql('SELECT count(*) c FROM samples.nyctaxi.trips').collect()[0].c} rows visible"))

# COMMAND ----------

# target catalog storage (expected ❌ until bucket policy fixed; stall alarm because a
# blocked S3 path can hang the read for minutes rather than refuse)
def _catalog():
    spark.sql("SELECT * FROM mkazia_lw2_catalog_7474656734648830.default.t1 LIMIT 1").collect()
    return True, "catalog bucket reachable — blocker FIXED"
run_probe("catalog_bucket_read", _catalog)

# COMMAND ----------

# workspace root-storage bucket (expected ❌ — the log/artifact blackhole)
def _root_storage():
    socket.create_connection(
        ("mkazia-lw2-workspace-root-storage.s3-fips.us-east-1.amazonaws.com", 443), timeout=15).close()
    return True, "TCP connect OK — log delivery should work now"
run_probe("root_storage_tcp443", _root_storage, stall_s=30)

# COMMAND ----------

# PyPI egress (expected ❌ — blocks environment.dependencies on AIR runs)
def _pypi():
    with urllib.request.urlopen("https://pypi.org/simple/", timeout=15) as r:
        return r.status == 200, f"HTTP {r.status}"
run_probe("pypi_egress", _pypi, stall_s=30)

# COMMAND ----------

# MLflow artifact upload (expected ❌; skip if env lacks mlflow) — logs into the receipt
# run: params above prove control-plane tracking works even when this S3 leg fails
if mlflow is None:
    _probe_t0 = time.time()
    record("mlflow_artifact_upload", None, "SKIPPED — no mlflow/tracking in this env")
else:
    def _artifact():
        with open("/tmp/receipt.txt", "w") as f:
            f.write("uat network check")
        mlflow.log_artifact("/tmp/receipt.txt")
        return True, "upload OK"
    run_probe("mlflow_artifact_upload", _artifact, stall_s=60)

# COMMAND ----------

elapsed = round(time.time() - T0, 1)
print(f"NETWORK_BLOCKERS_DONE in {elapsed}s — probe time only; provisioning/env build happen "
      "before this notebook starts, so a slow *job* with a fast probe total is launch overhead")
if mlflow:
    with contextlib.suppress(Exception):
        mlflow.log_param("probe_elapsed_s", elapsed)
        mlflow.end_run()
dbutils.notebook.exit(json.dumps(
    {"check": "network-blockers", "elapsed_s": elapsed, "results": results}))
