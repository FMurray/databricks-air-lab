# Databricks notebook source
# MAGIC %md
# MAGIC # UAT check: network blockers
# MAGIC Env: any serverless (CPU fine; run on serverless **GPU** to reproduce the AIR-node view).
# MAGIC Returns JSON via `dbutils.notebook.exit` for the DRIVER; also printable standalone.
# MAGIC Expected-broken list as of 2026-07-24 — see `docs/06-uat-suite.md`.

# COMMAND ----------

import json, socket, urllib.request

results = {}

def record(name, ok, detail=""):
    results[name] = {"ok": bool(ok), "detail": str(detail)[:200]}
    print("✅" if ok else "❌", name, "—", str(detail)[:200])

# COMMAND ----------

# control: metastore + samples (should always pass)
try:
    n = spark.sql("SELECT count(*) c FROM samples.nyctaxi.trips").collect()[0].c
    record("control_samples_read", True, f"{n} rows visible")
except Exception as e:
    record("control_samples_read", False, e)

# COMMAND ----------

# target catalog storage (expected ❌ until bucket policy fixed)
try:
    spark.sql("SELECT * FROM mkazia_lw2_catalog_7474656734648830.default.t1 LIMIT 1").collect()
    record("catalog_bucket_read", True, "catalog bucket reachable — blocker FIXED")
except Exception as e:
    record("catalog_bucket_read", False, e)

# COMMAND ----------

# workspace root-storage bucket (expected ❌ — the log/artifact blackhole)
try:
    socket.create_connection(
        ("mkazia-lw2-workspace-root-storage.s3-fips.us-east-1.amazonaws.com", 443), timeout=15).close()
    record("root_storage_tcp443", True, "TCP connect OK — log delivery should work now")
except Exception as e:
    record("root_storage_tcp443", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# PyPI egress (expected ❌ — blocks environment.dependencies on AIR runs)
try:
    with urllib.request.urlopen("https://pypi.org/simple/", timeout=15) as r:
        record("pypi_egress", r.status == 200, f"HTTP {r.status}")
except Exception as e:
    record("pypi_egress", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# MLflow artifact upload with fail-fast alarm (expected ❌; skip if env lacks mlflow)
import signal, time
try:
    import mlflow
except ImportError:
    mlflow = None
    results["mlflow_artifact_upload"] = {"ok": None, "detail": "SKIPPED — no mlflow in this env"}
    print("⏭️ mlflow_artifact_upload — skipped (no mlflow in env)")

if mlflow is not None:
    def _alarm(sig, frame):
        raise TimeoutError("upload exceeded 60s")
    signal.signal(signal.SIGALRM, _alarm)
    with open("/tmp/receipt.txt", "w") as f:
        f.write("uat network check")
    signal.alarm(60)
    try:
        with mlflow.start_run(run_name="uat-network-blockers"):
            t0 = time.time()
            mlflow.log_artifact("/tmp/receipt.txt")
            record("mlflow_artifact_upload", True, f"OK in {time.time()-t0:.1f}s")
    except Exception as e:
        record("mlflow_artifact_upload", False, f"{type(e).__name__}: {e}")
    finally:
        signal.alarm(0)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"check": "network-blockers", "results": results}))
