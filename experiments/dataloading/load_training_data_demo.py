# Databricks notebook source
# MAGIC %md
# MAGIC # Loading training data on AIR — Delta table vs UC Volume
# MAGIC
# MAGIC Two ways to feed data into a serverless-GPU training job, side by side, against real datasets
# MAGIC already staged in `air_lab`:
# MAGIC
# MAGIC | Path | What lives there | This demo's dataset |
# MAGIC |---|---|---|
# MAGIC | **Delta table** | structured / tabular rows, queryable + versioned | `credit_card_transactions` — 284,807 anonymized card txns, `Class` = fraud label |
# MAGIC | **UC Volume files** | large sharded/unstructured blobs, streamed | `training_data/tokens/` — 40 parquet shards, 640k × 1024 GPT-2 token sequences |
# MAGIC
# MAGIC **Environment:** attach the **AI base environment (v5)** with an accelerator (A10 is plenty for
# MAGIC this demo). v5 has `torch`, `pyarrow`, and — inside the notebook `@distributed` runtime —
# MAGIC `serverless_gpu.data`. (v4 silently breaks GPU-job storage egress; use v5.)
# MAGIC
# MAGIC ---
# MAGIC ## When to use which
# MAGIC
# MAGIC **Reach for a Delta table when the data is structured and the *pipeline before training* matters.**
# MAGIC You want SQL joins/filters/feature engineering, schema enforcement, time-travel + reproducible
# MAGIC snapshots (`VERSION AS OF`), and the whole thing (or a materialized feature set) fits comfortably
# MAGIC in driver/executor memory as pandas or Spark DataFrames. Classic ML (XGBoost, sklearn), tabular
# MAGIC foundation models, and anything you'd rather curate in the lakehouse than on disk. The data path
# MAGIC is **Spark Connect → pandas** — there is no Photon/SparkML acceleration on the GPU node, so do the
# MAGIC heavy SQL/feature work on classic/serverless CPU and hand the GPU job a clean, right-sized table.
# MAGIC
# MAGIC **Reach for a UC Volume when the data is large, sharded, and you stream it every step.**
# MAGIC Token shards, image/webdataset tars, audio, pre-tokenized corpora — files too big to hold in
# MAGIC memory, read epoch after epoch. `serverless_gpu.data.UCVolumeDataset` + `DataLoader` give you
# MAGIC **automatic sharding across ranks × workers** (drop the hand-written `DistributedSampler`) and a
# MAGIC **local cache** so multi-epoch training doesn't re-read the volume over FUSE. This is the path for
# MAGIC multi-node pretraining/fine-tuning where the dataset never fits in RAM.
# MAGIC
# MAGIC **Rule of thumb:** *fits in memory after a SQL step → Delta. Streamed shards that never fit → Volume.*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Widgets default to the datasets staged for this session; override to point elsewhere.

# COMMAND ----------

dbutils.widgets.text("catalog", "forrest_serverless_stable_2_catalog", "catalog")
dbutils.widgets.text("schema", "air_lab", "schema")
dbutils.widgets.text("delta_table", "credit_card_transactions", "Delta table (tabular)")
dbutils.widgets.text("volume", "training_data", "UC volume")
dbutils.widgets.text("tokens_subdir", "tokens", "token-shards subdir in volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE_FQN = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('delta_table')}"
TOKENS_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{dbutils.widgets.get('volume')}/{dbutils.widgets.get('tokens_subdir')}"
print("Delta table :", TABLE_FQN)
print("Token shards:", TOKENS_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC # Approach A — Delta table (structured / tabular)
# MAGIC
# MAGIC Read the table with Spark, do any last filtering/feature work in SQL or the DataFrame API, then
# MAGIC materialize to pandas → tensors for the GPU. Here: standardize `Amount`, split, build a
# MAGIC `torch` `DataLoader` — training-ready for a tabular model.

# COMMAND ----------

sdf = spark.read.table(TABLE_FQN)
print(f"{TABLE_FQN}: {sdf.count():,} rows, {len(sdf.columns)} cols")
# Class balance — this dataset is heavily imbalanced (fraud is rare); note it before training.
display(sdf.groupBy("Class").count().orderBy("Class"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Feature/label prep in Spark, then one hop to pandas.** Do the heavy lifting (joins, filters,
# MAGIC aggregations) in Spark on CPU compute; only the final, right-sized frame crosses to the GPU node.

# COMMAND ----------

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# Everything except the label is a numeric feature (Time, V1..V28, Amount).
feature_cols = [c for c in sdf.columns if c != "Class"]
pdf = sdf.toPandas()  # ~285k×31 fits easily in driver memory — the Spark Connect → pandas hop

X = pdf[feature_cols].to_numpy(dtype=np.float32)
y = pdf["Class"].to_numpy(dtype=np.float32)

# Standardize Amount + Time (the PCA V-features are already ~centered); fit on train only.
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
cut = int(0.8 * len(X))
tr, te = idx[:cut], idx[cut:]

scale_cols = [feature_cols.index("Time"), feature_cols.index("Amount")]
mu = X[tr][:, scale_cols].mean(0)
sd = X[tr][:, scale_cols].std(0) + 1e-8
for split in (tr, te):
    X[np.ix_(split, scale_cols)] = (X[np.ix_(split, scale_cols)] - mu) / sd

train_ds = TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]))
train_dl = TorchDataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=2)

xb, yb = next(iter(train_dl))
print(f"train={len(tr):,}  test={len(te):,}  batch={tuple(xb.shape)}  "
      f"train fraud rate={y[tr].mean():.4%}")
print("→ feed train_dl to your model's training loop (class-weight the loss for the imbalance).")

# COMMAND ----------

# MAGIC %md
# MAGIC **Delta extras worth showing the room:**
# MAGIC - **Reproducible snapshots:** `spark.read.option("versionAsOf", 0).table(...)` pins the exact
# MAGIC   data a run trained on — pair the version with the MLflow run for audit.
# MAGIC - **Push work down:** filter/aggregate in SQL *before* `.toPandas()` so you only pull what the
# MAGIC   GPU needs (`spark.sql("SELECT ... WHERE ...")`).
# MAGIC - **Size ceiling:** `.toPandas()` collects to the driver. If the curated set stops fitting in
# MAGIC   memory, that's the signal to shard it to a **Volume** and switch to Approach B.

# COMMAND ----------

# MAGIC %md
# MAGIC # Approach B — UC Volume files (large / sharded / streamed)
# MAGIC
# MAGIC Files in a Volume are read as a stream. First, the portable view — any interpreter can read the
# MAGIC parquet shards with `pyarrow`:

# COMMAND ----------

import pyarrow.dataset as pads

shards = [f.path for f in dbutils.fs.ls(TOKENS_DIR) if f.path.endswith(".parquet")]
print(f"{len(shards)} shards under {TOKENS_DIR}")

ds = pads.dataset(TOKENS_DIR.replace("dbfs:", ""), format="parquet")
batch = next(ds.to_batches(batch_size=4))
row0 = batch.column("input_ids")[0].as_py()
print(f"schema: {ds.schema.field('input_ids').type}")
print(f"first sequence: len={len(row0)}  head={row0[:12]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The AIR-native path: `serverless_gpu.data`
# MAGIC
# MAGIC `UCVolumeDataset(dir)` enumerates the files and, under a distributed job, **auto-partitions them
# MAGIC across `world_size × num_workers` with no gaps or duplicates** — so you *don't* write a
# MAGIC `DistributedSampler` — and caches each file locally on first touch. You wrap it with a tiny
# MAGIC decode step (parquet bytes → tensors) and hand it to `serverless_gpu.data.DataLoader`.
# MAGIC
# MAGIC > ⚠️ **Two sharp edges (both verified in `experiments/dataloading/NOTES.md`):**
# MAGIC > 1. **`num_workers` must be identical on every rank.** The partition is a global stride over
# MAGIC >    `world × num_workers`; a mismatch silently drops/duplicates samples.
# MAGIC > 2. **CLI / `torchrun` path only:** `serverless_gpu.data` lives in the base interpreter, not the
# MAGIC >    `databricks-ai` torch venv — you need the `PYTHONPATH` bridge to import it there. Inside
# MAGIC >    *this notebook's* `@distributed` runtime torch and `serverless_gpu` already coexist, so the
# MAGIC >    import below works here.

# COMMAND ----------

import io, pyarrow.parquet as pq

def decode_shard(path: str):
    """parquet shard path -> torch LongTensor of shape [num_seqs, 1024]."""
    with open(path.replace("dbfs:", ""), "rb") as fh:
        tbl = pq.read_table(io.BytesIO(fh.read()), columns=["input_ids"])
    return torch.tensor(tbl.column("input_ids").to_pylist(), dtype=torch.long)

try:
    try:
        from serverless_gpu.data import UCVolumeDataset, DataLoader
    except ImportError:
        from databricks.serverless_gpu.data import UCVolumeDataset, DataLoader

    base = UCVolumeDataset(TOKENS_DIR.replace("dbfs:", ""))
    # NUM_WORKERS must match across ranks — set it once, from config, on every rank.
    NUM_WORKERS = 4
    loader = DataLoader(base, batch_size=1, num_workers=NUM_WORKERS,
                        collate_fn=lambda paths: decode_shard(paths[0]))
    seqs = next(iter(loader))
    print(f"✅ UCVolumeDataset streamed a shard: {tuple(seqs.shape)} (seqs × seq_len)")
    print("   → in your training loop, iterate `loader` and chunk each shard into token batches.")
except Exception as e:
    print(f"⏭️  serverless_gpu.data not available here ({type(e).__name__}: {e}).")
    print("   Falling back to the pyarrow reader above — that streams the same shards in any env.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Why the Volume path for tokens:** 655M tokens across 40 shards is far past what you'd collect to
# MAGIC pandas, and pretraining reads it many times. `UCVolumeDataset` gives you cross-rank sharding for
# MAGIC free on multi-node and serves epoch 2+ from local cache instead of re-reading FUSE. Add shards to
# MAGIC the Volume and the dataset grows with no code change.

# COMMAND ----------

# MAGIC %md
# MAGIC # Which one? — the short version
# MAGIC
# MAGIC | | **Delta table** | **UC Volume files** |
# MAGIC |---|---|---|
# MAGIC | Data shape | structured rows | large sharded blobs (tokens/images/audio) |
# MAGIC | Read pattern | query → collect once | stream every step, multi-epoch |
# MAGIC | Fits in memory? | yes (after a SQL step) | no — that's the point |
# MAGIC | Sharding across GPUs | you handle it (or Spark) | **automatic** (world × workers) |
# MAGIC | Superpowers | SQL, schema, time-travel, joins | local cache, no `DistributedSampler`, grows by dropping files |
# MAGIC | Watch out for | `.toPandas()` driver memory ceiling; no GPU-side Spark accel | `num_workers` must match ranks; CLI needs the `PYTHONPATH` bridge; use env v5 |
# MAGIC | Typical job | classic ML, tabular FM, feature-joined fine-tune | multi-node LLM pretraining / large fine-tune |
# MAGIC
# MAGIC When a curated Delta set outgrows driver memory, shard it to a Volume and move to Approach B —
# MAGIC that migration *is* the transition from single-node to multi-node training on AIR.
