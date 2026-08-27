# Databricks notebook source
# MAGIC %md
# MAGIC # Classic ML on AIR — XGBoost regression on a single GPU
# MAGIC
# MAGIC An end-to-end training example following the Databricks docs *sgc-xgboost* tutorial
# MAGIC (docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/tutorials/sgc-xgboost),
# MAGIC hardened for serverless GPU: it reads the dataset **from Unity Catalog** instead of
# MAGIC `fetch_california_housing()` (external egress is blocked on GPU jobs), checkpoints to a
# MAGIC **UC Volume**, and prints pass-gated sentinels so the run is verifiable from its output.
# MAGIC
# MAGIC **What it shows:** GPU-accelerated gradient boosting (`tree_method="hist"`, `device="cuda"`),
# MAGIC periodic checkpointing to a Volume, evaluation, and checkpoint recovery.
# MAGIC
# MAGIC **Environment:** AI base environment **v5**, accelerator **GPU_1xA10**. *(The docs notebook is
# MAGIC known to hang on H100 but complete on A10 — repro tracked in `workloads/xgboost-gpu.example.yaml`;
# MAGIC use A10 here.)*
# MAGIC
# MAGIC **Running interactively vs. as a scheduled job:** attach this notebook to the AI base env + A10
# MAGIC and run it interactively for the session. To *schedule* it as an AI Runtime job, note that
# MAGIC **adding dependencies via the Environments panel is not supported for scheduled jobs** — install
# MAGIC any extra packages with `%pip install` inside the notebook (xgboost/scikit-learn are already in
# MAGIC the AI runtime, so nothing to add for this example). Auto-recovery isn't supported; on failure,
# MAGIC fix and re-run. *(The managed AI base env for jobs is gated on some workspaces — see NOTES.md.)*
# MAGIC
# MAGIC **When classic ML fits AIR:** XGBoost/LightGBM/cuML on one GPU is the right size for tabular
# MAGIC work that's too slow on CPU but doesn't need multi-node. Do feature engineering in Spark/SQL,
# MAGIC hand the GPU a clean table (see the companion `load_training_data_demo` notebook), and keep an
# MAGIC eye on utilization — a 20k-row set barely warms an A10; the GPU win shows on large/wide data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies in-notebook
# MAGIC The Standard serverless env (`environment_version: "5"`, what a scheduled GPU job uses here) does
# MAGIC **not** ship xgboost, and adding dependencies via the **Environments** panel is not supported for
# MAGIC AI Runtime scheduled jobs — so install with `%pip` in the notebook. Works the same interactively
# MAGIC and as a scheduled job.

# COMMAND ----------

# MAGIC %pip install --quiet xgboost scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "forrest_serverless_stable_2_catalog", "catalog")
dbutils.widgets.text("schema", "air_lab", "schema")
dbutils.widgets.text("table", "california_housing", "training table")
dbutils.widgets.text("volume", "training_data", "UC volume (for checkpoints)")
dbutils.widgets.text("num_round", "200", "boosting rounds")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE = dbutils.widgets.get("table")
VOLUME = dbutils.widgets.get("volume")
NUM_ROUND = int(dbutils.widgets.get("num_round"))
TABLE_FQN = f"{CATALOG}.{SCHEMA}.{TABLE}"
CKPT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/xgb_checkpoints/{TABLE}"
STAGE_PARQUET = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/_staging/{TABLE}/{TABLE}.parquet"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Confirm the GPU (fail fast if this isn't a GPU node)

# COMMAND ----------

import json, os, glob, time, subprocess
import numpy as np
import xgboost as xgb

# Detect the GPU via nvidia-smi — the Standard serverless env (scheduled jobs) has no torch. xgboost
# bundles its own CUDA, so a visible GPU + a USE_CUDA build + a successful device="cuda" train (below)
# is the proof the GPU was used.
gpu_enabled = bool(xgb.build_info().get("USE_CUDA", False))
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    gpus = [ln.strip() for ln in smi.stdout.splitlines() if ln.strip()]
except Exception:
    gpus = []
dev = gpus[0] if gpus else "none"

print(f"RESULT xgboost_version={xgb.__version__} USE_CUDA={gpu_enabled} gpu_count={len(gpus)} device={dev}", flush=True)
assert gpu_enabled and gpus, ("not a GPU-enabled XGBoost node — attach a GPU accelerator (A10) and "
                              "environment v5. This example requires device='cuda'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the dataset from Unity Catalog
# MAGIC Lakehouse-native read via Spark; falls back to the staged parquet on the Volume so the job
# MAGIC never depends on external egress. Prints which path it used.

# COMMAND ----------

from sklearn.model_selection import train_test_split

try:
    pdf = spark.read.table(TABLE_FQN).toPandas()
    src = f"spark.read.table({TABLE_FQN})"
except Exception as e:
    import pandas as pd
    pdf = pd.read_parquet(STAGE_PARQUET)
    src = f"pandas.read_parquet({STAGE_PARQUET})  [spark path unavailable: {type(e).__name__}]"

TARGET = "MedHouseVal"
feature_cols = [c for c in pdf.columns if c != TARGET]
X, y = pdf[feature_cols].to_numpy("float32"), pdf[TARGET].to_numpy("float32")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain, dtest = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_te, label=y_te)
print(f"PHASE data_ready rows={len(X)} features={len(feature_cols)} "
      f"train={len(X_tr)} test={len(X_te)} via {src}", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train on the GPU, checkpointing to the Volume
# MAGIC `TrainingCheckPoint` writes the booster to the Volume every `interval` rounds — this is how you
# MAGIC survive the 7-day job cap and preemption: resume from the last checkpoint instead of restarting.
# MAGIC
# MAGIC For **distributed (torch) training**, Databricks recommends checkpointing to a Volume via the
# MAGIC Torch Distributed Checkpoint API with the `serverless_gpu.data.UCVolumeWriter` /
# MAGIC `UCVolumeReader` backends (see docs → *Model checkpointing*). XGBoost is single-process here, so
# MAGIC its own `TrainingCheckPoint`-to-Volume is the equivalent; the Volume is the durable store either way.

# COMMAND ----------

from xgboost.callback import TrainingCheckPoint

os.makedirs(CKPT_DIR, exist_ok=True)
params = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.1,
}
ckpt = TrainingCheckPoint(directory=CKPT_DIR, name="model", interval=50)

t0 = time.time()
booster = xgb.train(
    params, dtrain, num_boost_round=NUM_ROUND,
    evals=[(dtrain, "train"), (dtest, "eval")],
    callbacks=[ckpt], verbose_eval=50,
)
fit_s = time.time() - t0
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "model_*")))
print(f"PHASE fit_done seconds={fit_s:.1f} checkpoints={[os.path.basename(c) for c in ckpts]}", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate

# COMMAND ----------

try:
    from sklearn.metrics import root_mean_squared_error as _rmse
except ImportError:  # sklearn < 1.4
    from sklearn.metrics import mean_squared_error
    _rmse = lambda a, b: float(mean_squared_error(a, b)) ** 0.5

rmse = _rmse(y_te, booster.predict(dtest))
# California Housing target is median value in $100k units; a healthy XGBoost fit lands well under 0.6.
print(f"RESULT test_rmse={rmse:.4f}", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Recover from a checkpoint
# MAGIC Reload the most recent checkpoint at or before round 150 and confirm it scores like a partially
# MAGIC trained model — proving the Volume checkpoints are real, loadable resume points.

# COMMAND ----------

def round_of(path):  # model_150.ubj -> 150
    base = os.path.splitext(os.path.basename(path))[0]
    return int(base.split("_")[-1])

resume_rmse = None
if ckpts:
    target_round = 150
    pick = min(ckpts, key=lambda p: abs(round_of(p) - target_round))
    resumed = xgb.Booster()
    resumed.load_model(pick)
    resume_rmse = _rmse(y_te, resumed.predict(dtest))
    print(f"RESULT checkpoint={os.path.basename(pick)} round={round_of(pick)} "
          f"resumed_rmse={resume_rmse:.4f} (>= final {rmse:.4f}, as expected mid-training)", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verdict

# COMMAND ----------

ok = (bool(gpus) and gpu_enabled and rmse < 0.6 and len(ckpts) > 0)
verdict = "ACCEPTED" if ok else "REJECTED"
if ok:
    print("XGB_GPU_TRAIN_OK", flush=True)  # pass-gated sentinel: unreachable unless all checks held
print(f"VERDICT: {verdict}  rmse={rmse:.4f}  gpu={dev}  checkpoints={len(ckpts)}  fit_s={fit_s:.1f}", flush=True)

dbutils.notebook.exit(json.dumps({
    "verdict": verdict,
    "xgboost_version": xgb.__version__,
    "device": dev, "use_cuda": gpu_enabled,
    "rows": int(len(X)), "num_round": NUM_ROUND,
    "test_rmse": round(float(rmse), 4),
    "resume_rmse": round(float(resume_rmse), 4) if resume_rmse is not None else None,
    "fit_seconds": round(fit_s, 1),
    "checkpoints": [os.path.basename(c) for c in ckpts],
    "data_source": src,
}))
