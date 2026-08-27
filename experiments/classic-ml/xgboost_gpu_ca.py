"""XGBoost single-GPU training on AIR (CLI/`air run` path) — the script twin of the
`xgboost_gpu_train` notebook, for automated end-to-end GPU verification.

The docs sgc-xgboost example is a *notebook* run interactively on serverless GPU; the Jobs API
cannot select the `databricks_ai_v5` base env (docs/06), so the reproducible GPU path from a repo
checkout is the `air` CLI. Same training as the notebook: California Housing regression, GPU hist,
checkpoints to a UC Volume, RMSE eval + checkpoint reload. Reads the dataset from the Volume
(egress-free); prints pass-gated sentinels so `air logs <run_id>` is the receipt.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from xgboost.callback import TrainingCheckPoint

try:
    from sklearn.metrics import root_mean_squared_error as _rmse
except ImportError:  # sklearn < 1.4
    from sklearn.metrics import mean_squared_error
    _rmse = lambda a, b: float(mean_squared_error(a, b)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="parquet on a UC Volume (features + MedHouseVal)")
    ap.add_argument("--ckpt-dir", required=True, help="UC Volume dir for TrainingCheckPoint")
    ap.add_argument("--num-round", type=int, default=200)
    ap.add_argument("--target", default="MedHouseVal")
    args = ap.parse_args()

    # 1. Confirm GPU — fail fast, loud, before any training. NB: on the `air` CLI path the default
    # interpreter has no torch (torch lives in the databricks-ai venv), so we detect the GPU via
    # nvidia-smi, not torch.cuda. xgboost bundles its own CUDA — a successful device="cuda" train
    # below is the proof the GPU was actually used.
    import subprocess
    gpu_enabled = bool(xgb.build_info().get("USE_CUDA", False))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        gpus = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        gpus = []
    dev = gpus[0] if gpus else "none"
    print(f"RESULT xgboost_version={xgb.__version__} USE_CUDA={gpu_enabled} "
          f"gpu_count={len(gpus)} device={dev}", flush=True)
    assert gpu_enabled and gpus, "not a GPU-enabled XGBoost node — expected AI env v5 + A10"

    # 2. Load from the Volume (no external egress).
    pdf = pd.read_parquet(args.data)
    feat = [c for c in pdf.columns if c != args.target]
    X, y = pdf[feat].to_numpy("float32"), pdf[args.target].to_numpy("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain, dtest = xgb.DMatrix(X_tr, label=y_tr), xgb.DMatrix(X_te, label=y_te)
    print(f"PHASE data_ready rows={len(X)} features={len(feat)} "
          f"train={len(X_tr)} test={len(X_te)} src={args.data}", flush=True)

    # 3. Train on GPU, checkpoint to the Volume.
    os.makedirs(args.ckpt_dir, exist_ok=True)
    params = {"tree_method": "hist", "device": "cuda", "objective": "reg:squarederror",
              "eval_metric": "rmse", "max_depth": 6, "learning_rate": 0.1}
    ckpt = TrainingCheckPoint(directory=args.ckpt_dir, name="model", interval=50)
    t0 = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=args.num_round,
                        evals=[(dtrain, "train"), (dtest, "eval")], callbacks=[ckpt],
                        verbose_eval=50)
    fit_s = time.time() - t0
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "model_*")))
    print(f"PHASE fit_done seconds={fit_s:.1f} "
          f"checkpoints={[os.path.basename(c) for c in ckpts]}", flush=True)

    # 4. Evaluate.
    rmse = _rmse(y_te, booster.predict(dtest))
    print(f"RESULT test_rmse={rmse:.4f}", flush=True)

    # 5. Recover from a checkpoint (proves the Volume checkpoints are loadable resume points).
    resume_rmse = None
    if ckpts:
        rnd = lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        pick = min(ckpts, key=lambda p: abs(rnd(p) - 150))
        resumed = xgb.Booster()
        resumed.load_model(pick)
        resume_rmse = _rmse(y_te, resumed.predict(dtest))
        print(f"RESULT checkpoint={os.path.basename(pick)} round={rnd(pick)} "
              f"resumed_rmse={resume_rmse:.4f}", flush=True)

    ok = bool(gpus) and gpu_enabled and rmse < 0.6 and len(ckpts) > 0
    if ok:
        print("XGB_GPU_TRAIN_OK", flush=True)  # pass-gated: unreachable unless all checks held
    print(f"VERDICT: {'ACCEPTED' if ok else 'REJECTED'} rmse={rmse:.4f} gpu={dev} "
          f"checkpoints={len(ckpts)} fit_s={fit_s:.1f}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
