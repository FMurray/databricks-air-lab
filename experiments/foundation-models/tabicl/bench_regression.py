"""Regression head-to-head: TabICLRegressor vs per-task XGBoost on public regression tasks.

Companion to bench_sprawl.py (classification). Motivation: 85-90 of the consolidation-target
models are regression + classification; every prior accuracy receipt is classification-only.
Records RMSE, R², time-to-model, GPU peak. OpenML fetch needs egress (run on e2).
"""

import argparse
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder

# (name, openml data_id) — classic OpenML regression sets; per-task try/except skips bad ids
OPENML_TASKS = [
    ("cpu-act", 197),          # system activity → utilization
    ("pol", 201),              # telecom
    ("elevators", 216),        # control
    ("house-sales", 42731),    # King County house prices (marketing-adjacent value regression)
    ("diamonds", 42225),       # price regression, mixed dtypes
]

XGB_SEARCH_SPACE = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [3, 4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}


def encode_for_xgb(X_train: pd.DataFrame, X_test: pd.DataFrame):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))
    return X_train.astype(float), X_test.astype(float)


def reg_scores(y_true, y_pred, y_std):
    """RMSE normalized by target std so tasks are comparable; plus R²."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"nrmse": rmse / y_std, "r2": float(r2_score(y_true, y_pred))}


def run_dataset(name, data_id, args, reg_tabicl):
    from xgboost import XGBRegressor

    bunch = fetch_openml(data_id=data_id, as_frame=True)
    X, y = bunch.data, pd.to_numeric(bunch.target, errors="coerce").values
    keep = ~np.isnan(y)
    X, y = X[keep], y[keep]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    if len(X_tr) > args.max_train_rows:
        X_tr, y_tr = X_tr.iloc[: args.max_train_rows], y_tr[: args.max_train_rows]
    y_std = float(np.std(y_te)) or 1.0
    row = {"dataset": name, "n_train": len(X_tr), "n_features": X.shape[1]}

    # --- TabICL regressor: fit = store context, predict = forward pass
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        reg_tabicl.fit(X_tr, y_tr)
        pred = reg_tabicl.predict(X_te)
    except Exception:  # numeric-encoding fallback, as in bench_sprawl
        X_tr_n, X_te_n = encode_for_xgb(X_tr, X_te)
        reg_tabicl.fit(X_tr_n, y_tr)
        pred = reg_tabicl.predict(X_te_n)
    row["tabicl_s"] = time.time() - t0
    row.update({f"tabicl_{k}": v for k, v in reg_scores(y_te, pred, y_std).items()})
    if torch.cuda.is_available():
        row["tabicl_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # --- XGBoost defaults
    X_tr_x, X_te_x = encode_for_xgb(X_tr, X_te)
    t0 = time.time()
    xgb = XGBRegressor(tree_method="hist", n_jobs=-1)
    xgb.fit(X_tr_x, y_tr)
    row["xgb_default_s"] = time.time() - t0
    row.update({f"xgb_default_{k}": v
                for k, v in reg_scores(y_te, xgb.predict(X_te_x), y_std).items()})

    # --- XGBoost tuned (the per-task pipeline cost)
    t0 = time.time()
    search = RandomizedSearchCV(
        XGBRegressor(tree_method="hist", n_jobs=-1), XGB_SEARCH_SPACE,
        n_iter=args.tune_iters, cv=3, scoring="neg_root_mean_squared_error",
        random_state=0, n_jobs=1)
    search.fit(X_tr_x, y_tr)
    row["xgb_tuned_s"] = time.time() - t0
    row.update({f"xgb_tuned_{k}": v
                for k, v in reg_scores(y_te, search.predict(X_te_x), y_std).items()})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-train-rows", type=int, default=30_000)
    p.add_argument("--tune-iters", type=int, default=15)
    p.add_argument("--n-estimators", type=int, default=8)
    args = p.parse_args()

    from tabicl import TabICLRegressor

    reg = TabICLRegressor(n_estimators=args.n_estimators)
    rows = []
    for name, data_id in OPENML_TASKS:
        try:
            print(f"=== {name} (openml {data_id})")
            rows.append(run_dataset(name, data_id, args, reg))
        except Exception:
            print(f"!!! {name} failed, skipping:\n{traceback.format_exc()}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    print(f"RESULT regression_bench tasks_completed={len(rows)}/{len(OPENML_TASKS)}")

    try:
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="tabicl-regression-bench")
        for r in rows:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"{r['dataset']}.{k}", v)
        df.to_csv("/tmp/regression_bench.csv", index=False)
        mlflow.log_artifact("/tmp/regression_bench.csv")
    except Exception:
        print("(mlflow logging unavailable — results printed above)")


if __name__ == "__main__":
    main()
