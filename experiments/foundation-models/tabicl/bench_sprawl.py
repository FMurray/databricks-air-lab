"""Sprawl head-to-head: one TabICL checkpoint vs per-task XGBoost across marketing-ish tasks.

Story: N point models (each needing its own tuning pipeline) vs one foundation-model
checkpoint that "fits" via a forward pass. Records accuracy, time-to-model, GPU peak mem.

Datasets come from OpenML (needs egress; general egress from AIR verified 2026-07-16 in
the zerobus experiment). Swap in Delta tables via --delta-table catalog.schema.table:target
once the pilot uses customer-shaped data.
"""

import argparse
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# (name, openml data_id) — marketing-flavored, mostly binary targets
OPENML_TASKS = [
    ("bank-marketing", 1461),        # term-deposit propensity (the on-theme one)
    ("churn", 40701),                # telecom churn
    ("credit-g", 31),                # credit risk
    ("adult", 1590),                 # income propensity proxy
    ("click-prediction", 1220),      # ad click propensity
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


def scores(y_true, y_pred, y_proba):
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    try:
        if y_proba.shape[1] == 2:
            out["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def run_dataset(name, data_id, args, clf_tabicl):
    from xgboost import XGBClassifier

    bunch = fetch_openml(data_id=data_id, as_frame=True)
    X, y = bunch.data, LabelEncoder().fit_transform(bunch.target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    if len(X_tr) > args.max_train_rows:
        X_tr, y_tr = X_tr.iloc[: args.max_train_rows], y_tr[: args.max_train_rows]
    row = {"dataset": name, "n_train": len(X_tr), "n_features": X.shape[1]}

    # --- TabICL: "fit" = store context, predict = one forward pass
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        clf_tabicl.fit(X_tr, y_tr)
        proba = clf_tabicl.predict_proba(X_te)
    except Exception:  # some frames need the numeric encoding fallback
        X_tr_n, X_te_n = encode_for_xgb(X_tr, X_te)
        clf_tabicl.fit(X_tr_n, y_tr)
        proba = clf_tabicl.predict_proba(X_te_n)
    row["tabicl_s"] = time.time() - t0
    row.update({f"tabicl_{k}": v for k, v in scores(y_te, proba.argmax(1), proba).items()})
    if torch.cuda.is_available():
        row["tabicl_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # --- XGBoost, defaults (the "quick" baseline)
    X_tr_x, X_te_x = encode_for_xgb(X_tr, X_te)
    t0 = time.time()
    xgb = XGBClassifier(tree_method="hist", n_jobs=-1, eval_metric="logloss")
    xgb.fit(X_tr_x, y_tr)
    proba = xgb.predict_proba(X_te_x)
    row["xgb_default_s"] = time.time() - t0
    row.update({f"xgb_default_{k}": v for k, v in scores(y_te, proba.argmax(1), proba).items()})

    # --- XGBoost, tuned (the honest per-task pipeline cost)
    t0 = time.time()
    search = RandomizedSearchCV(
        XGBClassifier(tree_method="hist", n_jobs=-1, eval_metric="logloss"),
        XGB_SEARCH_SPACE, n_iter=args.tune_iters, cv=3, scoring="roc_auc_ovr_weighted",
        random_state=0, n_jobs=1,
    )
    search.fit(X_tr_x, y_tr)
    proba = search.predict_proba(X_te_x)
    row["xgb_tuned_s"] = time.time() - t0
    row.update({f"xgb_tuned_{k}": v for k, v in scores(y_te, proba.argmax(1), proba).items()})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-train-rows", type=int, default=30_000)
    p.add_argument("--tune-iters", type=int, default=15)
    p.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble size")
    args = p.parse_args()

    from tabicl import TabICLClassifier

    clf_tabicl = TabICLClassifier(n_estimators=args.n_estimators)
    rows = []
    for name, data_id in OPENML_TASKS:
        try:
            print(f"=== {name} (openml {data_id})")
            rows.append(run_dataset(name, data_id, args, clf_tabicl))
        except Exception:
            print(f"!!! {name} failed, skipping:\n{traceback.format_exc()}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(4).to_markdown(index=False))

    try:
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="tabicl-sprawl-bench")
        for r in rows:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"{r['dataset']}.{k}", v)
        df.to_csv("/tmp/sprawl_bench.csv", index=False)
        mlflow.log_artifact("/tmp/sprawl_bench.csv")
    except Exception:
        print("(mlflow logging unavailable — results printed above)")


if __name__ == "__main__":
    main()
