"""Multi-class timeseries classification: TabICL (zero-shot + fine-tuned) vs XGBoost.

The BASELINE leg for the continued-pretraining experiment (T2 of the timeseries program):
public UCR multi-class datasets, each fixed-length univariate series windowed to a table
(one column per timestep — lengths 46-140 sit inside the customer's 100-500-col range).
The T3 treatment (continued-pretrain on a synthetic timeseries prior) re-runs this exact
eval and reports deltas; sprawl-bench re-run covers the catastrophic-forgetting check.

Metrics: accuracy + macro-F1 (multi-class). aeon downloads from
timeseriesclassification.com (needs egress — run on e2).
"""

import argparse
import time
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# UCR multi-class sets: (name, n_classes) — series length becomes the column count
UCR_TASKS = [
    ("ECG5000", 5),           # len 140 — medical-ish, 5-class
    ("ElectricDevices", 7),   # len 96 — usage curves, 7-class
    ("MedicalImages", 10),    # len 99 — 10-class
    ("Crop", 24),             # len 46 — 24-class, 24K series
    ("FaceAll", 14),          # len 131 — 14-class
]


def load_tabular(name):
    """UCR set -> (train_df, y_train, test_df, y_test); one column per timestep."""
    from aeon.datasets import load_classification

    X_tr, y_tr = load_classification(name, split="train")
    X_te, y_te = load_classification(name, split="test")
    # univariate (n, 1, length) -> (n, length)
    X_tr = pd.DataFrame(X_tr.squeeze(1), columns=[f"t{i}" for i in range(X_tr.shape[-1])])
    X_te = pd.DataFrame(X_te.squeeze(1), columns=[f"t{i}" for i in range(X_te.shape[-1])])
    le = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    return X_tr, le.transform(y_tr), X_te, le.transform(y_te)


def cls_scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def run_dataset(name, n_classes, args):
    from tabicl import TabICLClassifier
    from xgboost import XGBClassifier

    X_tr, y_tr, X_te, y_te = load_tabular(name)
    if len(X_tr) > args.max_train_rows:
        X_tr, y_tr = X_tr.iloc[: args.max_train_rows], y_tr[: args.max_train_rows]
    row = {"dataset": name, "n_classes": n_classes,
           "n_train": len(X_tr), "series_len": X_tr.shape[1]}

    # --- TabICL zero-shot
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    zs = TabICLClassifier(n_estimators=args.n_estimators,
                          model_path=args.checkpoint or None)
    t0 = time.time()
    zs.fit(X_tr, y_tr)
    pred = zs.predict(X_te)
    row["zeroshot_s"] = time.time() - t0
    row.update({f"zeroshot_{k}": v for k, v in cls_scores(y_te, pred).items()})
    if torch.cuda.is_available():
        row["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # --- TabICL fine-tuned ("oneshot", matching the customer's ~100-iter recipe).
    # Isolated: the upstream fine-tune path CUDA-asserts on some UCR sets (scatter-gather
    # index OOB in _train_forward, run 892979459471614) — a fine-tune crash must not lose
    # the zero-shot row.
    if args.finetune:
        try:
            from tabicl import FinetunedTabICLClassifier
            ft = FinetunedTabICLClassifier(epochs=args.finetune_epochs,
                                           model_path=args.checkpoint or None)
            t0 = time.time()
            ft.fit(X_tr, y_tr)
            pred = ft.predict(X_te)
            row["finetuned_s"] = time.time() - t0
            row.update({f"finetuned_{k}": v for k, v in cls_scores(y_te, pred).items()})
        except Exception as e:  # noqa: BLE001 — record the upstream failure, keep the row
            row["finetuned_error"] = f"{type(e).__name__}"
            print(f"    finetune leg failed (upstream): {type(e).__name__}: {e}", flush=True)

    # --- XGBoost on the same tabular framing
    t0 = time.time()
    xgb = XGBClassifier(tree_method="hist", n_jobs=-1, eval_metric="mlogloss")
    xgb.fit(X_tr, y_tr)
    pred = xgb.predict(X_te)
    row["xgb_s"] = time.time() - t0
    row.update({f"xgb_{k}": v for k, v in cls_scores(y_te, pred).items()})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-train-rows", type=int, default=30_000)
    p.add_argument("--n-estimators", type=int, default=8)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--finetune-epochs", type=int, default=100,
                   help="match the customer's oneshot-100-iters recipe")
    p.add_argument("--checkpoint", default="",
                   help="alternate TabICL checkpoint path (T3 treatment); default = released")
    p.add_argument("--run-name", default="tabicl-ts-baseline")
    args = p.parse_args()

    rows = []
    for name, n_classes in UCR_TASKS:
        try:
            print(f"=== {name} ({n_classes} classes)")
            rows.append(run_dataset(name, n_classes, args))
        except Exception:
            print(f"!!! {name} failed, skipping:\n{traceback.format_exc()}")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))
    print(f"RESULT ts_bench tasks_completed={len(rows)}/{len(UCR_TASKS)} "
          f"checkpoint={'custom' if args.checkpoint else 'released'}")
    if not rows:
        raise SystemExit(1)  # exit code derived from results — zero completions is a failure

    try:
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=args.run_name)
        for r in rows:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    mlflow.log_metric(f"{r['dataset']}.{k}", v)
        df.to_csv("/tmp/ts_bench.csv", index=False)
        mlflow.log_artifact("/tmp/ts_bench.csv")
    except Exception:
        print("(mlflow logging unavailable — results printed above)")


if __name__ == "__main__":
    main()
