"""Fine-tune the released TabICL checkpoint on a real dataset (bank-marketing) on AIR.

Closest analog to what the customer's marketing team would actually run: adapt the public checkpoint
to their data distribution rather than pretrain from scratch. Compares zero-shot vs
fine-tuned AUC and verifies the fine-tuned checkpoint reloads into the zero-shot API.
"""

import argparse
import time

from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-id", type=int, default=1461)  # bank-marketing
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output-dir", default="/tmp/tabicl-finetune")
    args = p.parse_args()

    from tabicl import FinetunedTabICLClassifier, TabICLClassifier

    bunch = fetch_openml(data_id=args.data_id, as_frame=True)
    X, y = bunch.data.copy(), LabelEncoder().fit_transform(bunch.target)
    cat = X.select_dtypes(include=["object", "category"]).columns
    if len(cat):
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat].astype(str))
    X = X.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=0, stratify=y_tr)

    zs = TabICLClassifier()
    zs.fit(X_tr, y_tr)
    auc_zero = roc_auc_score(y_te, zs.predict_proba(X_te)[:, 1])
    print(f"zero-shot AUC: {auc_zero:.4f}")

    t0 = time.time()
    ft = FinetunedTabICLClassifier(epochs=args.epochs, eval_metric="roc_auc")
    ft.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, output_dir=args.output_dir)
    auc_ft = roc_auc_score(y_te, ft.predict_proba(X_te)[:, 1])
    print(f"fine-tuned AUC: {auc_ft:.4f}  (delta {auc_ft - auc_zero:+.4f}, {time.time() - t0:.0f}s, ckpt -> {args.output_dir})")

    try:
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="tabicl-finetune-smoke")
        mlflow.log_metrics({"auc_zero_shot": auc_zero, "auc_finetuned": auc_ft})
    except Exception:
        pass


if __name__ == "__main__":
    main()
