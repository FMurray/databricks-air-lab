"""TabICL inference memory envelope: rows -> GPU peak memory / wall time, until OOM.

Feeds open-q #17 ("does TabICL need B300?") with concrete numbers on A10 (24GB) vs
H100 (80GB). Synthetic data at a fixed feature width; --offload tests offload_mode="auto".
"""

import argparse
import gc
import time

import numpy as np
import torch
from sklearn.datasets import make_classification

ROW_LADDER = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000, 600_000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, default=100)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--n-estimators", type=int, default=8)
    p.add_argument("--offload", action="store_true", help='pass offload_mode="auto"')
    args = p.parse_args()

    from tabicl import TabICLClassifier

    kwargs = {"n_estimators": args.n_estimators}
    if args.offload:
        kwargs["offload_mode"] = "auto"

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"device={gpu} total={total_gb:.0f}GB features={args.features} offload={args.offload}")
    results = []

    for n in ROW_LADDER:
        X, y = make_classification(
            n_samples=n + 2_000, n_features=args.features,
            n_informative=args.features // 2, n_classes=args.classes, random_state=0,
        )
        X_tr, y_tr, X_te = X[:n], y[:n], X[n:]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            clf = TabICLClassifier(**kwargs)
            clf.fit(X_tr, y_tr)
            clf.predict(X_te)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            results.append((n, f"{peak:.2f}", f"{dt:.1f}"))
            print(f"rows={n:>7,}  peak={peak:6.2f} GB  wall={dt:6.1f}s")
        except torch.cuda.OutOfMemoryError:
            results.append((n, "OOM", "-"))
            print(f"rows={n:>7,}  OOM")
            break

    try:
        import mlflow
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=f"tabicl-memprobe-{'offload' if args.offload else 'plain'}")
        mlflow.log_param("gpu", gpu)
        for n, peak, dt in results:
            if peak != "OOM":
                mlflow.log_metric("peak_gb", float(peak), step=n)
                mlflow.log_metric("wall_s", float(dt), step=n)
    except Exception:
        pass


if __name__ == "__main__":
    main()
