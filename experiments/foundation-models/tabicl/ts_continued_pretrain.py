"""T3 treatment: continued training on synthetic timeseries tasks, chained via model_path.

Proxy for the customer's plan (fusion weights + continued pretraining on a timeseries
prior): starting from the released checkpoint, sequentially fine-tune across K synthetic
regime-classification tasks from ts_prior.py, chaining weights task-to-task
(FinetunedTabICLClassifier(model_path=<prev>) -> fit -> newest ckpt -> next task).
Mechanism validated by T1 (released ckpt loads; model_path chaining per upstream API).

Honest labeling: this is the fine-tune-path analogue of continued pretraining, not the
trainer's prior machinery (that requires the customer's planned fork). The claim it
supports: "adapting the checkpoint to timeseries structure via continued training on
synthetic TS tasks changes downstream TS performance by Δ" — measured by re-running
bench_timeseries (--checkpoint) and bench_sprawl (--checkpoint, forgetting check) after.

Prints TS_CONTINUED_PRETRAIN_OK <final_ckpt> on success; exit 1 if no chain step succeeds.
"""

import argparse
import glob
import os
import time

from ts_prior import sample_task


def newest_ckpt(d):
    hits = sorted(glob.glob(os.path.join(d, "**", "*.ckpt"), recursive=True),
                  key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tasks", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--out", default="/tmp/ts-continued")
    args = p.parse_args()

    import pandas as pd
    from tabicl import FinetunedTabICLClassifier

    current = None  # None -> released checkpoint (auto-download)
    done = 0
    t_start = time.time()
    for k in range(args.n_tasks):
        X, y = sample_task(args.seed0 + k)
        Xdf = pd.DataFrame(X, columns=[f"t{i}" for i in range(X.shape[1])])
        out_dir = os.path.join(args.out, f"task_{k:03d}")
        try:
            ft = FinetunedTabICLClassifier(epochs=args.epochs, model_path=current)
            ft.fit(Xdf, y, output_dir=out_dir)
            nxt = newest_ckpt(out_dir)
            if nxt:
                current = nxt
                done += 1
            print(f"[task {k}] classes={len(set(y))} shape={X.shape} "
                  f"ckpt={'chained' if nxt else 'MISSING'} elapsed={time.time()-t_start:.0f}s",
                  flush=True)
        except Exception as e:  # noqa: BLE001 — record and continue the chain
            print(f"[task {k}] FAILED: {type(e).__name__}: {e}", flush=True)

    print(f"RESULT ts_continued tasks_chained={done}/{args.n_tasks} "
          f"wall={time.time()-t_start:.0f}s", flush=True)
    if not done or not current:
        raise SystemExit(1)
    print(f"TS_CONTINUED_PRETRAIN_OK {current}", flush=True)

    try:
        import mlflow
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            from mlflow.tracking import MlflowClient
            c = MlflowClient()
            c.log_param(run_id, "ts_continued_final_ckpt", current[:490])
            c.log_metric(run_id, "tasks_chained", done)
    except Exception as e:  # noqa: BLE001
        print(f"receipt logging failed: {e}", flush=True)


if __name__ == "__main__":
    main()
