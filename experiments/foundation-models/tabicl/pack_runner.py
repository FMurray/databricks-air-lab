"""Pack-runner PoL: bin-pack independent single-GPU TabICL fine-tunes onto one 8xH100 node.

The sizing question this answers: CCB's steady demand is A10-shaped (oneshot fine-tunes,
~4GB GPU peak measured) while the reservation is 8xH100 nodes — can one reserved node act
as "8+ A10s" for batches of small tasks? Phases:

  1. PREFETCH  — warm the OpenML dataset cache (serializes egress out of the timings).
  2. SOLO     — one task alone on GPU 0: the same-node reference wall (also warms the HF
                checkpoint cache for the pack phase).
  3. PACK     — one worker subprocess per GPU (CUDA_VISIBLE_DEVICES pinned), each claiming
                tasks from a shared directory via atomic rename. One task with a bogus
                OpenML id is injected on purpose: it must be RECORDED as a task error while
                every other task completes (per-task isolation, the bench_timeseries bug
                class), not kill the worker or the batch.

Headline metrics (logged to the ambient MLflow run): packed-vs-solo per-task slowdown,
tasks/hour/node, per-GPU assignment evidence. Prints `PACK_RUNNER_OK ...` only if the
assertions pass; anything else prints PACK_RUNNER_FAIL + reasons and exits 1.

Local pre-flight: `--stub --workers N` replaces the tabicl task with a timed busy-loop and
needs no GPU/tabicl/mlflow — verifies supervisor/claiming/isolation/sentinel logic only.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SENTINEL_OK = "PACK_RUNNER_OK"
SENTINEL_FAIL = "PACK_RUNNER_FAIL"
# Sprawl-bench marketing set; 1461 (bank-marketing) is the measured A10 baseline (94s @ epochs=10).
DATASETS = [1461, 40701, 31, 1590, 1220]
BAD_DATA_ID = 999999999  # injected failure: nonexistent OpenML id


def build_tasks(rounds: int, workers: int, epochs: int) -> list[dict]:
    tasks = [
        {"task_id": i, "data_id": DATASETS[i % len(DATASETS)], "epochs": epochs}
        for i in range(rounds * workers)
    ]
    tasks.append({"task_id": len(tasks), "data_id": BAD_DATA_ID, "epochs": epochs})
    return tasks


def run_one_task(task: dict, stub: bool) -> dict:
    """Execute a single fine-tune; returns a result row. Raises on task failure."""
    t0 = time.monotonic()
    if stub:
        if task["data_id"] == BAD_DATA_ID:
            raise ValueError("stub injected failure")
        time.sleep(0.2 + 0.05 * (task["task_id"] % 3))
        return {"wall_s": time.monotonic() - t0, "auc": 0.5, "gpu_peak_gb": 0.0}

    import torch
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from tabicl import FinetunedTabICLClassifier

    bunch = fetch_openml(data_id=task["data_id"], as_frame=True)
    X, y = bunch.data.copy(), LabelEncoder().fit_transform(bunch.target)
    cat = X.select_dtypes(include=["object", "category"]).columns
    if len(cat):
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat].astype(str))
    X = X.astype(float)
    # Binary AUC needs binary targets; multi-class sets in the list are all binary here.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=0, stratify=y_tr)

    torch.cuda.reset_peak_memory_stats()
    ft = FinetunedTabICLClassifier(epochs=task["epochs"], eval_metric="roc_auc")
    ft.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, output_dir=f"/tmp/pack/{task['task_id']}")
    proba = ft.predict_proba(X_te)
    auc = roc_auc_score(y_te, proba[:, 1]) if proba.shape[1] == 2 else float("nan")
    return {
        "wall_s": time.monotonic() - t0,
        "auc": auc,
        "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def worker_main(args) -> int:
    """Claim tasks from tasks_dir until empty; one JSONL row per task, errors recorded not raised."""
    tasks_dir, claimed = Path(args.tasks_dir), Path(args.tasks_dir) / "claimed"
    out = Path(args.results_dir) / f"worker-{args.gpu}.jsonl"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    with out.open("w") as f:
        while True:
            pending = sorted(p for p in tasks_dir.glob("task-*.json"))
            if not pending:
                break
            target = claimed / pending[0].name
            try:
                pending[0].rename(target)  # atomic claim; loser of the race retries
            except OSError:
                continue
            task = json.loads(target.read_text())
            row = {"task_id": task["task_id"], "data_id": task["data_id"], "gpu": args.gpu, "cvd": cvd}
            try:
                row.update(run_one_task(task, args.stub))
                row["status"] = "ok"
            except Exception as e:  # noqa: BLE001 — isolation is the point
                row.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            f.write(json.dumps(row) + "\n")
            f.flush()
    return 0


def supervisor_main(args) -> int:
    if args.stub:
        n_workers = args.workers or 2
    else:
        import torch

        n_workers = args.workers or torch.cuda.device_count()

    work = Path(args.workdir)
    tasks_dir, results_dir = work / "tasks", work / "results"
    (tasks_dir / "claimed").mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.rounds, n_workers, args.epochs)
    real_ids = sorted({t["data_id"] for t in tasks if t["data_id"] != BAD_DATA_ID})
    print(f"pack_runner: {len(tasks)} tasks ({len(tasks) - 1} real + 1 injected-bad) on {n_workers} workers")

    # 1. PREFETCH — dataset cache warm, egress out of the timings
    if not args.stub:
        from sklearn.datasets import fetch_openml

        t0 = time.monotonic()
        for did in real_ids:
            fetch_openml(data_id=did, as_frame=True)
        print(f"prefetch: {len(real_ids)} datasets in {time.monotonic() - t0:.0f}s")

    # 2. SOLO — same-node single-task reference on GPU 0 (also warms the HF checkpoint cache)
    solo_task = {"task_id": -1, "data_id": DATASETS[0], "epochs": args.epochs}
    t0 = time.monotonic()
    if args.stub:
        solo = {"wall_s": 0.25}
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        solo = run_one_task(solo_task, stub=False)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    print(f"solo: data_id={solo_task['data_id']} wall={solo['wall_s']:.1f}s")

    # 3. PACK — one pinned worker per GPU
    for t in tasks:
        (tasks_dir / f"task-{t['task_id']:03d}.json").write_text(json.dumps(t))
    procs = []
    pack_t0 = time.monotonic()
    for gpu in range(n_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--gpu", str(gpu),
               "--tasks-dir", str(tasks_dir), "--results-dir", str(results_dir)]
        if args.stub:
            cmd.append("--stub")
        procs.append(subprocess.Popen(cmd, env=env))
    exit_codes = [p.wait() for p in procs]
    pack_wall = time.monotonic() - pack_t0

    rows = []
    for f in sorted(results_dir.glob("worker-*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines()]
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] == "error"]
    injected_isolated = [r for r in errors if r["data_id"] == BAD_DATA_ID]
    gpus_used = sorted({r["cvd"] for r in rows})
    walls = sorted(r["wall_s"] for r in ok)
    median_wall = walls[len(walls) // 2] if walls else float("nan")
    tasks_per_hour = len(ok) / (pack_wall / 3600) if pack_wall else 0.0

    print(f"pack: {len(ok)} ok / {len(errors)} error in {pack_wall:.0f}s wall "
          f"| per-task wall min/median/max {walls[0]:.1f}/{median_wall:.1f}/{walls[-1]:.1f}s"
          if walls else "pack: no completed tasks")
    print(f"gpus used (CUDA_VISIBLE_DEVICES): {gpus_used}")
    print(f"packed/solo slowdown (median): {median_wall / solo['wall_s']:.2f}x | tasks/hour/node: {tasks_per_hour:.0f}")
    for r in errors:
        print(f"  task {r['task_id']} (data_id={r['data_id']}) -> {r['error']}")

    if not args.stub:
        try:
            import mlflow

            from mlflow_loggers import MLflowLogger

            lg = MLflowLogger(run_name="tabicl-pack-runner")
            lg.setup({"rounds": args.rounds, "epochs": args.epochs, "workers": n_workers,
                      "datasets": ",".join(map(str, real_ids)), "injected_bad_tasks": 1})
            lg.log_metrics({"solo_wall_s": solo["wall_s"], "pack_wall_s": pack_wall,
                            "packed_median_wall_s": median_wall,
                            "packed_over_solo": median_wall / solo["wall_s"],
                            "tasks_ok": len(ok), "tasks_error": len(errors),
                            "tasks_per_hour_node": tasks_per_hour})
            for r in ok:
                lg.log_metrics({"task_wall_s": r["wall_s"], "task_gpu_peak_gb": r["gpu_peak_gb"]},
                               step=r["task_id"])
            merged = work / "pack_results.jsonl"
            merged.write_text("\n".join(json.dumps(r) for r in rows))
            mlflow.log_artifact(str(merged))
            lg.finish()
        except Exception as e:  # noqa: BLE001 — evidence layer must not mask the verdict
            print(f"WARNING: mlflow logging failed: {type(e).__name__}: {e}")

    checks = {
        "all_workers_exited_zero": all(c == 0 for c in exit_codes),
        "distinct_gpu_per_worker": len(gpus_used) == n_workers,
        "all_real_tasks_ok": len(ok) == len(tasks) - 1,
        "injected_failure_isolated": len(injected_isolated) == 1 and len(errors) == 1,
        "solo_completed": solo["wall_s"] > 0,
    }
    if all(checks.values()):
        print(f"{SENTINEL_OK} workers={n_workers} distinct_gpus={len(gpus_used)} "
              f"tasks_ok={len(ok)}/{len(tasks) - 1} injected_failures_isolated=1 "
              f"slowdown={median_wall / solo['wall_s']:.2f}x tasks_per_hour_node={tasks_per_hour:.0f}")
        return 0
    print(f"{SENTINEL_FAIL} " + " ".join(f"{k}={v}" for k, v in checks.items()))
    return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3, help="real tasks = rounds x workers")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=0, help="0 = one per visible GPU")
    p.add_argument("--workdir", default="/tmp/pack-runner")
    p.add_argument("--stub", action="store_true", help="no-GPU pre-flight of supervisor logic")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--tasks-dir")
    p.add_argument("--results-dir")
    args = p.parse_args()
    sys.exit(worker_main(args) if args.worker else supervisor_main(args))


if __name__ == "__main__":
    main()
