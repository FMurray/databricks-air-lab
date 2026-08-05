"""Synthetic timeseries prior for continued pretraining (T3 treatment).

Generates multi-class timeseries-classification TASKS in the PFN/TimEE spirit: each task's
classes are distinct generative regimes (AR coefficients, seasonality, trend, noise scale),
and each sample is a fixed-length window — so a task is exactly a (rows × timesteps) table,
the same framing bench_timeseries.py uses for UCR eval. Pure numpy, seeded, no RNG in the
task->data mapping beyond the task seed (task k is reproducible).

Used two ways:
  - `sample_task(seed)` -> (X, y) one synthetic TSC task (for fine-tune-path continued training)
  - `--emit-dir N` -> write N tasks as CSVs (for a trainer-side custom prior, if forked)
"""

import argparse

import numpy as np


def _regime_series(rng, length, ar, season_period, season_amp, trend, noise):
    """One series from one generative regime."""
    x = np.zeros(length + 24)
    for t in range(2, length + 24):
        x[t] = ar[0] * x[t - 1] + ar[1] * x[t - 2] + rng.normal(0, noise)
    x = x[24:]
    t = np.arange(length)
    if season_period > 0:
        x = x + season_amp * np.sin(2 * np.pi * t / season_period)
    return x + trend * t / length


def sample_task(seed, n_classes=None, n_per_class=None, length=None):
    """One multi-class TSC task: classes = distinct regimes. Returns (X (n,length), y)."""
    rng = np.random.default_rng(seed)
    n_classes = n_classes or int(rng.integers(3, 11))
    n_per_class = n_per_class or int(rng.integers(40, 200))
    length = length or int(rng.integers(48, 145))          # matches UCR eval range 46-140

    X, y = [], []
    for c in range(n_classes):
        crng = np.random.default_rng(seed * 1_000_003 + c)
        # per-class regime: stationary-ish AR(2) + optional seasonality + trend
        a1 = crng.uniform(-0.9, 0.95)
        a2 = crng.uniform(-0.5, min(0.5, 0.95 - abs(a1)))
        regime = dict(
            ar=(a1, a2),
            season_period=int(crng.choice([0, 0, 6, 12, 24])),
            season_amp=float(crng.uniform(0.3, 2.0)),
            trend=float(crng.uniform(-2, 2)),
            noise=float(crng.uniform(0.3, 1.5)),
        )
        for i in range(n_per_class):
            srng = np.random.default_rng(seed * 7_000_003 + c * 10_007 + i)
            X.append(_regime_series(srng, length, **regime))
            y.append(c)
    X = np.asarray(X)
    y = np.asarray(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit-dir", default="")
    p.add_argument("--n-tasks", type=int, default=64)
    p.add_argument("--seed0", type=int, default=1000)
    args = p.parse_args()
    if args.emit_dir:
        import os
        import pandas as pd
        os.makedirs(args.emit_dir, exist_ok=True)
        for k in range(args.n_tasks):
            X, y = sample_task(args.seed0 + k)
            df = pd.DataFrame(X, columns=[f"t{i}" for i in range(X.shape[1])])
            df["__label__"] = y
            df.to_csv(f"{args.emit_dir}/task_{k:04d}.csv", index=False)
        print(f"TS_PRIOR emitted {args.n_tasks} tasks to {args.emit_dir}")
    else:  # smoke: print shape stats for 5 tasks
        for k in range(5):
            X, y = sample_task(args.seed0 + k)
            print(f"task {k}: X={X.shape} classes={len(set(y))} "
                  f"label_counts={np.bincount(y).tolist()}")


if __name__ == "__main__":
    main()
