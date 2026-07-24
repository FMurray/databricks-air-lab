# Track metrics with MLflow

Goal: get training and system metrics into MLflow with the minimum wiring — and know which
readings to distrust.

## System metrics: free, automatic

Every run gets per-node system metrics in MLflow with **zero config** (`system/node_0/…`): CPU %,
memory, disk, network, and per-GPU utilization/memory/power. ✅ Verified present even on a 14-second
smoke run (2026-07-16).

!!! warning "GPU-util readings under-sample short steps"
    Sampled gauges can read **0%** during sub-second forward passes while reading 99–100% on long
    ones — ✅ observed both ways on real runs (2026-07-17/22). Don't conclude "idle GPU" from the
    gauge on a short-step job; check wall-time and GPU memory instead.

## Custom metrics: attach to the run the platform made

The platform creates the MLflow run and hands you its ID. Attach, then log normally:

```python
import mlflow, os

mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])
mlflow.log_metric("loss", loss, step=step)
```

- **Multi-node: all nodes share one run — log from rank 0 only.**
- HF Transformers: `report_to="mlflow"` in `TrainingArguments` just works.
- Autologging (`mlflow.pytorch.autolog()`) is the recommended default; MLflow ≥ 3.7 for the DL
  workflow patterns.

!!! warning "The 10M metric-step limit"
    Long runs that log every batch will hit it. Log every N steps.

## Notebook extra

The GPU resources pane (right-side panel) gives per-GPU util/memory/temp at 10s polling with 2h
history, single- and multi-node. It pauses after 5 min of inactivity.

## When MLflow isn't enough

MLflow is per-run and UI-centric. For **SQL across all runs and teams** — who OOMed this week,
utilization by team, joins against billing — you want structured telemetry in Delta:
[Ship telemetry to Delta](ship-telemetry-to-delta.md). For frameworks that assume wandb, either
disable it (`WANDB_MODE=disabled`) or accept that step metrics won't reach MLflow without a shim.
