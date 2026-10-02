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

### What counts toward the limit

The quota is **total metric steps per run**: every logged value, across every key. System metrics
count the same as your own — `system/…` keys get no special treatment — and every node logs its own
`system/node_N/…` set into the shared run. On an 8-GPU node that's roughly 50 values per sample
(estimate), so system metrics alone use ~18k steps/hour per node at a 10s interval.

The limit in effect is printed in the error; trust that over the docs (1M has been seen in the
field):

```
BAD_REQUEST: Unable to add N metric(s) to run <run_id>. The maximum allowed total metric steps per
run is <limit>.
```

Once it's hit, every metric write fails but **training keeps running** — MLflow charts freeze at
that instant, which looks like a hung run. Grep node logs for `maximum allowed total metric steps`
before debugging a "hang".

### Staying under it

Each `system/…` key gets one point every `sampling_interval × samples_before_logging` seconds.

```yaml
# AIR workload YAML
env_variables:
  MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL: "60"   # seconds; default 10
```

```python
# or in code, before start_run
mlflow.set_system_metrics_sampling_interval(60)
mlflow.set_system_metrics_samples_before_logging(1)  # samples averaged per logged point
```

Turn system metrics off with `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false`,
`mlflow.disable_system_metrics_logging()`, or `mlflow.start_run(..., log_system_metrics=False)`.

!!! note "Composer overrides these"
    Composer's `MLFlowLogger` calls `set_system_metrics_sampling_interval(5)` and
    `set_system_metrics_samples_before_logging(6)` in its constructor (one point per 30s), which
    likely clobbers a YAML env var. Re-set after constructing the logger, or pass
    `log_system_metrics=False`. Unverified hands-on.

To go beyond the limit, open a support ticket.

### "Failed to abort upload … Presigned URLs API is not enabled"

Harmless. The Databricks SDK tries a presigned multipart upload, presigned URLs aren't available on
serverless GPU, so it aborts (that also fails — hence the warning) and falls back to a single-shot
Files API upload. Look for `Falling back to single-shot upload` right after it. Older SDKs
(< 0.72.0) could fail the upload outright here; upgrade `databricks-sdk` if the upload errors.
It has nothing to do with the metric limit.

## Notebook extra

The GPU resources pane (right-side panel) gives per-GPU util/memory/temp at 10s polling with 2h
history, single- and multi-node. It pauses after 5 min of inactivity.

## When MLflow isn't enough

MLflow is per-run and UI-centric. For **SQL across all runs and teams** — who OOMed this week,
utilization by team, joins against billing — you want structured telemetry in Delta:
[Ship telemetry to Delta](ship-telemetry-to-delta.md). For frameworks that assume wandb, either
disable it (`WANDB_MODE=disabled`) or accept that step metrics won't reach MLflow without a shim.
