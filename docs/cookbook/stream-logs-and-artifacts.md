# Stream logs and artifacts

Goal: know where every byte of your workload's output lands, and how to get it back.

✅ Verified against run 37776040541298 / MLflow run f38a5161… (2026-07-16, e2-demo-field-eng);
full notes in `reference/05-logging-observability.md`.

## The dual-run model

Every `air run` = **one job run + one MLflow run**, linked both ways:

- **Job run** (Jobs UI → *Job runs* tab): execution truth — status, compute, retries.
- **MLflow run**: experiment truth — params, metrics, artifacts, *including your logs*.
- Each retry = a new MLflow run in the same experiment; all retries share the one job run.

## Where stdout/stderr goes

Your `command`'s output is captured by the launcher and lands in three places:

```bash
air logs <run-id>                    # live stream (node 0 ONLY)
air logs <run-id> --node 1           # other nodes
air logs <run-id> --download-to DIR  # as files
```

1. The live stream above.
2. **MLflow artifacts**: `logs/node_N/logs-N.chunk.txt` — plus
   `logs/node_N/databricks-launcher.log-N.chunk.txt`, the platform-side lifecycle log.
3. The job run page (driver output).

`PYTHONUNBUFFERED=1` is preset in-container — you will not lose buffered prints on a crash.

## The launcher log is gold

When a run fails weirdly, read `databricks-launcher.log` before pinging anyone. Observed contents:

- lifecycle condition markers (`LauncherStarted`, FUSE-ready gate),
- an `SGC_DEBUG_INFO` block with **every correlation ID** — `job_id`, `job_task_run_id`,
  `mlflow_experiment_id`, `mlflow_run_id`, `workload_id`, `workspace_id` — exactly what an
  escalation asks for,
- per-secret resolution lines (`Resolved secret for env var name=… scope=…`) — confirms your
  `secrets:` block resolved without printing values.

## Archive the log with the finding

Logs on the workspace outlive the session, but if a run produced a fact worth keeping, copy the
raw log next to it (`experiments/<family>/run-<id>.log` in this repo). A claim someone can't
reconstruct from artifacts is a liability — see
[how to read the receipts](../reference/about-receipts.md).

## Next

- [Track metrics with MLflow](track-metrics-with-mlflow.md) — the numbers, not the text
- [Debug a failed run](debug-a-failed-run.md)
