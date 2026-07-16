# How logging works on AIR (native path)

From docs (`ai-runtime/tracking-observability`, `cli/track-runs`) + empirically verified against our
run 37776040541298 / MLflow run f38a5161be834c54b9219452a543084b (2026-07-16, e2-demo-field-eng).

## The dual-run model

Every `air run` submission = **one Databricks job run + one MLflow run**, linked both ways:
- **Job run** (Jobs & Pipelines page): execution truth — status, compute, retries, driver output.
- **MLflow run**: experiment truth — params, metrics, system metrics, artifacts (including logs).
- Each retry = new MLflow run in the same experiment; all retries share one job run.
- Notebook `@distributed` calls likewise auto-create an MLflow run (nested child if a run is active);
  default experiment `/Users/{user}/{notebook-name}`, override via `mlflow.set_experiment()` or
  `MLFLOW_EXPERIMENT_NAME`.

## Where the bytes go

stdout/stderr of your `command` is captured by the **SGC launcher** and lands in three places:
1. Live streaming: `air logs <run-id>` (node 0), `--node N` for others, `--download-to DIR` for files.
2. **MLflow artifacts**: `logs/node_N/logs-N.chunk.txt` (your output, chunked) +
   `logs/node_N/databricks-launcher.log-N.chunk.txt` (platform-side lifecycle log). Stored under
   `dbfs:/databricks/mlflow-tracking/<exp>/<run>/artifacts`.
3. Jobs run page shows driver output.

The launcher log is gold for debugging platform issues — observed contents: condition markers
(`LauncherStarted`), pod name/rank, a `SGC_DEBUG_INFO` block with every correlation ID
(`job_id`, `job_task_run_id`, `mlflow_experiment_id`, `mlflow_run_id`, `run_id` (job run),
`workload_id`, `workspace_id`), network-block gate check, and per-secret resolution lines
(`Resolved secret for env var name=... scope=... key=...`).

User-log preamble shows the runtime contract: FUSE-ready gate → `PWD=/mnt/work` → hyperparameters
materialized to `/Workspace/Users/<user>/.air/cli_launch/<experiment>/<run>_<hash>/training_config.yaml`
(the `parameters:` YAML block → `HYPERPARAMETERS_PATH` env var).

In-container env: `PYTHONUNBUFFERED=1` preset (no lost buffered prints);
`MOSAICML_LOG_DIR=/databricks/customer-logs`, `MOSAICML_GPU_LOG_FILE_PREFIX=gpu_` (MosaicML-lineage
log shipping dir — TODO: confirm files written there get shipped too).

## Metrics

- **System metrics: automatic, zero config**, per node in MLflow (`system/node_0/...`): CPU %, memory,
  disk usage/available, network rx/tx, and per-GPU utilization %, memory MB/%, power W/%.
  (Verified present even on our 14s smoke run.)
- **Custom metrics**: platform exposes `MLFLOW_RUN_ID` to your process; attach with
  `mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])` and log normally. Multi-node: all nodes
  share one run — **log from rank 0 only**. `report_to="mlflow"` in HF TrainingArguments just works.
- Autologging (e.g. `mlflow.pytorch.autolog()`) recommended; MLflow ≥3.7 for DL workflow patterns.
- Watch the **10M metric-step limit** — don't log every batch on long runs.

## Notebook-side extras

- GPU resources pane (right side panel): per-GPU util/memory/temp, 10s polling, 2h history,
  pauses after 5 min inactivity. Works single- and multi-node.

## Checkpoints (not logs, but same story)

- Write to UC Volumes via `serverless_gpu.data.UCVolumeWriter/Reader` + Torch DCP; stages through
  NVMe-backed `/tmp` (faster than direct FUSE writes); `.metadata` published only after shards land
  (atomicity). `dcp.async_save` for background uploads (needs `cpu:gloo,cuda:nccl` process group).
  Requires GPU env v5+ / serverless_gpu 0.5.16+. Data-pipeline position is NOT checkpointed — resume
  from epoch boundary or track sample offsets yourself.

## Native vs OTEL→Zerobus (why our experiment still matters)

Native logging is **per-run and UI-centric**: great for a person debugging one run; logs are text
chunks in MLflow artifacts, metrics live in the MLflow store. What it does NOT give you:
- SQL over structured events **across all runs/teams** (fleet view: who OOMed this week?)
- Joins against `system.billing.usage` / pool utilization (the [the customer] chargeback/visibility ask)
- A story for **non-Python/Java workloads** (no MLflow autolog; stdout chunks only)
OTLP→Zerobus lands structured logs/metrics in governed Delta tables and fills exactly that gap —
complementary, not competing. (See `experiments/docker-otel-zerobus/NOTES.md` for its current
SP-token blocker.)

## Open follow-ups

- Does writing files to `MOSAICML_LOG_DIR` ship them anywhere queryable?
- Log retention: chunk rotation size, and artifact lifetime policy for `dbfs:/databricks/mlflow-tracking`.
- Multi-node: verify `logs/node_N` fan-out and `air logs --node N` on a 2-node CLI run (pending 8xH100 test).
- Notebook path: where does `@distributed` fan-out output surface vs CLI chunks?
