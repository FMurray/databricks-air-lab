# Cookbook

Task-shaped recipes, each verified on a real workspace. Every recipe states its receipts;
anything unverified is labeled as such.

## Run things

- [Submit a workload](submit-a-workload.md) — the YAML field by field, with the schema quirks that cost us hours
- [Run multi-node training](run-multi-node-training.md) — torchrun across 8×H100 nodes, verified end-to-end
- [Use a custom Docker image](use-a-custom-docker-image.md) — native deps, non-Python stacks, cross-build traps
- [Run non-Python workloads](run-non-python-workloads.md) — the JVM ladder through the GA surface

## Observe things

- [Stream logs and artifacts](stream-logs-and-artifacts.md) — where every byte of output actually lands
- [Track metrics with MLflow](track-metrics-with-mlflow.md) — free system metrics, custom metrics, the gauges that lie
- [Ship telemetry to Delta](ship-telemetry-to-delta.md) — fleet-wide SQL over structured logs/metrics

## Survive things

- [Checkpoint past the 7-day cap](checkpoint-past-the-7-day-cap.md) — the runtime ceiling and how to duck it
- [Debug a failed run](debug-a-failed-run.md) — symptom → actual cause, from failures we actually hit
