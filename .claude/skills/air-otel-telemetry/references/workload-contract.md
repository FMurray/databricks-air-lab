# AIR workload YAML contract for OTEL telemetry

Complete working example (verified against air CLI v0.1.0). Field placement matters:
`env_variables` and `secrets` are **top-level** — putting them under `environment:` is rejected
with "Unknown field".

```yaml
experiment_name: my-training

environment:
  version: "4"
  dependencies:
    - opentelemetry-sdk>=1.27
    - opentelemetry-exporter-otlp-proto-grpc>=1.27
    - requests
    - nvidia-ml-py
  # OR docker_image: {url: docker.io/<org>/<img>:<tag>}  (then bake deps into the image;
  #    dependencies/version are mutually exclusive with docker_image)

env_variables:                     # TOP-LEVEL (not under environment:)
  WORKSPACE_ID: "<numeric-workspace-id>"
  WORKSPACE_URL: "https://<workspace-host>"
  ZEROBUS_REGION: <region>         # must match the workspace's region
  OTEL_LOGS_TABLE: <catalog>.<schema>.air_otel_logs
  OTEL_METRICS_TABLE: <catalog>.<schema>.air_otel_metrics

secrets:                           # TOP-LEVEL; scope/key refs, injected as env vars
  DATABRICKS_CLIENT_ID: <scope>/zerobus_sp_client_id
  DATABRICKS_CLIENT_SECRET: <scope>/zerobus_sp_client_secret

compute:
  num_accelerators: 1
  accelerator_type: GPU_1xA10

code_source:
  type: snapshot
  snapshot:
    root_path: .                   # resolves relative to THIS YAML's directory, not CWD

# REQUIRED for identity: without a parameters block, AIR does not inject HYPERPARAMETERS_PATH
# and air.requester derivation silently yields nothing. Any content works.
parameters:
  telemetry:
    pipeline: otel-zerobus

# Never dump raw env in command — workload secrets are env vars and would land in job logs.
command: |
  python $CODE_SOURCE_PATH/train.py

timeout_minutes: 60
mlflow_run_name: my-run
```

## Training code side

Reuse the canonical module (`utils/telemetry/airtel.py` in-repo; the skill's `assets/airtel.py`
copy only outside the repo). In-repo snapshot workloads ship it like this:

```yaml
code_source:
  type: snapshot
  snapshot:
    root_path: ../..            # repo root relative to this YAML
    include_paths: [utils/telemetry/, <your-training-dir>/]
command: |
  export PYTHONPATH=$CODE_SOURCE_PATH
  python $CODE_SOURCE_PATH/<your-training-dir>/train.py
```

```python
import logging
from utils.telemetry import airtel  # or `import airtel` when the file sits next to train.py

telemetry = airtel.init(service_name="my-training")   # logs + metrics + GPU gauges + identity
log = logging.getLogger("train")

log.info("training started")                    # → logs table (with identity attrs)
telemetry.loss_gauge.set(0.42, {"epoch": 1})    # → metrics table
# ... training ...
telemetry.shutdown()                            # flush before exit or final batches are lost
```

## Attribute conventions (what lands on every row)

| Attribute | Source | Notes |
|---|---|---|
| `service.name` | `init(service_name=...)` | top-level `service_name` column, cluster key |
| `air.requester` | derived from HYPERPARAMETERS_PATH | self-reported; null ⇒ parameters block missing |
| `air.team` | `AIR_TEAM` env var if set | inject at submit time (e.g. Training Hub) |
| `air.mlflow_run_id` | `MLFLOW_RUN_ID` (AIR-injected) | join key to Jobs/MLflow/system tables |
| `air.node_rank` / `air.world_size` | `POD_RANK` / `WORLD_SIZE` | multi-node separation |
| `gpu` (metrics only) | NVML device index | per-GPU series |
