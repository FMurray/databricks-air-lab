# Native `ai_runtime_task` recipe bundle

This bundle packages the repository's existing RDMA recipe implementations into one reproducible
`.tgz` and exposes each recipe as a native Jobs `ai_runtime_task`. It follows the Databricks
[productionization pattern](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/productionizing-training-workloads):
`databricks bundle deploy` builds and uploads the archive, and AI Runtime runs a small
`command.sh` adapter on every node with the archive mounted at `CODE_SOURCE_PATH`.

The bundle is pinned to the safe sandbox target:

- profile: `fevm-forrest-2`
- host: `https://fevm-forrest-serverless-stable-2.cloud.databricks.com/`
- environment: AI Runtime v5
- retries: 0

Databricks CLI **1.6.0 or newer** is required. Earlier CLIs only recognize the retired private
`gen_ai_compute_task` bundle field and will reject `ai_runtime_task`. `manage.sh` checks this before
contacting the workspace.

## Validate and deploy

From this directory:

```bash
./manage.sh validate
./manage.sh deploy
```

Deployment creates or updates Jobs resources and the code artifact but does not start GPU compute.

## Run one recipe

```bash
./manage.sh run rdma-m1-soak --yes
```

The explicit `--yes` guard is intentional: `run` starts billable compute. `rdma-m1-soak` is the
two-node native-task example (`GPU_8xH100`, `accelerator_count: 16`). The other recipe keys and
their shapes are shown by `./manage.sh list`.

Every job writes to the submitting user's `/Users/<user>/air-lab-rdma-stress` MLflow experiment.
The existing recipe code remains under `experiments/`; the adapters only restore the environment
variables that were top-level fields in the AIR CLI YAMLs.

## What maps from an AIR CLI recipe

| AIR YAML | Native job bundle |
|---|---|
| `experiment_name` | `ai_runtime_task.experiment` |
| `compute.accelerator_type` | `deployments[].compute.accelerator_type` |
| `compute.num_accelerators` | `deployments[].compute.accelerator_count` |
| `command` and `env_variables` | a checked-in file under `commands/` |
| `environment.version/dependencies` | job `environments[]` + task `environment_key` |
| `max_retries` / `timeout_minutes` | task `max_retries` / `timeout_seconds` |
| snapshot include paths | the `tgz` artifact's `include` list |

Multi-node topology variables (`NUM_NODES`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT`) are injected by AI Runtime. They are not set in this bundle.
