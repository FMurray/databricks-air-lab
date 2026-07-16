# AIR Feature Surface

Distilled from public docs (docs.databricks.com/aws/en/machine-learning/ai-runtime/), the internal
field guide (go/air/enablement), and eng threads. Last refreshed: 2026-07-16.

## What it is

Serverless GPU compute for custom training: managed GPU infra + a deep-learning runtime + lakehouse-native
data access (UC/Spark Connect) + MLflow/Lakeflow integration. Positioned for custom-code training that the
managed Model Training / FM fine-tuning products don't cover.

Naming history: GPU Pools → AI Runtime → "Serverless GPU Compute" during Beta → AIR at Preview. Internal
docs use all three; billing surfaces both `product_features.ai_runtime.*` and `product_features.serverless_gpu.*`.

## Hardware

| Accelerator | GPUs/node | Memory | Distributed |
|---|---|---|---|
| `GPU_1xA10` | 1 | 24 GB | no |
| `GPU_1xH100` | 1 | 80 GB | no |
| `GPU_8xH100` | 8 | 8×80 GB | `@distributed(gpus=8)` single-node; multi-node via CLI only |

- Only A10 and H100 today. B200/B300 (Blackwell) being scoped — active [the customer] leadership escalation.
- Cross-region GPU borrowing during capacity crunch (possible egress cost; "Cloudless Compute" design covers this).
- Reserved capacity = pools, tied to **one workspace**; rebalancing needs eng + ~48h notice.

## Connectivity / entry points

Notebooks (interactive) · IDE via SSH tunnel · scheduled Jobs / Jobs API / DABs · `air` CLI (YAML workloads).

## Environments

- Two managed Python envs: **Standard** (minimal) and **Databricks AI** (PyTorch, Transformers, etc. preloaded). Env `version: "4"` default; v5+ adds features (e.g. `timeout` on `@distributed`).
- Deps: pip/uv style — inline specs, `-r requirements.txt`, wheels, index flags. Installed with `uv`.
- Workspace **base environments** for serverless GPU are supported.
- **Custom Docker** (Beta, CLI only): `environment.docker_image.url` after `air register image`. Constraints: Docker Hub only (no ECR/GCR/GHCR), <20 GB, `WORKDIR` ignored, mutually exclusive with `dependencies`/`version`.
- Notebook jobs: no Environments-panel deps — use `%pip install`; no auto-recovery for incompatible package versions.

## Distributed API (`serverless_gpu`, Beta)

```python
from serverless_gpu import distributed
from serverless_gpu.compute import GPUType

@distributed(gpus=8, gpu_type=GPUType.H100, timeout=3*3600)  # timeout: v5+, default 3h, None to disable
def train(...): ...

result = train.distributed(...)   # creates/nests MLflow run, fans out, syncs env, collects results
```

- `gpu_type` optional (auto-detected from attached accelerator; mismatch = failure).
- Works with DDP / FSDP / DeepSpeed; Ray, HF Transformers, Composer supported.
- Notebook decorator path = single node (8xH100) only. Multi-node = CLI.

## AIR CLI workload YAML

`air -h config` / `air -h config.<section>` = source of truth. Top-level fields:

```yaml
experiment_name: my-training
environment:
  version: "4"
  dependencies: [torch, transformers]   # or -r requirements.txt / wheels / pip flags
  env_variables: {BATCH_SIZE: "32"}
  secrets: {HF_TOKEN: my_scope/hf_token}
  # OR docker_image: {url: docker.io/org/img:tag}   # excludes dependencies/version
compute:
  num_accelerators: 16            # whole-node multiples for multi-GPU types
  accelerator_type: GPU_8xH100    # 16 ⇒ 2 nodes: THE multi-node path
code_source:
  type: snapshot
  snapshot: {root_path: ., git: {branch: main, remote: true}, include_paths: [src/]}
command: torchrun --nproc_per_node=8 $CODE_SOURCE_PATH/train.py
parameters: {training: {batch_size: 32}}
max_retries: 2
timeout_minutes: 90
usage_policy_name: my-team-policy   # cost attribution → serverless usage/budget policy
mlflow_run_name: baseline-001
```

Note `command` is **arbitrary bash** — this is the only documented escape hatch for non-Python entrypoints.

## Data access

- Tabular: Spark Connect → Delta tables → pandas/NumPy.
- Files/unstructured: UC Volumes; `serverless_gpu.data.UCVolumeDataset` for cached/streaming file loading.
- RDMA + high-performance data loading for distributed jobs.

## Observability / governance

- MLflow runs auto-created; checkpointing supported; logs via workspace.
- Billing: `system.billing.usage` with `product_features.ai_runtime.compute_type` / `product_features.serverless_gpu.workload_type` (see `utils/`).
- `usage_policy_name` → serverless usage policies (Private Preview-era feature; enrollment matters).
- Known gap: reservation pools bill as a single aggregate record — no per-user/per-workload attribution yet (P0 for [the customer]).
- Access control gap: serverless GPU access rides the workspace serverless flag — no GPU-only entitlement subset yet (P0 for [the customer]; eng thread in `#eng-serverless-compute-cost-control` 2026-07-08).

## Limits (as of 2026-07)

- Regions (AWS): us-west-2, us-west-1, us-east-1, us-east-2, ca-central-1, sa-east-1.
- Max runtime 7 days → checkpoint/restart.
- Compliance: most standards incl. PCI (approved + enabled); NOT FedRAMP High / DoD IL5.
- Capacity can be constrained; cross-region fallback possible.

## Language support

Python-first, full stop: env model is pip/uv, distributed API is Python-only. **No first-class Java/Scala/JVM
surface.** Escape hatch = custom Docker + arbitrary bash `command` via CLI. See `docs/03-workload-matrix.md`
and `experiments/multi-language/`.
