# Pick your compute

Goal: choose the smallest accelerator that answers your question. Escalate shape only when the
question requires it.

## The hardware

| Accelerator | GPUs/node | GPU memory | Distributed |
|---|---|---|---|
| `GPU_1xA10` | 1 | 24 GB | no |
| `GPU_1xH100` | 1 | 80 GB | no |
| `GPU_8xH100` | 8 | 8×80 GB | single-node `@distributed(gpus=8)`; multi-node via CLI, multiples of 8×H100 |

Only A10 and H100 exist today. Multi-node: field guide reports max 16 nodes / 128 GPUs, sweet
spot 3–8 nodes, AWS-only (reported, not lab-verified at the max).

## Rules of thumb

- **A10 first.** Most probes, benches, and classic-ML jobs fit; it's the cheapest way to be wrong.
- **H100 when the memory or the question demands it** — an 80 GB card for an OOM question, Hopper
  for a Hopper-kernel question. Don't assume A10 results transfer (a known docs-notebook XGBoost
  job hangs on H100 but works on A10 — open-q #12).
- **8×H100 / multi-node only for genuinely distributed work** — see
  [Run multi-node training](../cookbook/run-multi-node-training.md).

## A measured example: how far each card actually goes

TabICL inference, 100 features (✅ measured 2026-07-22, e2-demo-field-eng,
runs 1022517780681952 + A10 override; full table in `experiments/foundation-models/NOTES.md`):

| Rows | A10 24 GB | H100 80 GB |
|---|---|---|
| 50K | 22.4 GB — fits, 28s | 54.4 GB, 10s |
| 100K | chunking kicks in, 114s | 64.6 GB, 23s |
| 200K | **killed — host RAM OOM (exit 137)** | chunks, 97s |
| 600K | — | 44.2 GB, 422s — no OOM |

Two portable lessons:

!!! warning "The node's CPU RAM can be the binding constraint"
    The A10 death at 200K rows was **host-RAM OOM (exit 137, no Python traceback)** — the offload
    path exhausted the small node's CPU memory, not the GPU's. Exit 137 ≠ CUDA OOM;
    see [Debug a failed run](../cookbook/debug-a-failed-run.md).

!!! warning "CPU:GPU ratio is fixed per accelerator type"
    You can't add dataloader CPUs to a GPU node. CPU-bound pipelines (synthetic data generation,
    heavy preprocessing) can starve an expensive GPU — measure GPU utilization vs. worker count
    before scaling up.

## Cost discipline

Teams default to H100s. MLflow logs per-GPU utilization automatically on every run
([Track metrics](../cookbook/track-metrics-with-mlflow.md)) — check whether you used 3% of an
H100 before booking a bigger one. Allocation managers: see
[Fleet ops](../fleet-ops/index.md) for the org-level view.
