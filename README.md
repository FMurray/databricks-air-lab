# Databricks AIR Lab

Workspace for testing, understanding, and building utilities for **Databricks AIR (AI Runtime)** —
the serverless GPU training stack (fka Serverless GPU Compute / GPU Pools).

Driving context: FDE/SSA engagement with **[the customer] [customer-org]** (see `docs/02-[the customer]-baseline.md`), which stresses
AIR across heterogeneous workloads: tabular foundation models, LLM fine-tuning, RL, classic ML
(XGBoost), and multi-language questions (Java in training).

## Layout

| Path | Purpose |
|---|---|
| `docs/` | Distilled knowledge: AIR feature surface, [the customer] baseline, workload-fit matrix, open questions |
| `workloads/` | AIR CLI workload YAML definitions (one dir per experiment as they mature) |
| `experiments/` | Hands-on tests by workload family: `foundation-models/`, `rl/`, `classic-ml/`, `multi-language/` |
| `utils/` | Utilities we build: attribution/billing queries, submission wrappers, monitoring |

## Quick reference

- Docs: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
- Internal field guide: go/air/enablement · SGC deck: go/sgc-deck
- Eng escalation channel: `#ai-runtime-oncall` · [the customer] channels: `#[the customer]-fde-[customer-org]-gpus`, `#[the customer]-model-training-gpu-blocker`
- CLI: `air` (experimental, in databricks/cli) — `air -h config` is the YAML schema source of truth

## Status

- Single-node AIR: Public Preview. `@distributed` (multi-GPU notebook API): Beta. Custom Docker: Beta.
- Accelerators: `GPU_1xA10`, `GPU_1xH100`, `GPU_8xH100` (multi-node only via CLI, e.g. 16×H100 = 2 nodes)
- Max workload runtime: 7 days (checkpoint + restart for longer)
