# Running the UAT suite on this workspace

This folder is a synced copy of the `databricks-air-lab` repo (source of truth: Forrest Murray's
local git; synced 2026-07-24). It contains every UAT workload for the serverless-GPU (AIR)
acceptance window. Start here, then see `docs/06-uat-suite.md` for the full status matrix.

## One-time setup (5 minutes, on your laptop)

AIR workloads are submitted with the `air` CLI — a persistent "Run now" job wrapper does NOT
work (verified 2026-07-24: `gen_ai_compute_task` via jobs/create fails snapshot resolution;
only the CLI's SUBMIT_RUN path works).

```bash
uv tool install databricks-air                    # or: pipx install databricks-air
databricks auth login --host https://fe-sandbox-mkazia-lw2.cloud.databricks.com --profile mkazia-lw2
# get the repo: download this folder (Workspace UI → ⋮ → Export) or ask Forrest for the git remote
cd databricks-air-lab
```

## Run a UAT workload

```bash
air run --file workloads/<workload>.yaml -p mkazia-lw2
```

| UAT item | Command | Cost note |
|---|---|---|
| Runtime probe | `air run --file workloads/exec-probe.yaml -p mkazia-lw2` | 1×A10, ~10 min |
| A1 GPU burn (dry) | `... --file workloads/gpu-burn.example.yaml ...` | 1×A10 |
| A1 GPU burn (acceptance, per node) | add `--override compute.accelerator_type=GPU_8xH100 env_variables.EXPECT_GPUS=8 env_variables.BURN_SECONDS=900` | 8×H100 — coordinate first |
| A2 all-reduce (dry) | `... --file workloads/nccl-allreduce.example.yaml --override compute.accelerator_type=GPU_1xA10 compute.num_accelerators=2` | 2×A10 |
| A2 all-reduce (NVLink / fabric) | default YAML (8×H100) / `--override compute.num_accelerators=16` | H100 — coordinate first |
| W3 LoRA (dry) | `... --file workloads/lora-finetune.example.yaml ...` | 1×A10; needs HF egress |
| W5 XGBoost A10 control | `... --file workloads/xgboost-gpu.example.yaml ...` | 1×A10 |
| W5 XGBoost H100 repro | add `--override compute.accelerator_type=GPU_1xH100` | if it times out at `PHASE data_ready` → repro confirmed, escalate |
| W6 vLLM | `... --file workloads/vllm-smoke.example.yaml ...` | 1×A10; needs HF egress |
| Classic ML TabICL | `... --file workloads/tabicl-bench.yaml ...` | 1×A10 |

## Where results land

- **Job runs**: Jobs & Pipelines → **Job runs** tab (NOT the Jobs tab — `air run` creates
  one-time submits). Green run + `RESULT ...` lines in logs = the acceptance receipt.
- **MLflow**: experiment `/Users/<you>/air-lab-<workload>` — params, system metrics
  (GPU util/mem per node, automatic), log artifacts under `logs/node_0/`.
- Record every run (id + date + outcome) in the matching `experiments/*/NOTES.md` in the
  source repo — the git log is the lab notebook.

## Ground rules

- Dry-run on A10 before ANY H100 submission; H100/multinode runs cost real money — announce
  in the team channel before pool-scale sweeps.
- Never put tokens/secrets in workload YAML `command:` or dump env in job logs. Secrets go in
  secret scopes (scope `air_lab` exists).
- Known workspace blockers (2026-07-24): target catalog storage 403s from serverless (no Delta
  writes until the bucket policy is fixed); no SP for the OTEL/Zerobus pipeline yet; Databricks
  Apps disabled (no Training Hub). Details: `docs/06-uat-suite.md`.
- First GPU run of the day may fail with a generic "Gen AI Compute Task" INTERNAL_ERROR
  (observed platform flake here; a straight resubmit succeeded). Retry once before escalating.
