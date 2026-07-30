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

## Zero-setup path: the notebook suite

No CLI needed for the environment/network checks and the notebook-surface GPU smoke:
**open `uat/DRIVER` (in this folder) and Run-all** on plain serverless. It reads
`uat/uat_config.py` (the suite config: env version, deps, and the check × accelerator-shape
matrix), launches each check notebook as a one-time notebook job **per shape** with the
accelerator pinned via the Jobs API (`compute.hardware_accelerator: GPU_1xA10|GPU_1xH100|
GPU_8xH100`), and prints one aggregated got-vs-expected matrix.

- **`shapes` widget** controls cost: default `GPU_1xA10` = cheap dry-run; `all` includes
  **8xH100 (real money — coordinate first)**; or a comma list like `GPU_1xA10,GPU_1xH100`.
- Verified 2026-07-24: A10 notebook job via this path → `1x NVIDIA A10G`, 66.4 TFLOPS bf16,
  NVML healthy (run 832802492734599).
- Add a check: drop a notebook in `uat/checks/` ending with `dbutils.notebook.exit(json)`,
  add an entry to `uat_config.py`. (The config is deliberately NOT named `environment.yml` —
  that filename is a live Databricks convention that overrides the folder's notebook
  environments and broke the CPU driver when we tried it.)

### Verify 20-node pool readiness (one notebook, one widget)

Open `uat/DRIVER`, set widget **`pool` = `on`** (leave `shapes` at its default), Run-all.
That arms `uat/checks/pool-readiness`, which submits — via the vendored `air` CLI, so the
CLI-only rule for distributed holds — a 20×(8×H100) burn sweep plus a 2-node NCCL fabric
probe, and verdicts purely from MLflow receipts: `burn=PASS` per node, **GPU-UUID
distinctness** (did we really touch 20 physical nodes), quota refusals classified, fabric
sentinel + busbw. ~15–20 min at defaults; one `pool_ready: true/false` row in the matrix.
**This takes the whole pool — announce in the team channel first.** With `pool` left `off`
the check is a free SKIP row. Knobs (edit `uat_config.py` params): `pool_nodes`,
`burn_seconds` (900 = A1 acceptance grade), `fabric_nodes` (up to 16).

The AIR *submission-path* workloads below still require the `air` CLI — and per the ground
rules, **all distributed multi-node runs are CLI-only** (no shapes beyond one node in the
notebook suite by design; `pool-readiness` fronts the CLI rather than using notebook shapes).

## Dependencies without PyPI — two recipes

**Run a workload that needs extra Python packages (vendored wheels):**
1. On your laptop, from repo root: `./experiments/env-flexibility/vendored-wheels/vendor_deps.sh`
   (edit `PACKAGES=(...)` in the script first; uv cross-targets linux/amd64 py3.12 from any host).
2. Make sure the `vendor/` dir is **committed / not gitignored** — the CLI's snapshot tar
   silently drops gitignored paths (verified).
3. In your workload YAML: `dependencies: []`, include your experiment dir in `include_paths`,
   and prefix the command with
   `export PYTHONPATH="$CODE_SOURCE_PATH/<your-dir>/vendor:$PYTHONPATH"`.
4. Submit as usual. Working example (verified PASS): `workloads/vendored-wheels-snapshot.example.yaml`.
   Full detail + UC-volume and default-package-repo alternatives:
   `experiments/env-flexibility/vendored-wheels/README.md`.

**Use the air CLI from a notebook (no laptop needed) — vendored CLI wheels:**
1. In any serverless notebook cell:
   `%pip install --no-index --find-links /Workspace/Shared/databricks-air-lab/uat/wheels databricks-air`
2. `air run --file /Workspace/Shared/databricks-air-lab/workloads/<x>.yaml` (auth is ambient
   in the notebook). Working example: `uat/checks/air-cli-from-notebook`.
   NB: the wheels live in **workspace files** (not a UC volume — volumes are blocked until the
   catalog-bucket fix); rebuild instructions: `uat/wheels/README.md` in the workspace mirror.

## Run a UAT workload

```bash
air run --file workloads/<workload>.yaml -p mkazia-lw2
```

| UAT item | Command | Cost note |
|---|---|---|
| Runtime probe | `air run --file workloads/exec-probe.yaml -p mkazia-lw2` | 1×A10, ~10 min |
| A1 GPU burn (dry) | `... --file workloads/gpu-burn.example.yaml ...` | 1×A10 |
| A1 GPU burn (acceptance, per node) | add `--override compute.accelerator_type=GPU_8xH100 compute.num_accelerators=8 env_variables.EXPECT_GPUS=8 env_variables.BURN_SECONDS=900` | 8×H100 — coordinate first |
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

- **All distributed multi-node training goes through the `air` CLI** (workload YAML,
  `compute.num_accelerators` spanning nodes). The notebook `@distributed` API is not
  production-ready for this engagement — don't build or demo multi-node on it. Notebooks
  here are for single-node/interactive work and the verification checks above.

- Dry-run on A10 before ANY H100 submission; H100/multinode runs cost real money — announce
  in the team channel before pool-scale sweeps.
- Never put tokens/secrets in workload YAML `command:` or dump env in job logs. Secrets go in
  secret scopes (scope `air_lab` exists).
- **Known workspace blockers (2026-07-24)** — details + receipts in `docs/06-uat-suite.md`:
  1. **Pin environment version 5** — env v4 breaks job-submitted GPU egress here (no run logs,
     artifact hangs, misleading TIMEDOUT/INTERNAL_ERROR). Repo YAMLs are pinned already;
     verify any hand-written YAML has `version: "5"`. PyPI is unreachable by design —
     vendor wheels, keep `dependencies: []`.
  2. Target catalog storage 403s from serverless SQL (no Delta writes).
  3. No SP yet for the OTEL/Zerobus pipeline; Databricks Apps disabled (no Training Hub).
- **Receipt pattern while logs are broken**: report results via MLflow params/metrics — the
  tracking API works. `mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])` then
  `mlflow.log_param("result", ...)`. Wrap any artifact/storage call in `signal.alarm` so a
  blocked upload fails fast instead of hanging your run to its timeout.
- A generic "Gen AI Compute Task" INTERNAL_ERROR means *your code exited non-zero OR a
  platform error* — they're indistinguishable (verified with a deliberate `exit 1`). With
  logs broken, use the params pattern above to see what actually happened.
