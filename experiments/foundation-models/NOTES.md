# Foundation-model training on AIR — experiment notes

## TabICL pretraining proof of life ([the customer] Marketing pilot, Q4)

Status 2026-07-17: research + scaffolding done, nothing launched yet.
Files: `tabicl/pol_stage1_smoke.sh`, `workloads/tabicl-pol.example.yaml`.
Context: [the customer] technical summary doc (13e09FKCfpHX6wVaKQOoOTEJkIO3zez4nM7uFeIBS19E).

### What TabICL pretraining actually is (verified against soda-inria/tabicl@main)

- Entry point: `torchrun --standalone --nproc_per_node=N -m tabicl.train`; recipes in
  `scripts/train_v2_{clf,reg}_stage{1,2,3}.sh` (v2 paper: arxiv 2602.11139).
- Three-stage curriculum, batch 64 throughout: stage 1 = 500K steps @ 1,024 samples/dataset;
  stage 2 = 40K steps @ 400–10,240; stage 3 = 10K steps @ 400–60,000 (micro_batch 1).
  Paper used **4 GPUs** — this is not a big-iron pretrain.
- **No real data.** Priors are synthetic (`graph_scm`), generated **on the fly on CPU** in
  dataloader workers (`--prior_device cpu --n_jobs 16`). No dataset ingest for PoL; UC/Spark
  Connect only enters later if [the customer] fine-tunes on their tables.
- Deps are mild: torch>=2.2, sklearn, einops, `[pretraining]` extra adds transformers/xgboost/wandb.
  No pinned CUDA extensions. FlashAttention-3 is **optional** (`--use_flash_attn3`), off in
  stage 1, on in stages 2–3 "when installed".
- Muon optimizer, float32 + AMP, checkpoint/resume built in (`--checkpoint_dir`, auto-resume;
  stages chain via `--checkpoint_path --only_load_model True`).

### Why we think this is tricky on AIR — ranked

1. **Upstream code maturity, not AIR.** README: v2 pretraining code was "vibe-migrated from the
   original (private) codebase" and "not yet tested end-to-end". Expect upstream bugs; attribute
   failures carefully before blaming AIR.
2. **CPU-bound prior generation on GPU-priced nodes.** Throughput is gated by CPU workers feeding
   synthetic batches. Sharp edge: AIR fixes the CPU:GPU ratio per accelerator type — on 1xA10/1xH100
   nodes the vCPU count may starve the GPU. Measure GPU util vs `--n_jobs` explicitly; this is the
   headline finding for "tricky to train on AIR".
3. **FlashAttention-3 availability** (stages 2–3). FA3 is Hopper-specific and built from source —
   check what the Databricks AI env actually ships (FA2 vs FA3?). Without it, stage 3's 60K-row
   attention at micro_batch 1, float32, `--recompute False` is the OOM candidate on 80GB H100 —
   this is exactly the memory probe that validates/kills the "needs B300" claim (open-q #17).
4. **500K-step stage 1 vs 7-day cap** → checkpoint to a UC volume + restart discipline (open-q #10).
   Upstream auto-resume should make this tractable — verify `max_retries` actually resumes.
5. **wandb is the assumed tracker** — disabled via `WANDB_MODE=disabled`; step metrics won't land in
   MLflow without a shim. Fine for PoL; needed for the real pilot.

### PoL ladder

- [ ] Rung 1 — `GPU_1xA10`, NPROC=1, 100 steps (the example YAML as-is): env resolves the git dep,
      training loop runs, checkpoints written. Also answers open-q #6-adjacent basics.
- [ ] Rung 2 — same on `GPU_1xH100` (stage-1 attention kernel sanity on Hopper; XGBoost-hang
      precedent says don't assume A10→H100 transfers, open-q #12).
- [ ] Rung 3 — `GPU_8xH100`, NPROC=8: torchrun single-node parity; GPU util vs n_jobs sweep.
- [ ] Rung 4 — stage-3 config short run on 8xH100: max_seq_len ramp 10K→60K, find the OOM point
      with/without FA3 and `--recompute True` → data for open-q #17 (B300).
- [ ] Restart test: CKPT_DIR on UC volume, kill mid-run, `max_retries: 1` — does it resume?

Launch (from repo root, live copy of the YAML minus `.example`):
`air workload submit workloads/tabicl-pol.yaml -p <profile>` — verify exact subcommand via `air -h`.

### Useful-while-testing tracks (2026-07-17)

Inference/fine-tune tracks that produce customer-legible artifacts while the PoL ladder runs.
TabICL API: sklearn-style `TabICLClassifier`/`TabICLRegressor` (checkpoints auto-download from
HF hub; `n_estimators`, `batch_size`, `offload_mode="auto"` are the memory levers) and
`FinetunedTabICLClassifier` (`pip install tabicl[finetune]`).

| Track | Files | Deliverable |
|---|---|---|
| Sprawl head-to-head | `tabicl/bench_sprawl.py` + `workloads/tabicl-bench.example.yaml` (A10) | One checkpoint vs per-task tuned XGBoost on 5 marketing-ish OpenML tasks (bank-marketing 1461, churn 40701, credit-g 31, adult 1590, click 1220): accuracy / time-to-model / GPU peak. The Q4 Marketing pilot slide. |
| Memory envelope | `tabicl/mem_probe.py` + `workloads/tabicl-memprobe.example.yaml` (H100; rerun on A10, ± `--offload`) | rows→GPU-GB curve to OOM at 100 features → concrete data for open-q #17 (B300). Upstream claim to check: 50K×100 fit+predict <10s on H100. |
| Fine-tune on real data | `tabicl/finetune_smoke.py` + `workloads/tabicl-finetune.example.yaml` (A10) | Zero-shot vs fine-tuned AUC on bank-marketing; verifies ckpt reload into zero-shot API. The realistic [the customer] path (they won't pretrain from scratch). |

Notes: datasets via `sklearn.fetch_openml` (egress from AIR assumed OK per zerobus finding, open-q #7b —
verify openml.org specifically); switch to Delta/UC tables when the pilot wants customer-shaped data;
all three log to the auto-created MLflow run when available and degrade to stdout locally.
