# Foundation-model training on AIR — experiment notes

## TabICL pretraining proof of life (customer marketing pilot, Q4)

Status 2026-07-17: research + scaffolding done, nothing launched yet.
Files: `tabicl/pol_stage1_smoke.sh`, `workloads/tabicl-pol.example.yaml`.
Context: customer technical summary doc (link in docs/private/customer-refs.md).

### What TabICL pretraining actually is (verified against soda-inria/tabicl@main)

- Entry point: `torchrun --standalone --nproc_per_node=N -m tabicl.train`; recipes in
  `scripts/train_v2_{clf,reg}_stage{1,2,3}.sh` (v2 paper: arxiv 2602.11139).
- Three-stage curriculum, batch 64 throughout: stage 1 = 500K steps @ 1,024 samples/dataset;
  stage 2 = 40K steps @ 400–10,240; stage 3 = 10K steps @ 400–60,000 (micro_batch 1).
  Paper used **4 GPUs** — this is not a big-iron pretrain.
- **No real data.** Priors are synthetic (`graph_scm`), generated **on the fly on CPU** in
  dataloader workers (`--prior_device cpu --n_jobs 16`). No dataset ingest for PoL; UC/Spark
  Connect only enters later if the customer fine-tunes on their tables.
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
| Fine-tune on real data | `tabicl/finetune_smoke.py` + `workloads/tabicl-finetune.example.yaml` (A10) | Zero-shot vs fine-tuned AUC on bank-marketing; verifies ckpt reload into zero-shot API. The realistic customer path (they won't pretrain from scratch). |

Notes: datasets via `sklearn.fetch_openml` (egress from AIR assumed OK per zerobus finding, open-q #7b —
verify openml.org specifically); switch to Delta/UC tables when the pilot wants customer-shaped data;
all three log to the auto-created MLflow run when available and degrade to stdout locally.

#### Sprawl bench — RESULTS (2026-07-17, e2-demo-field-eng, GPU_1xA10, run 949450981660633)

✅ SUCCESS end-to-end, **194s total** (env build 10s). Egress to openml.org AND HF hub works from
inside the workload (open-q #7b reinforced). MLflow run: experiment 4392293921068480,
run 8b80195159e24fb7bdbd903299923cb5.

| dataset | n_train | feats | TabICL AUC (s) | XGB default AUC (s) | XGB tuned AUC (s) |
|---|---|---|---|---|---|
| bank-marketing | 30,000 | 16 | **0.9420** (20.4s) | 0.9289 (0.1s) | 0.9336 (22.4s) |
| churn | 4,000 | 20 | **0.9270** (1.1s) | 0.9202 (0.1s) | 0.9188 (12.4s) |
| credit-g | 800 | 20 | 0.8446 (1.6s) | 0.8105 (0.1s) | **0.8538** (7.3s) |
| adult | 30,000 | 14 | 0.9254 (7.5s) | 0.9263 (0.1s) | **0.9269** (20.2s) |
| click-prediction | 30,000 | 9 | 0.6857 (6.5s) | 0.6758 (0.1s) | **0.6921** (23.5s) |

Read: zero-tuning TabICL beats tuned XGBoost on 2/5, ties 1, narrowly loses 2 — with no per-task
pipeline. GPU peak **10.4 GB at 30K rows × 16 features** on the 24GB A10 → memory-bound confirmed;
30K×100-feature tasks will need H100 (mem probe next). Node GPU-util metric read 0% throughout
(sampling artifact? forward passes are seconds long) — don't trust that gauge for short jobs.

#### Memory probe — RESULTS (2026-07-22, e2-demo-field-eng, 100 synthetic features, n_estimators=8)

H100 run 1022517780681952 (SUCCESS, 837s) · A10 run via `--override compute.accelerator_type`
(FAILED at 200K — see below). MLflow experiment 3794164435314621.

| rows | A10 24GB peak (wall) | H100 80GB peak (wall) |
|---|---|---|
| 1K | 4.1 GB (18s incl. ckpt dl) | 4.2 GB (12s) |
| 5K | 9.7 GB (3s) | 9.7 GB (2s) |
| 10K | 14.5 GB (6s) | 16.7 GB (2s) |
| 25K | 17.9 GB (13s) | 37.6 GB (5s) |
| 50K | 22.4 GB (28s) | 54.4 GB (10s) |
| 100K | 12.5 GB (114s) ← self-chunking kicks in | 64.6 GB (23s) |
| 200K | **KILLED — host-RAM OOM (exit 137)** | 41.8 GB (97s) ← self-chunking |
| 400K | — | 40.3 GB (225s) |
| 600K | — | **44.2 GB (422s) — no OOM** |

Findings (open-q #17 / B300):
- **Inference does NOT need B300.** TabICL v2 completes 600K×100 on one H100; above ~100K rows it
  self-manages memory (chunk/offload), trading wall time (~7 min at 600K) for footprint. It never
  hit torch CUDA OOM on either card.
- **The A10 death was HOST RAM, not GPU RAM**: exit 137 (OOM-killer SIGKILL, no traceback) at 200K
  rows — the 1xA10 node's CPU memory is the binding constraint for the offload path. Sharp edge for
  right-sizing guidance: "A10 to ~50K×100 in-GPU / ~100K chunked; beyond that, H100 or fail weird."
- A10 sweet spot confirmed: ≤50K rows × 100 feats fits in 24GB in-GPU; sub-30s.
- GPU-util gauge is real (pegged 99–100% on big rungs); 0% readings on short jobs are sampling misses.
- Remaining B300 question is now **pretraining-side only** (stage-3 60K-row context, FA3, float32) —
  PoL rung 4, not inference.
- `air run --override compute.accelerator_type=... mlflow_run_name=...` works as advertised.
- H100 capacity on e2-demo-field-eng: instantly scheduled (2026-07-22, ~00:25 UTC).

#### Multi-node probe — RESULTS (2026-07-22, e2-demo-field-eng, 2×8xH100, run 505819227973807)

✅ SUCCESS, 34s total node time (~40s submit→running for 2 nodes on-demand). Files:
`multinode/{probe_multinode.sh,allreduce_probe.py}` + `workloads/multinode-probe.example.yaml`.

- **Multinode is live**: silently PuPr'd 2026-07-17 (Ben Hansen, #research-on-air) — the field guide
  (updated May 17) and public docs still say Private Preview; a customer engineer hit that doc lag
  2026-07-22 (customer channel — see docs/private/customer-refs.md). Shapes: multiples of 8xH100 only; guide says max
  16 nodes / 128 GPUs, sweet spot 3–8 nodes, AWS-only.
- **RDMA path confirmed end-to-end**: aws-ofi-nccl 1.15.0 over EFA (`efa-direct`, 32 NICs/node),
  GPUDirect RDMA (`GDRDMA`) inter-node channels, NVLS/NVLink intra-node. NCCL 2.26.2.
- **Bandwidth**: 256MB all_reduce across 16 ranks / 2 nodes: 1.4 ms/iter, algbw 191 GB/s,
  busbw ~359 GB/s — near the p5's 3.2 Tbps (400 GB/s) EFA line rate.
- **torchrun just works** on the snapshot path via injected env (open-q #3 closed) — no Docker needed.
- Schema quirks: `environment.version` requires a `dependencies` list (use `[]`); torch preinstalled.
- CLI log stream shows node 0 only; `air logs <run> --node 1` for others.

This closes the "can we even do multi-node" question for the LLM ladder — rung 3 (16xH100 FSDP)
is unblocked; next missing piece is just `train_fsdp.py`.

#### Distributed correctness probe — pre-registered success criteria (written BEFORE submit, 2026-07-22)

File: `multinode/distributed_correctness_probe.py` (+ `workloads/multinode-correctness.example.yaml`).
Two assertion-gated proofs on 2×8xH100; the run succeeds iff rank 0 prints all three sentinels:

1. `PROOF1_MATMUL_EXACT_OK` — 16 ranks each compute a distinct formula-derived integer matmul
   shard (A_r: 4096×1024 @ B_r: 1024×4096, fp64-exact); all-reduced sum must (a) equal the
   **pre-registered checksum 3093** computed on a MacBook (torch 2.13.0 CPU) via the independent
   colsum·rowsum identity BEFORE the run, and (b) match rank 0's full single-GPU reference
   **bit-for-bit** (`torch.equal`). Per-rank partial sums are all distinct/nonzero
   (-1108, 4180, -5079, 6135, 3040, 3061, 6113, -5093, 4157, -1072, -3083, -2063, 2039, -8168,
   2047, -2013 → Σ=3093), so a missing/duplicated/wrong rank provably changes the total.
2. `PROOF2_GRAD_PARITY_OK` — per-rank MSE gradients on distinct data shards, all-reduce-averaged,
   must match rank 0's single-process gradient over the full global batch (8,192 rows) to <1e-9
   (fp64). Local CPU pre-flight measured 1.6e-14.
3. `DISTRIBUTED_CORRECTNESS_OK` — both proofs passed + barrier.

Local pre-flight (single-process CPU, this Mac): `LOCAL_VERIFY_OK checksum=3093 grad_diff=1.610e-14`.

**RESULTS — ✅ VERIFIED 2026-07-22, run 723000000990125, e2-demo-field-eng, 55s total.**
Raw evidence: job run 723000000990125 (retrieve: `air logs 723000000990125 [--node 1] -p
e2-demo-field-eng`; logs not committed by policy — see experiment-verification skill).
Line refs below are into the node-0 log. Claim→evidence:

| Claim | Evidence (log line) |
|---|---|
| 16 GPUs computed 16 distinct exact matmul shards; all-reduce produced the pre-registered off-cluster checksum AND matched a single-GPU reference bit-for-bit | L1470: `PROOF1_MATMUL_EXACT_OK checksum=3093 (pre-registered match, bit-exact vs reference, 9.7s)` — assertion-gated, unreachable on any mismatch |
| Data-parallel gradient averaging across 2 nodes = single-process gradient over the full 8,192-row batch | L1485: `PROOF2_GRAD_PARITY_OK max_abs_diff=6.106e-16 over 9352 params` (measured; fp64; tolerance was 1e-9) |
| Both proofs + barrier completed | L1496: `DISTRIBUTED_CORRECTNESS_OK`; L1504: `Job status: SUCCESS` |

Epistemic labels: checksum/bit-exactness = measured-exact (integer arithmetic, no tolerance);
grad diff 6.1e-16 = measured (order-of-summation noise at fp64 machine epsilon). The checksum was
computed on a MacBook via the colsum·rowsum identity BEFORE submission (see pre-registration above),
so the cluster could not have "learned" the answer from the code path that produced the reference.

**Process-evidence audit (self-applied, 2026-07-22).** The claim "this followed the verification
skill" was itself audited. Verifiable: pre-registration ordering is attested server-side — the
submit-time snapshot tarball (`.air/repo_snapshots/databricks-air-lab_20260722_210449.tar.gz`,
workspace `created_at` 1784768690197 = 01:04:50 UTC) contains `EXPECTED_CHECKSUM = 3093.0`
(line 35), and the run emitted the matching result at 01:06:37 UTC. Gaps found and addressed:
(1) evidence-ordering: the snapshot tarball's server-side timestamp is the pre-registration
mechanism (automatic on every submit; pre-run commits optional); (2) node-1 checked for both
runs (`air logs <id> --node 1`); (3) pre-flight re-executed 2026-07-22, output verbatim:
`LOCAL_VERIFY_OK checksum=3093 grad_diff=1.610e-14` (cmd: `python3 distributed_correctness_probe.py --local`);
(4) versions: air CLI v0.1.0; NCCL 2.26.2+cuda12.2 (run logs); runtime torch version NOT
captured — neither probe prints it, a gap to close in future probes; local pre-registration
used torch 2.13.0 CPU. Policy decision 2026-07-22: raw logs are NOT committed to the repo
(anonymization + evidence-strength rationale in the skill); platform holds raw evidence.

## FSDP training loop (BR-4 / suites #4 & #8) — `train_fsdp.py`

Files: `fsdp/train_fsdp.py` (trainer + `--local`) + `workloads/fsdp-multinode.example.yaml`.
Plan: `plans/train-fsdp.md`. Unblocks suite #4 (FSDP loop) + the checkpoint half of #8; feeds
open-q #10 (`max_retries` resume) and half-closes open-q #17 (memory envelope / B300).

**What this proves that the platform hasn't:** fabric + numerics are green at 160 GPUs, but only
with synthetic collectives (`nccl-allreduce`, `distributed_correctness_probe.py` — an all-reduce
parity check on a *replicated* model). FSDP2 reduces gradients with **reduce-scatter**, a different
collective on a sharded memory layout the probe never touches. Four properties are unproven, each
behind its own assertion-gated sentinel (unreachable unless assertions passed — "exited 0" is not
evidence).

#### Pre-registered success criteria (written BEFORE any GPU submit, 2026-07-28)

Runtime torch build is captured on rung 1 (`FSDP_VERSIONS torch=… nccl=… fully_shard=…`) — the
sharding assertion + DCP API are version-scoped; local pre-registration used **torch 2.13.0 CPU**.

1. `FSDP_SHARDING_OK world=W full=P local≈P/W …` — per-rank **parameter storage**
   `sum(p.to_local().numel())` ≈ `full/world` within a **3% padding tolerance** (last dim-0 shard
   padded to divide evenly). Asserted on persistent param storage, **not** peak memory (FSDP
   all-gathers full layers transiently → peak ≫ full/world; `max_memory_allocated` logged as
   observed only). **Wrapping strategy = per-block `fully_shard` + one top-level call** (pinned; it
   changes the exact ratio). **Vacuous at world=1** (smoke test only). Full open-q #17 envelope
   logged after the first optimizer step: `param + grad(post-reduce-scatter shard) + optim(m+v)`
   bytes vs the full-model counterfactual `(P+G+2P)·4B`, with **fp32 master params pinned**
   (MixedPrecisionPolicy keeps persistent storage at 4+4+8 B; a fully-bf16 model would not).
2. `FSDP_REDUCE_OK world=W grad_diff=… tol=…` — FSDP model + a single-process reference from an
   **identical init state_dict** (same seed, same process), **one backward, fp32, AMP off**, global
   batch split across ranks, mean-reduction loss. Gather each reduce-scattered grad with
   `p.grad.full_tensor()` (collective) and assert `max|g_full − g_ref| < REDUCE_TOL`. **Tolerance
   `2e-4`** — looser than the probe's fp64 `1e-9` (cross-rank reduce-scatter+all-gather is not
   bit-exact; float add-order differs). Compared **after the first backward, before any optimizer
   step**; the K-step endpoint comparison is **never done** (FSDP+AMP compounds per-step). Vacuous
   at world=1. The probe's all-reduce parity is **companion** evidence for the *replicated* path,
   not a substitute.
3. `FSDP_TRAIN_OK step0=… final=… drop=…` — K steps on the synthetic task; assert loss finite
   throughout AND final-window (20-step) mean below step-0 by ≥ `LOSS_DROP_MARGIN=0.05` **and**
   below `EXPECTED_LOSS_CEILING=4.10`. The ceiling is a **loose upper bound** (< `ln(64)=4.159`
   uniform-init step-0), not an invariant — it shifts with seed/LR/K/precision; bf16-on-H100 ≠
   fp32-on-CPU, so it transfers only as a generous bound. **Scope: shows the loop converges, NOT
   that reduction is correct** (a subtly-wrong reduction can still fall — that's Proof 2's job).
   **Triage rule:** if `FSDP_REDUCE_OK` is green, a Proof 3 miss is **tuning/precision** (re-triage
   K/LR/precision), *not* a platform fault — do **not** report "FSDP doesn't work on AIR."
4. `FSDP_CKPT_RESUME_OK fingerprint_match=True resumed_from=loss@N=…` — DCP sharded save/load to a
   UC volume, two-phase in one run: train N steps, save (**model + optimizer + scheduler**), record
   `loss@N` + a **fingerprint = sha256 of one fixed gathered param `blocks.0.mlp.0.weight` + its
   Adam m+v moments**; reconstruct fresh, load, assert fingerprint **bit-identical** AND next-step
   loss within `1e-4` of the pre-save trajectory. **Fingerprint uses the named `get_state_dict`
   mapping, not `opt.state[param]`** — FSDP2 reshards params into fresh DTensor objects across
   fwd/bwd, so an object-identity lookup silently misses the moments (found + fixed in pre-flight;
   would have made a dropped-moment regression invisible). DCP gated behind a **collective
   probe-write-first** (every rank writes a solo-timeout-guarded sentinel, all-reduce MIN the flag,
   branch collectively — never per-rank-abort a save mid-flight, which desyncs the PG → TIMEDOUT).
   Under a UC-volume 403 (BR-2): `FSDP_CKPT_PROBE_FAILED` → Proof 4 = `blocked-on-BR-2`, which does
   **not** fail BR-4.

**Completion lines (keep BR-4 separate from the checkpoint proof):**
- `FSDP_BR4_COMPLETE` — Proofs 1+2+3. **THE BR-4 acceptance receipt.**
- `FSDP_SUITE8_COMPLETE` — all four. `FSDP_SUITE8_BLOCKED` if Proof 4 is `blocked-on-BR-2` (BR-4
  receipt still stands).

**Synthetic task (pinned — no RNG in data, pure function of global step).** Next token =
`(x[t-1] + x[t-2]) mod 64` (Fibonacci-mod, `TASK_VOCAB=64 TASK_WINDOW=2 TASK_COEFFS=(1,1)
TASK_BIAS=0`); the first 2 tokens of row *r* at step *s* are seeded `(gid·131 + i·17 + 5) mod 64`
with `gid = s·1_000_003 + row_start + r`. Genuine context dependence (exercises attention/MLP, no
position→token shortcut) and, being a pure function of global step, **resume regenerates the
identical batch for step N with no sampler to checkpoint** (precondition for Proof 4). Model init
uses RNG (`INIT_SEED=1234`), so init on **CPU under the seed then `.to(device)`** for the training
model + the Proof-2 reference (CPU/CUDA RNG streams differ); rung-4 stress model is the exception
(meta/sharded init, to avoid host-OOM before sharding). Default shape: `L2 d256 h4 seq64 vocab64`.

**Local CPU/gloo pre-flight — ✅ RUN 2026-07-28 (this Mac, torch 2.13.0, `uv run --with torch
--with numpy`).** Exact command + verbatim output:
```
$ python3 train_fsdp.py --local --steps 300 --local-world 2 --proof4 --ckpt-steps 20
FSDP_VERSIONS torch=2.13.0 nccl=None cuda=None fully_shard=True world=2 device=cpu
FSDP_REDUCE_OK world=2 grad_diff=2.421e-08 tol=2.0e-04 global_batch=16
FSDP_SHARDING_OK world=2 full=1629248 local=814624 ratio=1.0000 mem=0.00GB state_local=0.0121GB state_full=0.0243GB [param=3258496 grad=3258496 optim=6516992 bytes]
[step 0] loss=4.3302
[step 299] loss=0.0441
FSDP_TRAIN_OK step0=4.3302 final=0.0467 drop=4.2835 ceiling=4.1 steps=300
FSDP_BR4_COMPLETE proofs=1,2,3 (sharding+reduce+convergence)
FSDP_CKPT_RESUME_OK fingerprint_match=True resumed_from=loss@20=3.867697 (post=3.867697)
FSDP_SUITE8_COMPLETE proofs=1,2,3,4 (adds checkpoint/resume)
```
**What `--local` actually exercised (the load-bearing rung-0 question):** FSDP2 `fully_shard`
**does run under CPU/gloo at world≥2** — verified at world=2 (ratio 1.0000) and world=4
(`local=407312 = full/4`, ratio 1.0000). So the CPU pre-flight de-risks real sharding,
reduce-scatter (grad_diff 2.4e-8 ≪ 2e-4), convergence (4.33→0.047; step-0 4.33 ≈ `ln(64)=4.159`
sanity anchor), and collective DCP save/resume (fingerprint bit-match, loss continuity 3.867697 →
3.867697). One macOS-only quirk: `fully_shard`'s device auto-detect trips on
`torch.mps.is_initialized`; passing an explicit `init_device_mesh("cpu", …)` dodges it (Linux/CUDA
unaffected). **Caveat:** this is CPU/gloo, not H100/NCCL — the fp32 tolerances, the DCP-on-real-
multi-GB-shard behavior, and peak-memory (rung 4) are still first-real on GPU. Rung 1's API gate is
**necessary but not sufficient** (world=1 → sharding/reduce vacuous).

**Open-q #17 envelope (from the local numbers, to reframe on GPU):** at world=2, `state_local ≈
state_full/2` (0.0121 ≈ 0.0243/2) — sharding roughly halves the persistent train-state footprint
(param+grad+optim = 4+4+8 = 16 B/param, of which params are only 1/4; measuring params alone
under-counts ~3×). This is *persistent* state; the customer's OOM/B300 question is governed by
**peak** memory (transient all-gathered layer + activations), which rung 4's DDP-OOM/FSDP-fit
counterfactual addresses — and even that only proves "sharding defers OOM for a state-dominated
model of size X," **not** the customer FM (egress-gated). #17 stays half-open.

**Still to pin before H100 spend:** `K` (300 reaches loss ≪ ceiling on CPU; confirm/adjust for the
bf16-on-H100 regime), rung-2 wall-clock from the A10 step time → `timeout_minutes` ≈ 2×, and the
**H100 spend approver** (TBD — name here before submitting rung 2/3). Raw logs not committed
(policy); platform + local MLflow archive hold evidence.

#### AIR CLI schema findings (v0.1.0, verified via --dry-run 2026-07-17)

- `environment.env_variables` **rejected** ("Unknown field"; only dependencies/docker_image/version).
  Docs lag. Workaround: inline env in `command` (`HF_HOME=/tmp/hf python …`). `secrets` unverified.
- `code_source.snapshot.root_path` resolves **relative to the YAML file's directory**, not CWD —
  YAMLs in `workloads/` need `root_path: ..`.
- `usage_policy_name` referencing a nonexistent policy fails validation (e2-demo-field-eng has no
  `air-lab` policy) — omit unless the workspace has one (open-q #5 still open).
- Submit = `air run -f <yaml> [-p profile] [--watch|--dry-run]`; `--dry-run` does full validation
  incl. workspace API calls — use it always.
