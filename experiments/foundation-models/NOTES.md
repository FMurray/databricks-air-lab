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

#### Training-side receipts — pre-registered (2026-08-05, before submit)

Review gap (correctly called): every TabICL receipt to date is inference/in-context scoring;
the fine-tune and pretrain assets (7/17) were never run, while the customer conversation
is about their pretraining scripts. Two A10 submits on e2 (egress needed for HF/pip):

1. **Fine-tune smoke** (`tabicl-finetune.example.yaml` as-is, 10 epochs, bank-marketing).
   Success = SUCCESS + prints zero-shot AUC then fine-tuned AUC on the same split +
   reloaded-checkpoint AUC ≡ fine-tuned (reload works). Expectation: fine-tuned ≥ zero-shot
   (0.9417 baseline); a *drop* is a finding (their quarterly path would inherit it), not a
   failure. Wall-clock is the "minutes-scale" claim check — currently an assumption in the
   deck; this run replaces it with a number.
2. **Pretrain PoL rung 1** (`tabicl-pol.example.yaml` as-is: stage-1, MAX_STEPS=100,
   NPROC=1). Success = env resolves the git dep, `tabicl.train` runs 100 steps, loss prints,
   checkpoint written. This is the "can their pretraining script even run on the platform"
   receipt — upstream self-describes v2 pretrain code as not tested end-to-end, so attribute
   failures carefully (upstream vs platform) per the 7/17 plan.

#### Training-side + 500-feature RESULTS (2026-08-05, e2-demo-field-eng, all archived)

**Fine-tune smoke ✅ run 327679047486914** (A10, 10 epochs, bank-marketing), verbatim:
`zero-shot AUC: 0.9412` → `fine-tuned AUC: 0.9429  (delta +0.0017, 90s, ckpt ->
/tmp/tabicl-finetune)`. The quarterly-refresh path is now measured: **90 s on an A10**,
fine-tune ≥ zero-shot, checkpoint written+reloaded. "Minutes-scale" is no longer an assumption.

**Pretrain PoL rung 1 — four attempts, final attribution: UPSTREAM BUG + undersized GPU;
platform exonerated.**
- Attempt 1 (939327917565018): env build fail — launcher **splits requirements on
  whitespace** (`tabicl[pretraining] @ git+…` → 3 reqs, bare `@` parse error). Platform bug
  #1, workaround: space-free PEP 508. Report to eng.
- Attempt 2 (529615913980817): deps "Successfully installed", then
  `ModuleNotFoundError: xgboost` — **extras dropped from git direct refs** during env
  resolution. Platform bug #2, workaround: pin extras explicitly.
- Attempt 3 (94698857812517): same for transformers (extras list completed).
- Attempt 4 (629012463931175): **env clean, trainer RAN — one full training step executed
  on AIR** (verbatim: `Step: 1%| 1/100 [01:08<1:53:18, 68.67s/it, accuracy=0.05, ce=2.16,
  prior_time=56.5, train_time=11.9]`), then `Warning: OOM error in micro-batch 16/16 at
  step 0. Skipping.` → `RuntimeError: Expected to have finished reduction in the prior
  iteration…` — **upstream's skip-on-OOM path breaks DDP's reduction contract** (skipped
  micro-batch ⇒ params unused in loss ⇒ crash). The 7/17 "code maturity, not AIR" ranking
  holds: platform runs the loop; upstream crashes it.
- **Bonus receipt — CPU-bound prior-gen hypothesis CONFIRMED (measured):** step 1 spent
  56.5s generating priors (CPU, N_JOBS=8) vs 11.9s training (GPU) — **83% of wall-clock is
  the CPU stage** on a 1xA10 node. This was ranked hypothesis #2 for "why tricky on AIR";
  it's now a number. Right-sizing implication: pretrain rungs need high-CPU shapes or more
  N_JOBS headroom, and GPU util gauges will read low regardless.
- Next rung when wanted: H100 (bigger micro-batches dodge the A10 OOM→skip path entirely),
  or upstream issue for the DDP bug.

**Memory probe at 500 features, H100 — run 635048884698037** (FAILED at the last rung;
ladder itself is the result). Measured:

| rows×500feat | peak GB | wall |
|---|---|---|
| 1K | 19.6 | 9.4s |
| 5K | 50.2 | 8.4s |
| 10K | 55.8 | 12.5s |
| 25K | **71.8** (allocator OOM-retries, completed) | 26.8s |
| 50K | 44.0 ← chunking kicks in | 89.4s |
| 100K | 44.1 | 159.5s |
| 200K | 44.5 | 485.4s |
| 400K | **KILLED exit 137 — HOST RAM** (GPU fine) | — |

Findings vs pre-registration: (1) **the customer shape 60K×500 FITS on one H100** —
bracketed by the 50K/100K rungs at ~44 GB chunked, ~2 min wall; naive-5× hypothesis was
wrong because chunking triggers on total elements, earlier at 500 feats. (2) Sharp edge:
**~25K×500 is the danger zone** — unchunked peak 71.8 GB of 80, with allocator retries;
slightly wider tables or fatter dtypes could tip it (offload/chunk-forcing knob is the
mitigation to document). (3) Ceiling on this node type is again **host RAM** (400K×500,
exit 137) — same constraint class as the A10 at 200K×100; the offload path binds on CPU
memory, not GPU. B300 answer at THEIR shape: still no — with a receipt this time.

#### Sprawl bench on H100 — pre-registered (2026-08-05, before submit)

Gap flagged in review: sprawl bench had only the A10 run. H100 variant = same YAML +
`--override compute.accelerator_type=GPU_1xH100`, e2-demo-field-eng (egress works there).
Success = SUCCESS + all 5 tasks complete; expectations: **AUC within noise of the A10 run**
(same model/checkpoint — accuracy is card-independent; a large delta means nondeterminism
worth knowing about), TabICL wall-times ≲ A10's (20.4s bank-marketing is the headline),
GPU peak ≤ A10's 10.4GB (same tensors, more headroom). Archive to local store after.

#### Memory probe at 500 features — pre-registered (2026-08-05, before submit)

The [customer-model]-shape gap: all prior memory data is 100-feature; the customer's tables are
500+ wide, 60K rows stated. Run: same `mem_probe.py`, `--features 500`, 1×H100,
e2-demo-field-eng (pip `tabicl` needs egress). Success = ladder runs until completion or
OOM (an OOM rung IS a finding, not a failure). Pre-registered expectations:
- The customer-relevant datum is **peak GB at the 50K–100K rungs** (brackets their 60K×500).
- Naive scaling guess: ~5× the 100-feat footprint at the same rows (54.4GB at 50K×100 →
  would NOT fit) — but chunking kicked in by 100K before; whether it triggers on width is
  exactly what we don't know. Label: hypothesis, not prediction.
- If 60K×500 fits under 80GB → the B300 answer stays "no" with their exact shape as receipt.
  If it OOMs → the recommendation shifts to offload mode (`--offload` follow-up) or feature
  chunking, still not B300, but with data instead of extrapolation.

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
Unblocks suite #4 (FSDP loop) + the checkpoint half of #8; feeds open-q #10 (`max_retries`
resume) and half-closes open-q #17 (memory envelope / B300).

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
#### Multinode probe on the reserved pool — pre-registered success criteria (written BEFORE submit, 2026-07-24)

Context: 20 dedicated H100 nodes now attached to fe-sandbox-mkazia-lw2 (UAT plan phase 1;
docs/private/uat-plan-2026-07.md). First distributed test against the pool:
`workloads/multinode-probe.yaml` as-is (2 nodes × 8xH100), CLI-only per the engagement rule.
2×A10 plumbing dry-run on this workspace already ✅ (run 128835177125736, 2026-07-24).

Probe change for this workspace's log blackout (docs/06-uat-suite.md root cause): stdout is
unretrievable here, so `allreduce_probe.py` now also writes a rank-0 MLflow receipt
(params: `probe_sentinel`, `world_size`, `nodes_seen` via all_gather_object across ranks,
torch/NCCL versions — closes the version-capture gap from 2026-07-22; metrics: allreduce
ms/iter, algbw, busbw), wrapped in `signal.alarm(120)` so a blocked tracking call can't hang
the run. Receipt is written after all asserts + barrier, so it is unreachable on failure.

Success = ALL of:
1. Run state SUCCESS (2×8xH100, timeout 30 min, max_retries 0).
2. MLflow receipt present: `probe_sentinel=MULTINODE_PROBE_OK`, `world_size=16`,
   `nodes_seen=0,1` (both node ranks checked in — not a single-node fallback).
3. Bandwidth metrics recorded. Expectation from e2-demo-field-eng baseline (run
   505819227973807): busbw in the hundreds of GB/s (~359 GB/s there). Smoke-grade number —
   health signal for the pool fabric, not a customer-deck benchmark.
4. Scheduling latency (UAT A3): submit→RUNNING wall time recorded from client-side timestamps.
   No pass bar — this sets customer expectations for the reserved pool.

Failure modes distinguishable even with no logs: INTERNAL_ERROR + no receipt = crash before
rank-0 receipt (indistinguishable user/platform per the exit-1 differential); SUCCESS + no
receipt = MLflow receipt path broken (probe still passed its asserts); receipt with
`nodes_seen=0` only = torchrun fell back to single node.

**RESULTS — ✅ VERIFIED 2026-07-24 (2026-07-25T02:31Z), run 968264353316767,
fe-sandbox-mkazia-lw2, 2×8xH100 reserved pool.** MLflow run 509fb5d84831447f96fd031c61f4c8f9
(experiment air-lab-multinode-probe; local archive run 1da64ccff9274b899e80d811924405cd,
submit log under client_logs/). Claim→evidence (receipt = MLflow params/metrics, quoted
verbatim; stdout unretrievable on this workspace as expected):

| Claim | Evidence |
|---|---|
| All 16 ranks across both nodes ran and passed asserts | params `probe_sentinel=MULTINODE_PROBE_OK`, `world_size=16`, `nodes_seen=0,1` (all_gather_object across ranks — receipt unreachable on any assert failure) |
| H100 pool hardware + stack | params `gpu_name=NVIDIA H100 80GB HBM3`, `torch_version=2.7.1+cu126`, `nccl_version=2.26.2` (version gap from 2026-07-22 closed) |
| Fabric healthy (smoke-grade) | metrics `allreduce_256mb_ms=1.52`, `algbw_gbps=177.0` (measured), `busbw_gbps=331.88` (derived: algbw·2(n−1)/n) — same order as e2 baseline 359 GB/s; smoke-grade, not a customer-deck number (nccl-tests for that) |
| A3 scheduling latency (reserved pool) | submit 02:30:54Z → accepted 02:31:01Z (7 s) → SUCCESS 02:32:54Z; execution_duration 112 s (Jobs API). ~2 min wall total for 2×8xH100 incl. env — measured, single sample |
| No post-success TIMEDOUT hang | run TERMINATED/SUCCESS at 02:32:54, not held to timeout — differs from the A10-plane log-shipping hangs (docs/06); single observation, not yet a finding |

Log delivery still broken as documented: zero log artifacts in the MLflow run (only the
receipt + automatic system metrics) — the receipt pattern carried all evidence.

#### AIR CLI schema findings (v0.1.0, verified via --dry-run 2026-07-17)

- `environment.env_variables` **rejected** ("Unknown field"; only dependencies/docker_image/version).
  Docs lag. Workaround: inline env in `command` (`HF_HOME=/tmp/hf python …`). `secrets` unverified.
- `code_source.snapshot.root_path` resolves **relative to the YAML file's directory**, not CWD —
  YAMLs in `workloads/` need `root_path: ..`.
- `usage_policy_name` referencing a nonexistent policy fails validation (e2-demo-field-eng has no
  `air-lab` policy) — omit unless the workspace has one (open-q #5 still open).
- Submit = `air run -f <yaml> [-p profile] [--watch|--dry-run]`; `--dry-run` does full validation
  incl. workspace API calls — use it always.
