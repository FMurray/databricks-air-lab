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

#### PoL rungs 2+3 — pre-registered (2026-08-05, before submit)

Same wrapper/trainer (tabicl==2.1.1 @ 46b9196), e2, MAX_STEPS=100, timeout 120 min.
- **Rung 2, 1×H100**: hypothesis — 80GB absorbs the micro-batch that OOM'd the A10, so the
  skip-on-OOM path never triggers and the run completes 100 steps + writes checkpoints.
  Success = SUCCESS + step 100 reached + `checkpoints written` listing non-empty. Key
  measurements: steps/s after warmup (step-1 prior_time includes cold prior-gen; the
  informative number is steady-state s/it), and prior_time:train_time ratio on this node's
  vCPU count (wrapper prints `cpus`). If it OOM-skips even at 80GB → upstream bug bites
  everywhere; report upstream with both tracebacks.
- **Rung 3, 8×H100 NPROC=8**: hypothesis — torchrun 8-rank DDP works single-node (fabric
  not involved; NVLink only). Success = SUCCESS + 100 steps + all 8 ranks alive at exit.
  Key measurement: does 8× the prior-gen demand (8 ranks × N_JOBS=8 workers on one node's
  CPUs) starve the GPUs — steady-state s/it vs rung 2's, and the prior:train ratio. This is
  THE dataloader-bound datapoint for the customer's "most efficient pretrain setup" question
  (their scripts assume 4×H100/node; nodes here are 8×).
Failure attribution rule stays: env/launcher = platform; in-trainer = upstream (their code,
their recipe, pinned commit).

**RESULTS — ✅ BOTH RUNGS PASS (2026-08-05).**
- Rung 2, 1×H100 (run 765994843027630, 16 vCPUs): **100/100 steps in 7:55**, steady-state
  ~4.75 s/it, ce 2.16→1.34, accuracy 0.05→0.464, `step-100.ckpt` (220MB) written. The A10
  micro-batch OOM never triggered at 80GB — upstream's skip-on-OOM bug is real but moot on
  right-sized GPUs.
- Rung 3, 8×H100 NPROC=8 (run 661368351202999, **192 vCPUs**): **100/100 in 2:13**,
  1.26–1.34 s/it, ce 1.27, ckpt written. 8-rank single-node DDP works as published;
  64 prior-gen workers on 192 vCPUs → no starvation. Wall speedup vs rung 2: **3.6×**
  (measured; not 8× — per-step batch semantics unchanged, so label as wall-clock ratio only).
- **CORRECTION to the 83%-CPU-bound finding: it is a COLD-START effect, not steady state.**
  Steady-state `prior_time=0` on both rungs — the CPU prior-generation workers pipeline
  ahead of the GPU once warm. The A10 step-1 measurement (56.5s prior / 68.7s step) stands
  as a warmup datum on an 8-vCPU-class node, but the sizing guidance flips: at H100-node
  vCPU counts (16 at 1×, 192 at 8×), **pretraining is GPU-bound at steady state**.
  Open residual: rung-2 steady state shows train_time 2.41 of ~4.75 s/it — the other ~2.3s
  is unattributed (optimizer/host sync?); label observed, not diagnosed.

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

The customer-shape gap: all prior memory data is 100-feature; the customer's tables are
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

#### Timeseries + regression program — pre-registered (2026-08-05, before submit)

Driven by the customer deep dive (notes in docs/private/): consolidation
targets are 85–90 reg+classification models; their training plan is fusion of internal
TabICLv2 weights + continued pretraining with custom timeseries priors (likely TimEE-style,
arXiv 2607.07500). Program: R1 regression bench, T1 continue-mechanics, T2 TS baseline,
T3 treatment (blocked on T1+T2).

- **R1** (`bench_regression.py`, 1×A10): TabICLRegressor vs XGB default/tuned on 5 OpenML
  regression tasks; metrics nRMSE (RMSE/std) + R². Success = ≥4/5 tasks complete. Expectation
  from the classification analogue: competitive-with-tuned-XGB; no pre-commitment on wins.
- **T1** (`continued_pretrain_smoke.sh`, 1×H100): phase A 60 steps from scratch → ckpt;
  phase B fresh trainer `--checkpoint_path <A> --only_load_model True` 60 more steps.
  PASS = phase-B first logged ce **well below cold start** (~2.3 = ln(10) for 10-class
  prior) — loaded weights vs silent re-init. Phase C: released HF inference ckpt into the
  trainer — loads or not, both are findings (the internal-weights format question (can published/externally produced checkpoints chain into the trainer)).
- **T2** (`bench_timeseries.py --finetune`, 1×A10): 5 UCR multi-class sets (5–24 classes,
  series len 46–140 = columns, inside the customer's 100–500 range), TabICL zero-shot +
  fine-tuned (100 epochs ≈ their oneshot recipe) vs XGB on identical tabular framing;
  accuracy + macro-F1. Success = ≥4/5 complete. This is the T3 baseline; no directional
  pre-commitment (TabICL's prior is not temporal — mid AUC vs XGB here is plausible and
  would MOTIVATE the customer's timeseries-prior plan rather than undermine it).
- **T3** (design, runs after T1+T2): synthetic regime-classification sequence generator →
  continued-pretrain N steps from base → re-run T2 eval (delta = improvement claim) AND
  re-run sprawl tasks (delta = catastrophic-forgetting check, the customer's explicit bar).
  Both deltas are receipts regardless of sign.

**R1 RESULTS — ✅ CLEAN SWEEP (run 1080666107616587, 1×A10, 2026-08-05): TabICL beats
TUNED XGBoost on all 5/5 regression tasks, both metrics.** Verbatim table in run log
(regression_bench.csv artifact):

| task | TabICL nRMSE (s) | XGB-tuned nRMSE (s) | TabICL R² / XGB R² |
|---|---|---|---|
| cpu-act | **0.1051** (5.0s) | 0.1275 (36.3s) | **0.9890** / 0.9838 |
| pol | **0.0843** (3.1s) | 0.1029 (28.6s) | **0.9929** / 0.9894 |
| elevators | **0.2637** (2.9s) | 0.3097 (15.8s) | **0.9305** / 0.9041 |
| house-sales | **0.2840** (4.2s) | 0.3120 (29.7s) | **0.9194** / 0.9026 |
| diamonds | **0.1271** (7.1s) | 0.1383 (24.7s) | **0.9838** / 0.9809 |

Stronger than the classification sprawl (2/5 wins there): zero-tuning TabICL regression
wins 5/5 vs per-task tuned XGB, 4–12× faster than the tuning loop, ≤7.3GB GPU peak (A10-
friendly). Directly relevant: most of the 85–90 consolidation targets are regression.
First T2 submit also surfaced a bench bug class worth keeping: job state was SUCCESS with
0/5 tasks completed (per-task try/except + exit 0) — bench scripts now raise SystemExit(1)
on zero completions (exit code derived from results).

**T1 RESULTS — ✅ ALL THREE PHASES (run 965728515658718, 1×H100, 2026-08-05):**
- Phase A: 60 steps from scratch, ce 2.30→1.38, `step-60.ckpt` written.
- Phase B: fresh trainer + `--checkpoint_path <A> --only_load_model True` → first-step
  **ce=1.44 vs cold-start 2.30** — weights genuinely loaded, training continues (1.44→1.39
  over the first 5 steps). The fusion→continue mechanics WORK on the platform.
- Phase C: **the released HF inference checkpoint loads into the trainer** (verbatim:
  `T1_PHASE_C_RELEASED_CKPT_LOADS=yes`) — the internal-weights-format question answered:
  published .ckpt chains into continued pretraining as-is.
- Driver nit: the in-script B_CE grep printed empty (progress-bar \r); continuity numbers
  extracted from the log directly. Fix with `tr '\r' '\n'` before T3 reuses the driver.
- R1/T2 first submits failed on missing xgboost (my YAML-generation bug — the sed dropped
  the [finetune] extra) + tabulate (to_markdown); both fixed (explicit xgboost dep,
  to_string), resubmitted as runs 1080666107616587 (R1) / 405178255580469 (T2).

**T2 partial (run 892979459471614, 2/5 — below the ≥4/5 bar; rerun in flight):**
- The 2 completed sets are STRONG for TabICL zero-shot on timeseries-as-tabular:
  ECG5000 acc 0.9464 / macro-F1 0.6466 vs XGB 0.9329/0.5471; MedicalImages acc 0.8408 /
  F1 0.8139 vs XGB 0.7118/0.6167. (Contradicts the pre-registered "mediocre plausible" —
  the tabular prior transfers to windowed TS better than expected, 2-task sample.)
- 3 failures are ALL in **upstream's fine-tune path**: CUDA scatter-gather
  `index out of bounds` inside `_finetune/classifier._compute_batch_loss` →
  `tabicl.py:496 _train_forward` (ElectricDevices, Crop, FaceAll). Zero-shot leg fine.
  Second upstream fine-tune-path defect this week (after the DDP skip-on-OOM) — worth one
  combined upstream report.
- ECG5000 fine-tune returned metrics BIT-IDENTICAL to zero-shot (early stopping reverting
  to initial weights at epochs=100/patience=8) — fine-tune didn't help there, observed.
- Bench bug fixed: a fine-tune crash was destroying the whole row incl. computed zero-shot
  numbers; fine-tune leg now isolated per-dataset (`finetuned_error` column).
  T2 rerun 806384006001663; T3 treatment submitted in parallel 908748703480331.

**T2 RESULTS — ✅ 5/5 zero-shot baseline (run 806384006001663, 1×A10, 2026-08-05):**

| set (classes) | TabICL zs acc / macro-F1 | XGB acc / macro-F1 | finetune leg |
|---|---|---|---|
| ECG5000 (5) | **0.9464 / 0.6466** | 0.9329 / 0.5471 | ran; bit-identical to zs (early-stop revert) |
| ElectricDevices (7) | **0.6907 / 0.6086** | 0.6759 / 0.5994 | upstream OutOfMemoryError |
| MedicalImages (10) | **0.8408 / 0.8139** | 0.7118 / 0.6167 | ran; +0.001 |
| Crop (24) | **0.8199 / 0.8201** | 0.7596 / 0.7596 | upstream OutOfMemoryError |
| FaceAll (14) | 0.7722 / **0.7980** | **0.7976** / 0.7759 | upstream AcceleratorError (scatter-gather assert) |

Zero-shot TabICL ≥ XGB on 4/5 accuracy and 5/5 macro-F1 — pre-registered "mediocre
plausible" hypothesis falsified; the tabular prior transfers to windowed TS well.
Upstream fine-tune path fails 3/5 UCR sets (typed above) — second fine-tune-path defect
class; combined upstream report justified.

**T3 first pass (run 908748703480331, 1×H100): treatment mechanics ✅, UCR delta ≈ 0.**
16/16 synthetic-TS tasks chained from the released ckpt in **300 s wall** (sentinel
`TS_CONTINUED_PRETRAIN_OK /tmp/ts-continued/task_015/best.ckpt`); treatment-ckpt UCR eval
5/5 vs T2 baseline: Δacc within ±0.004 on every set, mixed signs (ECG −0.0002, ElecDev
−0.0025, MedImg −0.0013, Crop +0.0019, FaceAll −0.0042). **Read: naive AR/seasonal regime
prior moves NOTHING on real UCR — the mechanics work end-to-end; the value hinges on
prior design (the customer's paper-based approach), not on infrastructure.** Forgetting
check lost to bench_sprawl's to_markdown crash (fixed → to_string); full rerun
344953280315973 (ckpt was container-local; chain re-executes — fine-tune stochasticity
not seed-pinned, so treatment ckpt is statistically-similar, not bit-identical; label
accordingly).

**T3 COMPLETE (rerun 344953280315973, 1×H100, 2026-08-05) — the three-number story:**
1. **Baseline** (T2): zero-shot TabICL ≥ XGB on 4/5 acc, 5/5 macro-F1 on UCR multi-class.
2. **Treatment TS delta ≈ 0**: 16 chained synthetic-TS fine-tunes (310 s wall), then UCR
   re-eval — Δacc within ±0.004, mixed signs, REPRODUCED across two independent chains
   (908748703480331, 344953280315973). The naive AR/seasonal regime prior adds nothing;
   value hinges on prior design (TimEE-class / customer's data-adapted priors), not infra.
3. **Forgetting delta ≈ 0 — the customer's acceptance bar HOLDS**: treatment-ckpt sprawl
   AUCs within ±0.0011 of the released-ckpt baseline on all 5 tabular tasks
   (bank-marketing −0.0008, churn +0.0011, credit-g +0.0002, adult 0.0000, click +0.0003).
   Continued training on 16 out-of-domain tasks did NOT degrade standard tabular
   performance under this protocol (default fine-tune LR 1e-5, early stopping).
Protocol caveat for customer reuse: pin fine-tune seeds for bit-reproducibility; deltas
here are noise-level, labeled measured; "no forgetting" is protocol-scoped (16 tasks ×
10 epochs, LR 1e-5) — their 2-round plan at larger scale should re-run this check, which
is now a 15-minute one-command receipt (`tabicl-ts-treatment.example.yaml`).

#### AIR CLI schema findings (v0.1.0, verified via --dry-run 2026-07-17)

- `environment.env_variables` **rejected** ("Unknown field"; only dependencies/docker_image/version).
  Docs lag. Workaround: inline env in `command` (`HF_HOME=/tmp/hf python …`). `secrets` unverified.
- `code_source.snapshot.root_path` resolves **relative to the YAML file's directory**, not CWD —
  YAMLs in `workloads/` need `root_path: ..`.
- `usage_policy_name` referencing a nonexistent policy fails validation (e2-demo-field-eng has no
  `air-lab` policy) — omit unless the workspace has one (open-q #5 still open).
- Submit = `air run -f <yaml> [-p profile] [--watch|--dry-run]`; `--dry-run` does full validation
  incl. workspace API calls — use it always.

### MLflow loggers verification (V1 finetune logger, V2 pretrain wandb-shim) — e2-demo-field-eng

Code under test: `tabicl/mlflow_loggers.py` (MLflowLogger base + MLflowFinetuningLogger),
`tabicl/wandb_mlflow_shim.py` + `tabicl/train_with_mlflow.py` (committed cc61617). Local
pre-flight: 19/19 checks vs a sqlite store (attach/resume/ownership, param chunking, shim
init/log/resume-by-id) — `uv run --with mlflow python test_mlflow_loggers.py` → `ALL PASS`
(scratchpad, 2026-08-05).

**Success criteria (written before submission):**

- **V1** (`tabicl-finetune.yaml`, finetune_smoke --epochs 10, 1×A10): ONE MLflow run
  (`tabicl-finetune-smoke`, the AIR ambient run) holds ALL of: per-step history
  `train/loss` + `train/lr`, per-epoch `val/roc_auc` + `train/mean_loss`, estimator
  params (epochs=10, eval_metric=roc_auc), AND final `auc_zero_shot`/`auc_finetuned`.
  The single-run property is the point — upstream fit() calls finish() on its logger,
  and the non-owning attach must survive it so the post-fit AUCs land in the same run.
  Stdout AUC lines are corroboration, not the evidence (log delivery unreliable).
- **V2** (`tabicl-pol.yaml`, pol_stage1_smoke MAX_STEPS=100, 1×A10): the AIR ambient
  MLflow run holds `ce`/`accuracy`/`lr` metric history reaching step 100 plus TrainConfig
  params (~80: lr=0.0008, prior_type=graph_scm, …) — logged by upstream tabicl.train's
  own wandb.init/wandb.log calls routed through the shim (`--wandb_log True`, real wandb
  never imported). Checkpoint-listing tail (`--- checkpoints written:`) still prints.
- **Fail** = metrics split across runs, missing histories, or a second stray run created.

**V1 RESULT — ✅ PASS (run ca733fbad91242b9abc35a2bd8d84e5d / job 728393411258922, 1×A10, e2, 2026-08-05):**

| claim | evidence |
|---|---|
| fit-internal metrics reach MLflow via MLflowFinetuningLogger | `train/loss` 40-pt history (steps 1–40), `train/lr`, per-epoch `val/roc_auc` 10 pts + `train/mean_loss` |
| post-fit AUCs land in the SAME run (ownership survives fit's finish()) | same run: `auc_zero_shot` 0.9412, `auc_finetuned` 0.9428; stdout corroborates: "zero-shot AUC: 0.9412" / "fine-tuned AUC: 0.9428 (delta +0.0015, 94s, ...)" |
| params captured | estimator+args params incl. epochs=10, eval_metric=roc_auc; no stray runs (±10 min window = 1 run) |

Archived → local run 3670034ea3964d8fade0a19d9bcb2ab3. Experiment description set (both tabicl-finetune and, pending V2, tabicl-pol).

**V2 attempt log (failures as they happened):**
- Attempt 1 (run 1056674589011590, A10): FAIL at Trainer init — upstream latent bug only
  reachable with `--wandb_log True`: `configure_wandb` writes `checkpoint_dir/wand_id.txt`
  BEFORE anything creates checkpoint_dir (`_run.py:168` FileNotFoundError). Shim's
  wandb.init had already succeeded. Fix: `mkdir -p "$CKPT_DIR"` in pol_stage1_smoke.sh.
- Attempt 2 (run 409480203318783, A10): step 0 completed and **shim delivered on-platform**
  — ambient MLflow run 32446ff95d4a444b976e1d50604fe632 holds ce=2.167, accuracy=0.076,
  lr=0.0008, prior_time/train_time + 112 TrainConfig params. Then FAIL at step 1: the KNOWN
  upstream DDP skip-on-OOM bug (micro-batch OOM on 24GB A10 → "Expected to have finished
  reduction") — same class as PoL attempt 4. A10 was never a passing rung for this recipe;
  example YAML annotated. Not a shim defect (measured metrics above predate the crash).
- Attempt 3 (run 348739312638178, **1×H100** — the proven rung): PASS, result below.

**V2 RESULT — ✅ PASS (run b9262046757a49be85a8fd0a2ad4f93c / job 348739312638178, 1×H100, e2, 2026-08-05):**

| claim | evidence |
|---|---|
| upstream tabicl.train metrics reach MLflow via wandb shim | `ce`/`accuracy`/`lr` each 100 pts, steps 1–100, in the AIR ambient run (no stray run; ±10 min window = 1) |
| values are upstream's own trainer output | ce 2.2995→1.3355, accuracy 0.061→0.4757, lr 8e-4→1.0e-7 — final lr equals the recipe's `--cosine_lr_end 1e-7` exactly (measured) |
| TrainConfig captured | 112 params (lr=0.0008, prior_type=graph_scm, max_steps=100, wandb_log=True) |
| training artifacts unaffected | stdout tail: step-50.ckpt + step-100.ckpt (220 MB each) + `wand_id.txt` (32 bytes = an MLflow run id → upstream resume file now resumes the MLflow run) |

Archived → local runs 7ffcfd51ead3465b8ff934e777493fa8 (pass) and d41208b00f4e47058a0cfecfbe4f0f21
(attempt-2 partial, failure receipt). Experiment descriptions set on both tabicl-finetune and
tabicl-pol (e2). Both MLflow-logger paths (protocol logger + wandb shim) are now verified
end-to-end on AIR; upstream-PR candidate: MLflowLogger + `--logger mlflow` flag (the finetune
Protocol refactor signals maintainer openness), bundled with the two upstream bugs found here
(wand_id.txt-before-mkdir; DDP skip-on-OOM).

### Pack-runner PoL — pre-registered (2026-08-06, before submit)

Question (sizing conversation): can one reserved 8xH100 node act as "8+ A10s" for batches
of small A10-shaped tasks (the CTAB oneshot-fine-tune profile, ~4GB GPU peak measured)?
Files: `tabicl/pack_runner.py` + `workloads/tabicl-pack.example.yaml` (GPU_8xH100, e2,
rounds=3 → 24 real tasks + 1 injected-bad, epochs=10 to match the measured A10 baseline).
Local stub pre-flight PASS (4 workers, claiming/isolation/sentinel): `PACK_RUNNER_OK
workers=4 distinct_gpus=4 tasks_ok=8/8 injected_failures_isolated=1`.

**Success = sentinel `PACK_RUNNER_OK` printed, which requires ALL of:**
- all 8 workers exit 0 on 8 DISTINCT `CUDA_VISIBLE_DEVICES` (one per GPU);
- 24/24 real tasks complete; the injected bogus-OpenML-id task is RECORDED as a task error
  (isolation) — not a worker/batch crash;
- solo-reference phase completes (same-node single-task wall for the slowdown denominator).

**Headline numbers to read out (hypotheses):** packed/solo median slowdown < 1.3x (192
vCPUs, prior expectation: little contention for GPU-bound fine-tunes); tasks/hour/node vs
the A10 sequential anchor (94s/task ⇒ ~38 tasks/hr; if slowdown holds, node ≈ 8× solo
rate ⇒ "one reserved node ≈ N A10s" with N measured, not asserted). MLflow ambient run
carries per-task walls/GPU peaks + pack_results.jsonl artifact; AIR system metrics should
show all 8 GPUs active during the pack phase (corroboration).
**Fail modes that are still results:** slowdown ≫ 1.3x (shared host bottleneck — measure,
don't guess which); HF/OpenML cache races under 8-way concurrency (would show as spurious
task errors ≠ the injected one).

**PACK-RUNNER RESULT — ✅ PASS (run 797789043315676 / mlflow 291a190590dd41b1b933571542f3862d, 8×H100, e2, 2026-08-06):**

| claim | evidence |
|---|---|
| 8-way single-GPU packing works on one AIR 8xH100 job | sentinel `PACK_RUNNER_OK workers=8 distinct_gpus=8 tasks_ok=24/24 injected_failures_isolated=1`; 24 tasks / 135s wall |
| per-task isolation holds | injected bogus-OpenML task recorded as `OpenMLError` row; all 24 real tasks completed |
| no meaningful packing contention | same-dataset packed walls: bank-marketing min/med/max 44.9/49.7/60.4s vs cold solo 58.7s (max packed = 1.03× cold solo; solo includes one-time HF download — cold, labeled) |
| node ≈ **15 A10s** on the CTAB oneshot profile | warm-vs-warm bank-marketing: packed med 49.7s vs measured A10 94s → 1.9×/GPU × 8 GPUs (derived from measured walls); mixed-workload throughput 638 tasks/hr/node (measured) |
| headroom for >1 task/GPU | peak task GPU mem 16.4 GB of 80 (measured) → co-location viable, untested (`--per-gpu` follow-up) |

Caveats, labeled: effective parallelism this run 6.5× (887 task-s/135.5s wall — tail effect
on a 3-round queue; approaches 8 with longer queues, inferred). Pre-registered
`packed_over_solo=0.61x` is confounded (mixed-dataset median vs largest-dataset cold solo)
— use the like-for-like rows above, not that headline metric. AIR gpu-util gauges read 0
throughout (known sampled-gauge trap on short jobs); parallelism evidence is the
distinct-CVD assertion + task-seconds/wall arithmetic, not the gauge.
Archived → local run fa3bbdd8bb9c49539152825935a1fa4b (pack_results.jsonl attached).
