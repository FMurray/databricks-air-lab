# Mid-epoch resume — multi-node PyTorch on AIR, at the exact next sample

**Ask:** a PyTorch example for resuming mid-epoch on AIR (the customer's dataset is large enough
that epoch-boundary checkpoints are too coarse), using the AIR / serverless-GPU SDK as much as
possible: an explanatory notebook plus scripts the customer can reuse. Multi-node; resuming on a
different accelerator count is out of scope.

| File | Role |
|---|---|
| `mid_epoch_resume_walkthrough.py` | Databricks notebook: each nuance demonstrated live (`PROBE … OK` lines), then a single-node crash/resume run of `train.py`, then the multi-node recipe |
| `resumable_data.py` | **customer-reusable**: SDK resolver, `ResumableRows`, per-rank state gather, `StepControl`, `dataset_fingerprint`, `ignore_sigterm_in_worker` |
| `checkpointing.py` | **customer-reusable**: DCP via `UCVolumeWriter`/`Reader`, async save on its own gloo group, rank-0-decides resume, retention |
| `train.py` | **customer template**: multi-node DDP (default) / FSDP2 (`--fsdp`) trainer; `[demo]` flags for staging, fault injection, ledger |
| `make_synthetic_dataset.py` | demo data (`.npy` with a global sample id column) + its seeking `ParseFn` |
| `verify_ledger.py` | exactly-once proof from the per-rank ledger |
| `sdk_behavior_probe.py` | assertion probe of the SDK's checkpoint semantics (incl. three docs defects) |
| `local_e2e.py`, `local_notebook_smoke.py`, `_local_fs.py` | off-cluster pre-flight harnesses (real SDK code; `_local_fs` lifts only its `/Volumes` prefix check) |
| `../../../workloads/mid-epoch-resume.example.yaml` | multi-node AIR run: forced crash + `max_retries: 1` + ledger verify |

Related fix: `experiments/foundation-models/fsdp/resilient_8xh100_notebook.py` proof 4 (below).

## Sources read (2026-11-09)

- **SDK source**, `~/universe` @ `e316707c384dc` (2026-10-21): `ai-compute/skyrun/serverless_gpu/data/`
  (dist `databricks_serverless_gpu`, `__version__` 0.5.26) and `ai-compute/databricks-sdk-air/databricks/air/data/`
  (dist `databricks-sdk-air` 0.1.0). The two `data` packages are **byte-identical modulo the
  package name** (diffed). Loader checkpointing landed in **0.5.21** (changelog; commit
  `dfb11c3a19582`, 2026-07-30). The v5 image was last observed carrying 0.5.22 (run 838281261554839,
  `experiments/dataloading/NOTES.md`) → *inferred* to have it; not yet observed on a run.
- **Docs:** [performance-and-resiliency](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency)
  describes manual offset tracking and does not mention the loader's `state_dict()`;
  [SDK API reference](https://api-docs.databricks.com/python/databricks-sdk-air/api/data.html)
  documents `databricks.air.data` `DataLoader/UCVolumeDataset.state_dict()`, "GPU environment
  version 6 and above". The docs disagree with each other; the source is authoritative.

## Findings

Labels: **[source]** read from SDK source · **[local]** observed in the off-cluster pre-flight
(real SDK code, CPU/gloo, local dirs) · **[AIR]** observed on a workspace (none yet).

1. **[source][local]** The SDK `DataLoader.state_dict()` records, per (rank, worker) stream, files
   **delivered to the loop** (prefetch-safe), and pins `(world_size, num_workers)`; a different
   grid raises. Granularity is the **file**.
2. **[local] SDK docs defect — delegating wrapper.** `serverless_gpu/data/CLAUDE.md` says a wrapper
   that delegates `state_dict()` to `UCVolumeDataset` gets file-granular resume where "the whole
   seam file … is replayed (at-least-once)". Observed: the seam file's remaining rows are
   **skipped** (at-most-once, silent loss) — the inner stream counts a file delivered when it yields
   the path, before the wrapper emits its rows.
3. **[local] SDK docs defect — row-cursor example, first file.** The example snapshots the inner
   state in `__init__` (main process, stamped `worker_id=0`): a checkpoint taken while worker ≥1 is
   in its first file fails to restore (`ValueError: Cannot restore a checkpoint into a different
   stream`).
4. **[local] SDK docs defect — row-cursor example, epoch ≥ 2.** With persistent workers the
   example's snapshot still holds the previous epoch's terminal state during the new epoch's first
   file; a checkpoint there resumes past the whole slice (52 of 60 rows lost in the probe).
5. **[local]** DCP stores a non-tensor value once per key: per-rank values under a shared key
   collapse to one rank's (`loaded={0: 'rank1', 1: 'rank1'}`). → gather all ranks' loader states
   into one replicated JSON string.
6. **[local]** `dcp.async_save`'s background collectives on the default group collided with DDP's
   gloo all-reduce → `gloo::EnforceNotMet … Is there a distributed collective mismatch in your
   code?` on the first post-checkpoint step. Fixed by a dedicated gloo group for checkpoint
   collectives + waiting before the next gather. (On GPU the all-reduce is NCCL, so this exact
   collision is CPU-specific; the dedicated group is still the safe pattern for any CPU collective.)
7. **[local]** torchrun forwards SIGTERM to each rank's process group, killing DataLoader workers;
   torch then raised `DataLoader worker … killed by signal: Terminated` *inside the emergency
   save* and the resume fell back to the previous periodic checkpoint. Fixed by
   `ignore_sigterm_in_worker` as the `worker_init_fn`.
8. **[local]** The SDK loader requires the `fork` start method and raises otherwise; under
   `mp.spawn` children the default is `spawn` (hit by the notebook's `--local` path). Pass
   `multiprocessing_context="fork"` (also future-proofs Python ≥ 3.14, default `forkserver`).
9. **[local]** With `num_workers > 1` the post-resume **order** differs from an uninterrupted run
   (round-robin restarts at worker 0) while the **set** is exactly-once. Acceptance criterion:
   exactly-once per epoch, not a bit-identical curve.
10. **[source]** `UCVolumeDataset` never reorders files (sorted listing, fixed stride). Shuffle
    when writing; `ResumableRows(shuffle_seed)` adds a resumable within-file permutation.
11. **[design]** Step budget + per-rank epochs (ranks' streams differ in length, so a shared epoch
    boundary hangs or drops data); rank 0 decides the checkpoint; collective checkpoint cadence;
    dataset fingerprint in the checkpoint and compared across ranks; FUSE write→list visibility gate
    when staging (lag observed in run 455033116690072).

## Fix to `resilient_8xh100_notebook.py` (proof 4 was vacuous)

Before: saved `(epoch, sample_offset)` + shuffle RNG in a shared `app` DCP entry; on resume rebuilt
the loader from the epoch start without skipping (re-trained the epoch's head); the shuffle order
was computed and discarded; `RESILIENT_PIPELINE_OK` compared the first batch of two fresh loaders.
After: SDK `DataLoader.state_dict()` + embedded `ResumableRows` row cursor, per-rank gather in the
same DCP save, dedicated checkpoint PG, per-rank step-budget loop; proof 4a in-script
(`RESILIENT_PIPELINE_STATE_OK`: restored fresh loader replays nothing) and 4b in a new verify cell
(`RESILIENT_PIPELINE_OK`: ledger exactly-once across the relaunch). `--local` now runs the real SDK
(and wraps DDP — without it the CPU ranks diverged and DCP's replicated-tensor dedup made the resume
fingerprint mismatch on rank 1, observed). The `@distributed` sibling has the same defect; it got a
⚠️ banner, not a port (launch path out of scope for the engagement).

## Local pre-flight receipts — ✅ 2026-11-09, off-cluster (NOT an AIR run)

Environment: CPU/gloo on a dev box; SDK = `databricks.air.data` imported from `~/universe` @
`e316707c384dc` (`AIR_LAB_LOCAL_FS=1` lifts only the `/Volumes` prefix check); torch 2.15.0, Python
3.12.11. "2 nodes" = two torchrun agents with `--nnodes=2` on localhost (2 procs each, world=4).
All nine steps below ran in one pass against the final code (raw logs are local working copies,
not committed). Command per step:

```
export PYTHONPATH=~/universe/ai-compute/databricks-sdk-air
U="uv run --no-project --with torch --with mlflow-skinny --with numpy --with pyarrow python"
AIR_LAB_LOCAL_FS=1 $U sdk_behavior_probe.py --root /tmp/mer-probe
$U local_e2e.py --root /tmp/mer-e2e-base            # also: --fsdp | --shuffle-seed 7 | --sigterm
AIR_LAB_LOCAL_FS=1 $U local_notebook_smoke.py --scratch /tmp/mer-nb
$U /tmp/resilient_train.py --local --local-world 2 --steps 40 --save-every 15 --dim 64 --layers 2 \
   --heads 2 --seq 32 --shards 8 --rows-per-shard 64   # then again with --steps 80, then the verify cell
```

| Claim | Evidence (quoted) |
|---|---|
| SDK semantics + 3 docs defects (findings 1–5, 9, 10) | `PROBE_SUMMARY 11/11 checks matched expectations` · `SDK_BEHAVIOR_PROBE_OK` |
| delegating wrapper loses data | `PROBE delegating_wrapper OK … dupes=0 missing=12 (… the seam file's tail is SKIPPED: at-most-once)` |
| docs example, first file | `PROBE doc_wrapper_first_file OK … ValueError: Cannot restore a checkpoint into a different stream: saved (rank=0, worker_id=0) but current (rank=0, worker_id=1)` |
| docs example, epoch 2 | `PROBE doc_wrapper_epoch2 OK … epoch2_rows_delivered=8 of 60 missing=52 dupes=0` |
| `ResumableRows` fixes both | `PROBE rows_first_file OK … dupes=0 missing=0` · `PROBE rows_epoch2 OK … dupes=0 missing=0` |
| DCP non-tensor dedup | `PROBE dcp_nontensor_dedup OK saved rank0->'rank0', rank1->'rank1' under one key; loaded={0: 'rank1', 1: 'rank1'}` |
| order differs, set exact (nw=3) | `PROBE order_after_resume OK exactly_once=True(dupes=0,missing=0) identical_order=False` |
| 2-node crash mid-epoch → resume, DDP | `FORCED_FAILURE rank=0 step=60 (last checkpoint step=50)` → `RESUMED_FROM_STEP=50 … rank0_epoch=1` → `LEDGER exact_cover OK dataset_rows=1024 complete_epochs={0: 1024, 1: 1024}` · `LEDGER_EXACTLY_ONCE_OK ranks=4 steps=80 attempts=2 complete_epochs=2` |
| same, FSDP2 | `ENV … strategy=fsdp2` · `RESUMED_FROM_STEP=50` · `LEDGER_EXACTLY_ONCE_OK ranks=4 steps=80 attempts=2 complete_epochs=2` |
| same, within-file shuffle | `RESUMED_FROM_STEP=50` · `LEDGER_EXACTLY_ONCE_OK ranks=4 steps=80 attempts=2 complete_epochs=2` |
| SIGTERM → collective emergency save → resume there | `SIGTERM_CHECKPOINTED step=93; exiting 143` → `RESUMED_FROM_STEP=93` · `SIGTERM_RESUME OK emergency_ckpt_step=93 resumed_from=93` · `LEDGER_EXACTLY_ONCE_OK ranks=4 steps=400 attempts=2 complete_epochs=12` |
| walkthrough notebook runs top to bottom | `NOTEBOOK_SMOKE_OK all code cells ran` (incl. `RESUMED_FROM_STEP=100` · `LEDGER_EXACTLY_ONCE_OK ranks=1 steps=300 attempts=2 complete_epochs=4`) |
| fixed resilient notebook, 2 launches | run 1 `RESILIENT_SUITE_COMPLETE proofs=1,2,3,4a … steps=0->40`; run 2 `RESUMED_FROM_STEP=40 rank0_epoch=1` … `RESILIENT_SUITE_COMPLETE … steps=40->80`; verify cell `RESILIENT_PIPELINE_OK exactly-once across the relaunch` (`complete_epochs=[0, 1] dupes=0 overlaps=0`) |
| verifier catches a replay (negative control) | 8 ids of an already-trained batch injected into the relaunch ledger → `LEDGER no_dupes: FAIL` · `LEDGER exact_cover: FAIL`, no `RESILIENT_PIPELINE_OK` |
| refuses unsafe resumes | world 4→2: `ValueError: checkpoint has data state for 4 ranks but world_size=2`; file added: `dataset changed since the checkpoint (fingerprint d62c2245216622d6 -> dd48faf4f73771ab)` |

Failures hit on the way (each fixed and now covered above): async-save/DDP gloo collective
mismatch (finding 6); SIGTERM killed DataLoader workers mid-save, resume fell back to step 75
(finding 7); SDK loader rejected `spawn` under `mp.spawn` (finding 8); resilient `--local` without
DDP → `RESILIENT_RESUME_FAIL fp_match=False` (fixed by wrapping DDP).


## AIR runs — pre-registered success criteria (written before submit; none submitted yet)

Blocked locally: no `air` CLI and no lab workspace profile on this machine.

1. **API gate, `GPU_1xA10` (single node — NOT a multi-node receipt):**
   `--override compute.num_accelerators=1 compute.accelerator_type=GPU_1xA10`, same YAML.
   Pass = attempt 0 prints `ENV … sdk=serverless_gpu.data` (v5) and `FORCED_FAILURE … step=700
   (last checkpoint step=600)`; attempt 1 prints `RESUMED_FROM_STEP=600`; node 0 prints
   `LEDGER_EXACTLY_ONCE_OK … attempts=2`. Also answers: is the loader checkpointing present on the
   v5 image (finding 1 → [AIR]); does `max_retries` resubmit run the identical command (open-q #10).
2. **Headline, 2× `GPU_8xH100`** (needs H100 approval): the YAML as committed. Pass = the same
   three sentinels, plus node 1's log (`air logs <id> --node 1`) showing its ranks' `FORCED_FAILURE`
   in attempt 0 and no errors in attempt 1; `LEDGER … ranks=16`.
3. **Walkthrough notebook** on a serverless GPU notebook (v5 and v6): every `PROBE … OK` and the
   section-7 asserts pass; on v6 the setup cell must report `databricks.air.data`.
4. **`resilient_8xh100_notebook`** on `GPU_8xH100`: `RESILIENT_SUITE_COMPLETE` in both launches
   and `RESILIENT_PIPELINE_OK` in the verify cell.

## Still open

- Does AIR send SIGTERM before stopping/preempting a workload, and with what grace? (The emergency
  save is best-effort until this is measured.)
- `databricks.air.data` on env v6: interpreter layout on the CLI path (is the PYTHONPATH bridge
  still needed?), and the `version: "databricks_ai_v6"` string seen in universe sentinel probes.
- File-level docs defects (findings 2–4) should go to the SDK owners (#ai-runtime-oncall) — not
  filed yet.
