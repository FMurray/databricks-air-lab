# Resume mid-epoch

Goal: on a large dataset, resume a multi-node PyTorch job at the **exact next sample** on every
GPU after a crash, preemption, or the 7-day cap, instead of restarting the epoch.

!!! note "Epistemic status: local pre-flight + SDK source; not yet AIR-verified"
    Built on the AIR SDK's own data checkpointing (read from source, `serverless_gpu` 0.5.21+ /
    `databricks-sdk-air` 0.1.0). Every rule below was exercised by running the **real SDK code**
    off-cluster (CPU/gloo, two simulated nodes, local dirs standing in for `/Volumes`) — receipts in
    `experiments/dataloading/mid-epoch-resume/NOTES.md`. The multi-node AIR run
    (`workloads/mid-epoch-resume.example.yaml`) has not been submitted yet; FUSE, NCCL, and the
    platform's retry/SIGTERM behaviour are therefore unverified here.

## What to use

Everything lives in `experiments/dataloading/mid-epoch-resume/`:

- **`mid_epoch_resume_walkthrough.py`** — Databricks notebook that demonstrates each nuance live
  against your volume. Start here.
- **`resumable_data.py`, `checkpointing.py`** — copy into your project.
- **`train.py`** — multi-node DDP/FSDP2 template using both; **`verify_ledger.py`** proves
  exactly-once delivery.

## The rules (each one broke something when skipped)

1. Use the SDK `DataLoader` over a **checkpointable wrapper with a row cursor** (`ResumableRows`).
   The SDK resumes per file. A wrapper that just delegates `state_dict()` to `UCVolumeDataset`
   **skips** the rest of the file it was in (the SDK docs say it replays it), and the docs'
   row-cursor example fails during each stream's first file (restore raises on worker ≥ 1;
   epoch ≥ 2 skips the epoch).
2. Put weights, optimizer, and **every rank's** `{epoch, loader state, RNG}` in **one** DCP save via
   `UCVolumeWriter`. Gather the per-rank states into one replicated entry first: DCP keeps one copy
   of a non-tensor value per key.
3. Rank 0 picks the checkpoint and broadcasts it. Decide *when* to checkpoint collectively.
4. Give `dcp.async_save` its own gloo process group, and wait for it before gathering the next
   state. On the default group its background collectives collided with DDP's.
5. Same nodes × GPUs × `num_workers` on every attempt; set `num_workers` explicitly; freeze the
   dataset (fingerprint it); have far more files than `world × num_workers` streams.
6. Budget in steps; let each rank keep its own epoch. A shared epoch barrier hangs or drops data.
7. SIGTERM: collective emergency save, and make DataLoader workers ignore SIGTERM (torchrun signals
   them too, and their death aborts the save). Pass `multiprocessing_context="fork"`.
8. Accept on "every sample exactly once per epoch", not a bit-identical loss curve: with
   `num_workers > 1` the order after a resume legitimately differs.
