# Checkpoint past the 7-day cap

Goal: survive AIR's hard 7-day max workload runtime on long trainings.

!!! note "Epistemic status: doc-sourced"
    This page distills the product docs; unlike most of the cookbook it has **not** been
    re-verified by a lab run yet. The big open question is flagged below.

## The pattern

Write checkpoints to UC Volumes with `serverless_gpu.data.UCVolumeWriter/Reader` + Torch
Distributed Checkpoint (DCP):

- Writes stage through NVMe-backed `/tmp`, then upload — faster than writing through FUSE directly.
- `.metadata` is published only after all shards land, so a partial checkpoint is never mistaken
  for a complete one.
- `dcp.async_save` does background uploads (needs a `cpu:gloo,cuda:nccl` process group).
- Requires GPU env **v5+** / `serverless_gpu` 0.5.16+.

!!! warning "Your data pipeline's position is NOT checkpointed"
    Resume from an epoch boundary, or track sample offsets yourself.

Frameworks with built-in resume (HF Trainer, or e.g. TabICL's `--checkpoint_dir` auto-resume)
slot into this directly — point them at the volume.

## The open question: does `max_retries` resume or restart?

Unverified (open-q #10). Whether a retry after the cap picks up your latest checkpoint or starts
from scratch depends entirely on your entrypoint being resume-aware: make your `command`
idempotent — "load latest checkpoint if present, else start" — and it stops mattering.
The planned restart test: checkpoint to a UC volume, kill mid-run, `max_retries: 1`, observe.

If you run this test before we do: archive the log and update this page with the receipt.
