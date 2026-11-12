"""Multi-node PyTorch training on Databricks AIR that resumes mid-epoch, at the exact next row.

Launch on every node with torchrun (see workloads/mid-epoch-resume.example.yaml):

    torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \\
      --master_port=$MASTER_PORT --nproc_per_node=$LOCAL_WORLD_SIZE --max-restarts=0 \\
      train.py --data-dir /Volumes/<cat>/<schema>/<vol>/data \\
               --ckpt-dir /Volumes/<cat>/<schema>/<vol>/ckpt --steps 2000

The script is idempotent: every attempt (first run, a ``max_retries`` resubmit, a manual rerun)
starts by looking for the newest complete checkpoint in ``--ckpt-dir`` and continues from it --
model, optimizer, RNG, and every rank's position inside its data stream. That is the whole
resume story; there is no "resume" flag to remember.

AIR SDK pieces used (``databricks.air.data`` / ``serverless_gpu.data``):
  UCVolumeDataset   file streams from the UC volume, split across ranks x workers, cached on NVMe
  DataLoader        prefetching loader whose state_dict() tracks *delivered* (not prefetched) data
  UCVolumeWriter    DCP storage backend: stages on local NVMe, uploads to the volume, .metadata last
  UCVolumeReader    the matching reader for resume
Added on top (resumable_data.py / checkpointing.py): row-level resume inside large files, every
rank's loader state folded into one checkpoint entry, lockstep checkpoint/SIGTERM decisions,
rank-0-decides resume discovery, retention.

Training is budgeted in **steps**, and each rank cycles its own loader. With per-(rank, worker)
file slices, ranks do not finish an epoch at the same step; a shared epoch boundary would either
hang (one rank stops calling collectives) or drop data. Each rank tracks its own epoch instead.

Demo/verification-only flags (omit in your own runs): ``--make-data`` stages a synthetic dataset,
``--fail-at-step`` hard-kills a cold attempt to exercise resume, ``--ledger-dir`` records which
sample ids each rank trained on (verify_ledger.py proves exactly-once from it).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from checkpointing import Checkpointer  # noqa: E402
from make_synthetic_dataset import VOCAB, parse_npy_rows, write_dataset  # noqa: E402
from resumable_data import (ResumableRows, StepControl, gather_data_state,  # noqa: E402
                            dataset_fingerprint, ignore_sigterm_in_worker, import_air_data,
                            my_data_state, restore_loader)


# --------------------------------------------------------------------------------------------
# Model: a small causal transformer LM (stand-in for yours)
# --------------------------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, seq: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True,
                                       norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.norm(x))


def build_model(args, device):
    model = TinyLM(VOCAB, args.dim, args.layers, args.heads, args.seq).to(device)
    if args.fsdp:
        from torch.distributed.fsdp import fully_shard
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        return model
    if dist.get_world_size() > 1:
        ids = [device.index] if device.type == "cuda" else None
        return nn.parallel.DistributedDataParallel(model, device_ids=ids)
    return model


# --------------------------------------------------------------------------------------------
# RNG state (rides in each rank's data-state entry; matters as soon as you add dropout/augment)
# --------------------------------------------------------------------------------------------
def _b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.numpy().tobytes()).decode()


def rng_state() -> dict:
    out = {"cpu": _b64(torch.get_rng_state())}
    if torch.cuda.is_available():
        out["cuda"] = _b64(torch.cuda.get_rng_state())
    return out


def set_rng_state(s: dict) -> None:
    def un(b):
        return torch.frombuffer(bytearray(base64.b64decode(b)), dtype=torch.uint8)
    torch.set_rng_state(un(s["cpu"]))
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state(un(s["cuda"]))


# --------------------------------------------------------------------------------------------
# Demo-only: per-rank ledger of trained sample ids, flushed with each checkpoint
# --------------------------------------------------------------------------------------------
class Ledger:
    """Records (step, epoch, ids) per trained batch. Flushed only when a checkpoint is saved, so
    the ledger on disk is exactly the trajectory a resume continues from (steps after the last
    checkpoint are lost work -- they are retrained after resume and must not count twice)."""

    def __init__(self, root, attempt, rank):
        self.dir = os.path.join(root, f"attempt_{attempt}") if root else None
        self.rank, self.buf = rank, []
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def record(self, step, epoch, ids):
        if self.dir:
            self.buf.append({"step": step, "epoch": epoch, "ids": ids})

    def flush(self, upto_step):
        if not self.dir or not self.buf:
            return
        path = os.path.join(self.dir, f"rank_{self.rank:05d}_upto_{upto_step:09d}.jsonl")
        with open(path, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in self.buf))
        self.buf = []


def _log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def _mlflow_logger(rank):
    """Rank-0 metric logger into the run AIR created for this workload (MLFLOW_RUN_ID), if any."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if rank != 0 or not run_id:
        return lambda *a, **k: None
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
    except Exception:  # noqa: BLE001 - metrics are optional
        return lambda *a, **k: None

    def log(key, value, step):
        try:
            client.log_metric(run_id, key, value, step=step)
        except Exception:  # noqa: BLE001
            pass
    return log


# --------------------------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="UC volume dir of data files")
    p.add_argument("--ckpt-dir", required=True, help="UC volume dir for checkpoints (never /tmp)")
    p.add_argument("--steps", type=int, default=2000, help="total optimizer steps (all attempts)")
    p.add_argument("--batch-size", type=int, default=16, help="rows per rank per step")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers per rank. Pinned into the checkpoint: keep it fixed "
                        "across attempts, and the same on every rank")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="shuffle rows within each file per epoch (deterministic, resumable)")
    p.add_argument("--ckpt-every-steps", type=int, default=200)
    p.add_argument("--ckpt-every-minutes", type=float, default=0.0,
                   help="also checkpoint on wall-clock cadence (agreed across ranks)")
    p.add_argument("--keep-last", type=int, default=3, help="complete checkpoints to retain")
    p.add_argument("--fsdp", action="store_true", help="FSDP2 fully_shard instead of DDP")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    # demo / verification only
    p.add_argument("--make-data", type=int, default=0, metavar="FILES",
                   help="[demo] rank 0 stages FILES synthetic files into --data-dir if missing")
    p.add_argument("--rows-per-file", type=int, default=512, help="[demo] with --make-data")
    p.add_argument("--fail-at-step", type=int, default=0,
                   help="[demo] every rank hard-exits 137 at this step, on a cold-start attempt only")
    p.add_argument("--ledger-dir", default=None, help="[demo] record trained sample ids per rank")
    args = p.parse_args()

    # ---- process group: CPU backend alongside NCCL is required by dcp.async_save ---------------
    cuda = torch.cuda.is_available()
    dist.init_process_group("cpu:gloo,cuda:nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    air, air_path = import_air_data()
    _log(rank, f"ENV world={world} nodes={os.environ.get('NUM_NODES', '?')} sdk={air_path} "
               f"torch={torch.__version__} device={device.type} strategy={'fsdp2' if args.fsdp else 'ddp'}")

    # ---- data ------------------------------------------------------------------------------------
    if args.make_data and rank == 0:
        n = write_dataset(args.data_dir, args.make_data, args.rows_per_file, args.seq)
        _log(rank, f"DATASET_READY files={args.make_data} newly_written={n}")
    dist.barrier()
    if args.make_data:
        # UC FUSE has a write->list lag (a listing right after a write saw 0 files: run
        # 455033116690072, experiments/dataloading/NOTES.md). UCVolumeDataset lists once, so wait
        # until this rank sees every staged file before building it.
        deadline = time.monotonic() + 300
        while (sum(1 for e in os.scandir(args.data_dir) if e.name.endswith(".npy"))
               < args.make_data):
            if time.monotonic() > deadline:
                raise SystemExit(f"staged {args.make_data} files but rank {rank} still sees fewer")
            time.sleep(2)
        dist.barrier()
    files = air.UCVolumeDataset(args.data_dir)
    # Sizing check. UCVolumeDataset deals the sorted file list round-robin over world x workers
    # streams, so each stream gets ~n_files/streams files; an empty stream makes its rank spin.
    n_files = sum(1 for e in os.scandir(args.data_dir) if e.is_file(follow_symlinks=False))
    streams = world * max(1, args.num_workers)
    per_stream = [len(range(i, n_files, streams)) for i in range(streams)]
    _log(rank, f"DATA files={n_files} streams={streams} (world {world} x workers "
               f"{max(1, args.num_workers)}) files_per_stream min={min(per_stream)} max={max(per_stream)}")
    if min(per_stream) == 0:
        raise SystemExit(f"{n_files} files cannot feed {streams} streams: some (rank, worker) streams "
                         "would be empty. Use more (smaller) files or fewer --num-workers.")
    fingerprint = dataset_fingerprint(args.data_dir)
    # Every rank must see the same listing, or streams overlap/miss files (e.g. a node listing a
    # directory over FUSE before another node's fresh writes are visible).
    seen = [None] * world
    dist.all_gather_object(seen, fingerprint)
    if len(set(seen)) != 1:
        raise SystemExit(f"ranks see different file listings under {args.data_dir}: {set(seen)}")
    rows_ds = ResumableRows(files, parse_npy_rows, shuffle_seed=args.shuffle_seed)
    # multiprocessing_context="fork": the SDK loader requires fork (its workers share one local
    # cache dir); explicit because the default is 'spawn' under mp.spawn and 'forkserver' on
    # Python >= 3.14.
    loader = air.DataLoader(rows_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            worker_init_fn=ignore_sigterm_in_worker,
                            multiprocessing_context="fork" if args.num_workers > 0 else None)

    # ---- model / optimizer / checkpointer -----------------------------------------------------------
    model = build_model(args, device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ckpt = Checkpointer(args.ckpt_dir, air, keep_last=args.keep_last)

    # ---- resume: newest complete checkpoint, chosen by rank 0 ---------------------------------------
    step, epoch = 0, 0
    latest = ckpt.latest_valid()
    cold = latest is None
    if latest is not None:
        step, data_json = ckpt.load(latest[1], model, optim)
        mine = my_data_state(data_json)
        if mine["dataset"] != fingerprint:
            raise SystemExit(f"dataset changed since the checkpoint (fingerprint {mine['dataset']} "
                             f"-> {fingerprint}): loader offsets would point at different data. "
                             "Restore the original files or start a new --ckpt-dir.")
        epoch = restore_loader(loader, rows_ds, data_json)
        set_rng_state(mine["rng"])
        _log(rank, f"RESUMED_FROM_STEP={step} ckpt={latest[1]} rank0_epoch={epoch}")
    else:
        _log(rank, "COLD_START")

    attempt = [time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"] if rank == 0 else [None]
    dist.broadcast_object_list(attempt, src=0)
    ledger = Ledger(args.ledger_dir, attempt[0], rank)
    control = StepControl(device, args.ckpt_every_steps, args.ckpt_every_minutes * 60)
    log_metric = _mlflow_logger(rank)

    def save(blocking=False):
        ckpt.wait()   # previous upload's background collectives must finish before we gather
        data_json = gather_data_state(loader, epoch, extra={"rng": rng_state(), "dataset": fingerprint})
        path = ckpt.save(step, model, optim, data_json, blocking=blocking)
        ledger.flush(step)
        control.mark_saved()
        _log(rank, f"CKPT_SAVED step={step} path={path} async={not blocking}")

    # ---- train ----------------------------------------------------------------------------------------
    model.train()
    it = iter(loader)
    t0 = time.monotonic()
    last_saved = step
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:              # this rank finished an epoch of its streams
            epoch += 1
            it = iter(loader)
            batch = next(it)
        tokens = batch["tokens"].to(device, non_blocking=True)
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step += 1
        ledger.record(step, epoch, [int(i) for i in batch["id"]])

        if step % args.log_every == 0:
            lv = loss.detach().float().item()
            _log(rank, f"step={step} loss={lv:.4f} epoch={epoch} "
                       f"{(time.monotonic() - t0) / args.log_every * 1000:.0f}ms/step")
            log_metric("train/loss", lv, step)
            t0 = time.monotonic()

        save_now, stop_now = control.sync(step)
        if stop_now:                       # SIGTERM on any rank: every rank saves, then exits
            if last_saved != step:
                save(blocking=True)        # finish the upload before we are killed
            ckpt.wait()
            _log(rank, f"SIGTERM_CHECKPOINTED step={step}; exiting 143")
            dist.destroy_process_group()
            return 143
        if save_now and step < args.steps:  # the final step is saved (blocking) after the loop
            save()
            last_saved = step
        if args.fail_at_step and cold and step == args.fail_at_step:
            # Hard kill (no cleanup, no final save) on every rank: the platform's max_retries
            # resubmits the job; the next attempt must resume from the last complete checkpoint.
            print(f"FORCED_FAILURE rank={rank} step={step} (last checkpoint step={last_saved})",
                  flush=True)
            os._exit(137)

    if last_saved != step:
        save(blocking=True)
    ckpt.close()
    _log(rank, f"TRAIN_DONE step={step} rank0_epoch={epoch} attempt={attempt[0]}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
