# Databricks notebook source
# MAGIC %md
# MAGIC # Mid-epoch resumption for PyTorch on Databricks AIR
# MAGIC
# MAGIC When the dataset is large, an epoch is too coarse a unit of recovery: a failure late in an
# MAGIC epoch should not cost the epoch, and restarting the epoch silently re-trains its head (biasing
# MAGIC the model toward those samples). This notebook walks through what it takes to resume
# MAGIC **at the exact next sample, on every GPU of a multi-node job**, using the AIR SDK
# MAGIC (`databricks.air.data`, formerly `serverless_gpu.data`) for everything it covers, and shows
# MAGIC — live, against your UC volume — each nuance that breaks the naive version.
# MAGIC
# MAGIC **What you take away:** three files you can copy into your project, next to this notebook:
# MAGIC
# MAGIC | File | What it is |
# MAGIC |---|---|
# MAGIC | `resumable_data.py` | SDK import, `ResumableRows` (row-level resume inside large files), per-rank state gather, lockstep checkpoint/SIGTERM control, dataset fingerprint |
# MAGIC | `checkpointing.py` | DCP to the UC volume via `UCVolumeWriter`/`UCVolumeReader`: async save, rank-0-decides resume, retention |
# MAGIC | `train.py` | a complete multi-node DDP/FSDP2 training script that uses both — the template for yours |
# MAGIC
# MAGIC plus `workloads/mid-epoch-resume.example.yaml` (multi-node `air run` with a forced crash and
# MAGIC `max_retries: 1`) and `verify_ledger.py` (proves exactly-once delivery from the run's ledger).
# MAGIC
# MAGIC ### How to run
# MAGIC Attach to a serverless GPU notebook (AI environment v5 or v6; one GPU is enough — sections 1–6
# MAGIC run on CPU worker processes, section 7 optionally trains on the attached GPUs). Fill in the
# MAGIC widgets with a **writable UC volume**. Run top to bottom; every demo ends in a `PROBE … OK`
# MAGIC line that only prints if the behaviour described above it actually happened.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog>", "catalog")
dbutils.widgets.text("schema", "<schema>", "schema")
dbutils.widgets.text("volume", "<volume>", "UC volume (writable)")
dbutils.widgets.text("base_dir", "mid-epoch-resume-walkthrough", "subdir for this notebook")
dbutils.widgets.text("base_site_packages", "/databricks/python3/lib/python3.12/site-packages",
                     "env v5 only: base python site-packages (has serverless_gpu)")

import os
import shutil
import sys

C, S, V = (dbutils.widgets.get(k) for k in ("catalog", "schema", "volume"))
assert "<" not in C + S + V, "Fill in catalog / schema / volume (a writable UC volume) first."
ROOT = f"/Volumes/{C}/{S}/{V}/{dbutils.widgets.get('base_dir')}"
DEMO = f"{ROOT}/demo"                  # tiny dataset for sections 1-6
DEMO_DATA = f"{DEMO}/data"

# This notebook's own folder holds the scripts; make them importable.
try:
    HERE = os.path.dirname("/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook()
                           .getContext().notebookPath().get())
except Exception:  # noqa: BLE001
    HERE = os.getcwd()
sys.path.insert(0, HERE)

# Env v5: serverless_gpu lives in the base python, torch in the AI venv. Bridge if needed (env v6
# ships databricks.air.data next to torch, so this is a no-op there).
try:
    from resumable_data import import_air_data
    air, AIR_PATH = import_air_data()
except ImportError:
    sys.path.append(dbutils.widgets.get("base_site_packages"))
    from resumable_data import import_air_data
    air, AIR_PATH = import_air_data()

import torch
print(f"AIR SDK data module: {AIR_PATH} ({air.__file__})")
print(f"torch {torch.__version__}; scripts from {HERE}; volume root {ROOT}")

# COMMAND ----------

from make_synthetic_dataset import write_dataset

shutil.rmtree(DEMO, ignore_errors=True)
write_dataset(DEMO_DATA, n_files=6, rows_per_file=10, seq=8)   # 60 rows, ids 0..59
ALL_IDS = list(range(60))
print(sorted(os.listdir(DEMO_DATA)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What the SDK already tracks: per-stream, *delivered* progress
# MAGIC
# MAGIC `UCVolumeDataset` lists the directory once (sorted), and deals the files round-robin over
# MAGIC **`world_size × num_workers` independent streams** — one per (rank, DataLoader worker). There
# MAGIC is no global cursor; each stream advances at its own pace in its own forked process.
# MAGIC
# MAGIC `air.DataLoader.state_dict()` collects one small entry per stream, harvested as batches reach
# MAGIC **your loop** — not as workers prefetch them. Below, 3 workers with `prefetch_factor=4` have
# MAGIC already read ahead up to 12 files, but after we consume 4 the state says exactly 4.

# COMMAND ----------

loader = air.DataLoader(air.UCVolumeDataset(DEMO_DATA), batch_size=None, num_workers=3)
it = iter(loader)
consumed = [next(it) for _ in range(4)]          # local cached paths — use immediately, don't keep
state = loader.state_dict()
delivered = sum(s["delivered"] for s in state["per_worker"].values())
print({k: v for k, v in state.items() if k != "per_worker"})
for w, s in sorted(state["per_worker"].items()):
    print(f"  worker {w}: {s}")
print(f"PROBE delivered_not_prefetched {'OK' if delivered == 4 else 'FAIL'} consumed=4 "
      f"state_says={delivered}")
del it, loader

# COMMAND ----------

# MAGIC %md
# MAGIC Restoring that state into a **fresh** loader (what a resubmitted container does) continues every
# MAGIC stream at its next undelivered file: exactly-once at **file** granularity.

# COMMAND ----------

import sdk_behavior_probe as probe   # the same scenario functions the repo's probe asserts on

probe.probe_file_level(DEMO_DATA, None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Files are too coarse: you need a row cursor — and the obvious wrappers lose data
# MAGIC
# MAGIC Real datasets are a few large files (parquet, .npy, MDS shards…), each fanned into many rows.
# MAGIC Resuming at the file level means a stream that was halfway through a 2 GB shard either
# MAGIC replays or loses half of it. Your wrapper dataset has to carry a **row cursor inside the
# MAGIC current file** and implement `state_dict()` / `load_state_dict()` itself (`air.DataLoader`
# MAGIC only checkpoints datasets that do — a wrapper without them makes `loader.state_dict()` raise).
# MAGIC
# MAGIC Three tempting implementations are wrong. Each is run below on the same interruption:
# MAGIC
# MAGIC 1. **Delegate `state_dict()` to the inner `UCVolumeDataset`.** The SDK docs describe this as
# MAGIC    "file-granularity, the seam file is replayed (at-least-once)". It is the opposite: the inner
# MAGIC    stream counts a file as delivered the moment it *yields* the path — before you've emitted its
# MAGIC    rows — so the rest of the seam file is **skipped**. Silent data loss.
# MAGIC 2. **The SDK docs' row-cursor example, snapshotting the inner state in `__init__`.** That
# MAGIC    snapshot is taken in the main process, so it is stamped `worker_id=0` (and `rank=0` if built
# MAGIC    before `init_process_group`). Checkpoint while worker 1 is still in its first file and the
# MAGIC    restore **raises** "Cannot restore a checkpoint into a different stream". With large files
# MAGIC    the first file lasts a long time — this window is most of a short job.
# MAGIC 3. **The same example in epoch ≥ 2.** `air.DataLoader` keeps workers alive across epochs, so the
# MAGIC    wrapper's snapshot still holds the *previous* epoch's final state until the first file of the
# MAGIC    new epoch finishes. Checkpoint in that window and the resume **skips the whole epoch slice**.

# COMMAND ----------

before, after, err = probe.interrupted(DEMO_DATA, "delegate", nw=2, warm_epochs=0, k_batches=2)
d, m = probe.summarize(ALL_IDS, before, after or [])
print(f"1. delegating wrapper      : error={err} duplicated={d} LOST={m}")
before, after, err = probe.interrupted(DEMO_DATA, "doc", nw=2, warm_epochs=0, k_batches=2)
print(f"2. docs example, 1st file  : error={err}")
before, after, err = probe.interrupted(DEMO_DATA, "doc", nw=2, warm_epochs=1, k_batches=2)
d, m = probe.summarize(ALL_IDS, before, after or [])
print(f"3. docs example, epoch 2   : error={err} duplicated={d} LOST={m} of {len(ALL_IDS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The fix: `ResumableRows` (in `resumable_data.py`)
# MAGIC
# MAGIC The snapshot of the inner stream is taken **inside the worker, right after the inner stream
# MAGIC yields each file**, minus one (so it means "about to start this file"), and the row cursor counts
# MAGIC rows *before* yielding them (the loader snapshots state while your generator is paused at the
# MAGIC `yield`). It never relies on state from `__init__` or from a previous epoch. You supply one
# MAGIC function, `parse_file(local_path, start_row)`, that reads a cached file from `start_row` — make
# MAGIC it **deterministic** (the seam file is re-read on resume) and **seek** if your format allows
# MAGIC (mmap'd `.npy`, a parquet row group) so resuming inside a huge file doesn't re-read its head.
# MAGIC
# MAGIC ```python
# MAGIC files  = air.UCVolumeDataset("/Volumes/.../data")
# MAGIC rows   = ResumableRows(files, parse_file, shuffle_seed=None)
# MAGIC loader = air.DataLoader(rows, batch_size=16, num_workers=4,
# MAGIC                         worker_init_fn=ignore_sigterm_in_worker, multiprocessing_context="fork")
# MAGIC ```
# MAGIC
# MAGIC Same two interruptions, now exactly-once:

# COMMAND ----------

for label, warm in (("1st file", 0), ("epoch 2", 1)):
    before, after, err = probe.interrupted(DEMO_DATA, "rows", nw=2, warm_epochs=warm, k_batches=2)
    d, m = probe.summarize(ALL_IDS, before, after or [])
    ok = err is None and d == 0 and m == 0
    print(f"PROBE resumable_rows_{label.replace(' ', '_')} {'OK' if ok else 'FAIL'} "
          f"before={len(before)} after={len(after or [])} duplicated={d} lost={m}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exactly-once is a property of the *set*, not the order
# MAGIC
# MAGIC Torch hands out worker batches round-robin. A resumed loader re-forks its workers and starts
# MAGIC the round-robin at worker 0, wherever the interrupted run was in the cycle. So with
# MAGIC `num_workers > 1` the **remaining rows are the same, but their order can differ** from an
# MAGIC uninterrupted run. That is fine for SGD; it means "bit-identical loss curve after resume" is not
# MAGIC a reasonable acceptance test at `num_workers > 1` — "no row repeated or skipped" is.
# MAGIC (With `num_workers ≤ 1` the order is reproduced exactly — section 4 shows it.)

# COMMAND ----------

probe.probe_order(DEMO_DATA, ALL_IDS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Shuffling: the SDK never reorders files — shuffle when you write, or within a file
# MAGIC
# MAGIC `UCVolumeDataset` streams each stream's files in sorted order, every epoch. Pre-shuffle rows
# MAGIC across files when you write the dataset. If you also want a different order per epoch,
# MAGIC `ResumableRows(shuffle_seed=…)` permutes rows **within each file** as a pure function of
# MAGIC `(seed, epoch, file position)` — so a resumed run reproduces the permutation (it materializes one
# MAGIC file's rows in memory to do so). A global RNG re-seeded at epoch start would *not* match a
# MAGIC mid-epoch offset. Below: epoch 1's order differs from epoch 0's, and an interrupted epoch 1
# MAGIC (single worker) resumes into the identical sequence.

# COMMAND ----------

probe.probe_shuffle(DEMO_DATA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The grid is pinned: same node count, same `num_workers`, same files
# MAGIC
# MAGIC Each saved offset is a position inside one stream's slice of the file list, and the slices are
# MAGIC defined by `world_size × num_workers`. So:
# MAGIC
# MAGIC - **Resume on the same shape.** DCP can reshard *model* weights onto a different GPU count; the
# MAGIC   data position cannot follow. `load_state_dict` raises on a different grid (below). A
# MAGIC   `max_retries` resubmit reuses the workload's shape, so this holds automatically there — it
# MAGIC   bites when someone "just reruns it on 4 nodes instead of 2".
# MAGIC - **Set `num_workers` explicitly** and never change it between attempts. `air.DataLoader`'s
# MAGIC   default depends on whether `torch.distributed` is initialised (6 vs 48), so an innocuous
# MAGIC   refactor can change it.
# MAGIC - **Freeze the dataset.** Offsets are positions in the sorted listing: adding, removing, or
# MAGIC   rewriting a file between attempts silently shifts every stream. `train.py` stores a
# MAGIC   `dataset_fingerprint` (names + sizes) in the checkpoint and refuses to resume on a mismatch,
# MAGIC   and all ranks compare fingerprints at startup (a node listing FUSE before another node's fresh
# MAGIC   writes are visible would otherwise build different streams). Write new data to a new,
# MAGIC   versioned directory.
# MAGIC - **Size the file count to the grid.** `files / (world × num_workers)` files per stream; an
# MAGIC   empty stream makes its rank spin, and a ragged split makes some ranks' epochs shorter. At
# MAGIC   2 nodes × 8 GPUs × 4 workers that is 64 streams — plan for hundreds of files, not tens.

# COMMAND ----------

probe.probe_grid(DEMO_DATA)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Multi-node: what changes when there are many ranks
# MAGIC
# MAGIC Everything above is per process. On a multi-node job every rank has its own loader state, and
# MAGIC the checkpoint has to capture all of them consistently with the weights. The rules, each of
# MAGIC which `train.py` + `checkpointing.py` implement and the local multi-node pre-flight exercised:
# MAGIC
# MAGIC 1. **One DCP save holds weights *and* data position.** `UCVolumeWriter` publishes `.metadata`
# MAGIC    only after every shard landed, so a step directory with `.metadata` is a consistent pair and
# MAGIC    one without is ignored. Never write the data position to a side file.
# MAGIC 2. **Gather every rank's loader state into one replicated entry.** DCP stores a non-tensor value
# MAGIC    *once per key* — per-rank values under a shared key collapse to one rank's (demonstrated in
# MAGIC    the next cell). `gather_data_state` all-gathers `{epoch, loader.state_dict(), rng}` into a JSON
# MAGIC    string that is identical on all ranks; each rank picks its own entry on resume.
# MAGIC 3. **Rank 0 picks the checkpoint and broadcasts it.** Ranks listing the FUSE dir independently
# MAGIC    can disagree while a save or prune is in flight.
# MAGIC 4. **Decide to checkpoint collectively.** The save is a collective; a wall-clock trigger
# MAGIC    evaluated per rank drifts and hangs the job. `StepControl` all-reduces "due" and "SIGTERM" each
# MAGIC    step (two floats). Step-count triggers are naturally identical across ranks.
# MAGIC 5. **Give async checkpoint collectives their own gloo group, and wait before gathering.**
# MAGIC    `dcp.async_save` finishes its upload with collectives on a background thread; on the default
# MAGIC    group they interleaved with DDP's all-reduce and crashed the local multi-node pre-flight
# MAGIC    ("Is there a distributed collective mismatch in your code?"). `init_process_group` still needs
# MAGIC    `"cpu:gloo,cuda:nccl"`.
# MAGIC 6. **Budget training in steps; let each rank keep its own epoch counter.** Ranks own different
# MAGIC    files, so they do not finish an epoch on the same step. A shared epoch boundary either hangs
# MAGIC    (one rank stops calling collectives) or drops data; each rank cycling its own loader does
# MAGIC    neither, and every rank-epoch stays exactly-once. Convert "N epochs" to steps up front.
# MAGIC 7. **SIGTERM: checkpoint on the way out — and keep the DataLoader workers alive.** torchrun
# MAGIC    forwards SIGTERM to each rank's whole process group, DataLoader workers included; when they
# MAGIC    die torch raises *inside your emergency save* and you fall back to the previous checkpoint
# MAGIC    (observed in the pre-flight). `ignore_sigterm_in_worker` as the `worker_init_fn` fixes it.
# MAGIC    Whether — and with how much grace — AIR sends SIGTERM before stopping a workload is **not yet
# MAGIC    verified**; treat the emergency save as best-effort and rely on the periodic cadence.
# MAGIC 8. **`multiprocessing_context="fork"`.** The SDK loader refuses other start methods (its workers
# MAGIC    share one local cache dir). It is the default under torchrun on Linux today, but not inside
# MAGIC    `mp.spawn` children or on Python ≥ 3.14 — pass it explicitly.
# MAGIC 9. **Work after the last checkpoint is lost and retrained.** That is correct (its gradient
# MAGIC    updates were lost too), and it is the real cost knob: checkpoint interval × step time ×
# MAGIC    cluster cost. With async saves the interval can be short; time-based cadence
# MAGIC    (`--ckpt-every-minutes`) keeps it bounded when step time varies.
# MAGIC 10. **MLflow across retries.** AIR resumes the same MLflow run across `max_retries` attempts, so
# MAGIC     the retrained window logs its steps a second time — expect a short overlap in the charts.

# COMMAND ----------

probe.probe_dcp(f"{DEMO}/dcp")   # 2 gloo processes on this node, real UCVolumeWriter/Reader

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Try it: crash and resume `train.py` on this notebook's GPUs (single node)
# MAGIC
# MAGIC Same script as the multi-node workload. Attempt 1 is hard-killed at step 140, 40 steps after
# MAGIC the step-100 checkpoint and in the middle of a file; attempt 2 is the identical command and
# MAGIC must resume at step 100 (mid-epoch) and finish. Then `verify_ledger.py` checks exactly-once
# MAGIC delivery from the per-rank ledger. Takes a few minutes on one A10.

# COMMAND ----------

import subprocess

import torch

RUN = f"{ROOT}/try-it"
shutil.rmtree(RUN, ignore_errors=True)
NPROC = max(1, torch.cuda.device_count())
env = dict(os.environ)
if AIR_PATH == "serverless_gpu.data":       # env v5: bridge the base-python SDK into the ranks
    env["PYTHONPATH"] = dbutils.widgets.get("base_site_packages") + ":" + env.get("PYTHONPATH", "")


def attempt(n):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NPROC}",
           "--max-restarts=0", f"{HERE}/train.py",
           "--data-dir", f"{RUN}/data", "--ckpt-dir", f"{RUN}/ckpt", "--ledger-dir", f"{RUN}/ledger",
           "--make-data", "16", "--rows-per-file", "64", "--seq", "64",   # several epochs in 300 steps
           "--steps", "300", "--batch-size", "16", "--num-workers", "2",
           "--ckpt-every-steps", "50", "--fail-at-step", "140", "--log-every", "50"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    keep = ("ENV", "DATA", "COLD_START", "RESUMED", "CKPT_SAVED", "FORCED", "TRAIN_DONE", "Error")
    print(f"--- attempt {n}: exit {p.returncode}")
    print("\n".join(l for l in (p.stdout + p.stderr).splitlines() if any(k in l for k in keep)))
    return p


a1 = attempt(1)
a2 = attempt(2)
assert a1.returncode != 0 and "FORCED_FAILURE" in a1.stdout, "attempt 1 should have been killed"
assert a2.returncode == 0 and "RESUMED_FROM_STEP=100" in a2.stdout, "attempt 2 should resume at 100"
v = subprocess.run([sys.executable, f"{HERE}/verify_ledger.py", "--ledger-dir", f"{RUN}/ledger",
                    "--data-dir", f"{RUN}/data", "--expect-resume"], capture_output=True, text=True)
print(v.stdout, v.stderr[-2000:])
assert "LEDGER_EXACTLY_ONCE_OK" in v.stdout

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run it multi-node with a real platform retry
# MAGIC
# MAGIC Multi-node on AIR goes through the `air` CLI (one `torchrun` per node, rendezvous via the
# MAGIC injected `NUM_NODES` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT`). From a checkout of this repo:
# MAGIC
# MAGIC ```bash
# MAGIC cp workloads/mid-epoch-resume.example.yaml workloads/mid-epoch-resume.yaml
# MAGIC # fill in the /Volumes/<catalog>/<schema>/<volume> paths, then:
# MAGIC air run workloads/mid-epoch-resume.yaml
# MAGIC ```
# MAGIC
# MAGIC 2 × `GPU_8xH100` (16 ranks × 4 workers = 64 streams over 256 files). Attempt 0 is killed at step
# MAGIC 700; `max_retries: 1` resubmits; attempt 1 prints `RESUMED_FROM_STEP=600` and finishes; node 0
# MAGIC then prints `LEDGER_EXACTLY_ONCE_OK`. Check node 1's log too (`air logs <run_id> --node 1`).
# MAGIC
# MAGIC **Adapting `train.py` to your job:** replace `TinyLM` with your model, `parse_npy_rows` with a
# MAGIC `parse_file(local_path, start_row)` for your format, and drop the three `[demo]` flags
# MAGIC (`--make-data`, `--fail-at-step`, `--ledger-dir`). Everything else — resume discovery, the data
# MAGIC state, cadence, SIGTERM, retention — is meant to stay as is.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checklist
# MAGIC
# MAGIC - [ ] `air.DataLoader` over a **checkpointable** wrapper with a row cursor (`ResumableRows`), not a
# MAGIC       delegating one; deterministic, seeking `parse_file`
# MAGIC - [ ] weights + optimizer + every rank's `{epoch, loader state, RNG}` in **one** DCP save to a UC
# MAGIC       volume (never `/tmp`), via `UCVolumeWriter`; `.metadata` = complete
# MAGIC - [ ] rank 0 chooses the checkpoint; checkpoint decisions all-reduced; async-save collectives on
# MAGIC       their own gloo group; wait before the next gather
# MAGIC - [ ] same nodes × GPUs × `num_workers` on every attempt; `num_workers` explicit; dataset frozen
# MAGIC       (fingerprint checked); file count ≫ streams
# MAGIC - [ ] step budget with per-rank epochs; no shared epoch barrier
# MAGIC - [ ] SIGTERM → collective emergency save; DataLoader workers ignore SIGTERM;
# MAGIC       `multiprocessing_context="fork"`
# MAGIC - [ ] entrypoint idempotent (always "resume if a checkpoint exists"), `max_retries ≥ 1`
# MAGIC - [ ] acceptance = exactly-once per epoch (ledger), not a bit-identical loss curve
