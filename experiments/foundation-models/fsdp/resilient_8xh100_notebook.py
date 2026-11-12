# Databricks notebook source
# MAGIC %md
# MAGIC # Resilient FSDP training on 8×H100 — AIR performance & resiliency best practices
# MAGIC
# MAGIC A single, self-contained training job that turns the Databricks AIR **Performance and
# MAGIC resiliency** guide
# MAGIC ([docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency))
# MAGIC into runnable code. Every load-bearing claim on that page is demonstrated and printed as a
# MAGIC sentinel that only appears if the property actually held:
# MAGIC
# MAGIC | # | Best practice (doc section) | Sentinel |
# MAGIC |---|---|---|
# MAGIC | 1 | **Load data efficiently to minimize idle GPU time** — the AIR SDK's `UCVolumeDataset` + `DataLoader` (`databricks.air.data` on env v6+, `serverless_gpu.data` on v5), auto cross-rank/worker sharding (no `DistributedSampler`), local cache | `RESILIENT_DATALOAD_OK` |
# MAGIC | 2 | **Checkpoint with DCP** + **Save asynchronously** — `dcp.async_save` via `UCVolumeWriter`, each rank writes its shard in parallel, prior save awaited before the next | `RESILIENT_ASYNC_CKPT_OK` |
# MAGIC | 3 | **Recover automatically from the most recent valid checkpoint** — pick the newest dir with a `.metadata` file, load via `UCVolumeReader` + `set_state_dict`, resume at the saved step | `RESILIENT_RESUME_OK` |
# MAGIC | 4a | **Checkpoint the data pipeline** — every rank's exact position (per DataLoader worker: which shard, which row inside it) rides in the *same* DCP save as the weights; restored into a fresh loader, the next batches replay nothing this rank already trained | `RESILIENT_PIPELINE_STATE_OK` |
# MAGIC | 4b | **…across a real relaunch** — after cell 4 resumes mid-epoch, the per-rank ledger proves every row of every completed epoch was trained exactly once (no repeats, no skips, no gaps in steps) | `RESILIENT_PIPELINE_OK` (verify cell) |
# MAGIC
# MAGIC > **Changed 2026-11-09 (proof 4 was vacuous before).** The earlier version saved `(epoch,
# MAGIC > sample_offset)` but on resume rebuilt the loader from the start of the epoch and never skipped
# MAGIC > ahead, so the resumed run re-trained the epoch's head; its shuffle order was computed and thrown
# MAGIC > away; and its check compared the first batch of two fresh loaders (always equal). It also kept
# MAGIC > `epoch`/`offset` in a shared non-tensor DCP entry, which DCP stores once (one rank's value) — so
# MAGIC > per-rank positions could not have survived anyway. Proof 4 now uses the SDK's own
# MAGIC > `DataLoader.state_dict()` plus a row cursor, gathered per rank. Background and the full set of
# MAGIC > rules: `experiments/dataloading/mid-epoch-resume/` (walkthrough notebook + scripts).
# MAGIC
# MAGIC The FSDP2 sharding (`fully_shard`) and bf16 autocast underneath are the standard 8×H100
# MAGIC training substrate — they carry the four properties above; they aren't themselves the point
# MAGIC of this doc.
# MAGIC
# MAGIC ### ⚠️ Launch path: `torchrun`, **not** `@distributed`
# MAGIC This notebook drives the 8 GPUs with **`torchrun --standalone --nproc_per_node=8`** in a
# MAGIC subprocess cell — the same elastic-launch you already use on MLR / bare metal. It
# MAGIC **deliberately does not use** `serverless_gpu`'s `@distributed` decorator: that convenience
# MAGIC layer is Beta and is out of scope for the customer engagement driving this repo (see
# MAGIC `AGENTS.md` and `docs/private/uat-plan-2026-07.md`). The `serverless_gpu.data` **data and
# MAGIC checkpoint utilities** (UCVolumeDataset / UCVolumeWriter / UCVolumeReader) are a different,
# MAGIC in-scope part of the package and are exactly what the doc prescribes — this notebook uses
# MAGIC those on the torchrun path.
# MAGIC
# MAGIC ### How to run
# MAGIC 1. Attach to an **AI base environment (v5)** with a **`GPU_8xH100`** accelerator. (v4 silently
# MAGIC    breaks GPU-job storage egress — use v5.)
# MAGIC 2. Fill in the **catalog / schema / volume** widgets — you need a writable UC volume for the
# MAGIC    token shards and the checkpoints. Checkpoints must live on a **durable UC volume, never
# MAGIC    `/tmp`**, or a resumed container has nothing to recover from.
# MAGIC 3. Run cells top to bottom. Cell 3 launches training; cell 4 relaunches it to prove
# MAGIC    auto-resume (mid-epoch, mid-shard) from the last valid checkpoint; cell 5 verifies the ledger.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Fill in `catalog` / `schema` / `volume` (a writable UC volume). Model shape defaults to a
# MAGIC small-but-real GPT so the 8 H100s are busy without a long run; scale `layers`/`dim`/`batch`
# MAGIC up for a heavier soak.

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog>", "catalog")
dbutils.widgets.text("schema", "<schema>", "schema")
dbutils.widgets.text("volume", "<volume>", "UC volume (writable)")
dbutils.widgets.text("base_dir", "resilient-fsdp", "subdir in the volume for this job")
dbutils.widgets.text("nproc", "8", "GPUs on this node (nproc_per_node)")
# model / training shape
dbutils.widgets.text("layers", "12", "transformer layers")
dbutils.widgets.text("dim", "1024", "model dim")
dbutils.widgets.text("heads", "16", "attention heads")
dbutils.widgets.text("seq", "512", "sequence length")
dbutils.widgets.text("batch", "8", "per-GPU micro-batch (sequences)")
dbutils.widgets.text("steps", "120", "optimizer steps")
dbutils.widgets.text("save_every", "40", "checkpoint every N steps")
dbutils.widgets.text("shards", "32", "synthetic token shards to stage")
dbutils.widgets.text("num_workers", "2", "DataLoader workers per GPU (pinned into the checkpoint)")
dbutils.widgets.dropdown("reset", "true", ["true", "false"],
                         "delete this notebook's ckpt + ledger dirs first (fresh demo)")
# v5-image paths for the PYTHONPATH bridge + ai-venv torchrun (verified 2026-08-27; version-scoped)
dbutils.widgets.text("ai_torchrun", "/opt/databricks-environments/databricks-ai/bin/torchrun",
                     "databricks-ai venv torchrun")
dbutils.widgets.text("base_site_packages", "/databricks/python3/lib/python3.12/site-packages",
                     "base python site-packages (has serverless_gpu)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
BASE_DIR = dbutils.widgets.get("base_dir")
NPROC = int(dbutils.widgets.get("nproc"))

assert "<" not in (CATALOG + SCHEMA + VOLUME), (
    "Fill in the catalog / schema / volume widgets with a writable UC volume before running.")

VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{BASE_DIR}"
TOKENS_DIR = f"{VOL_ROOT}/tokens_v2"   # UCVolumeDataset streams shards from here (v2: has row_id)
CKPT_DIR = f"{VOL_ROOT}/ckpt_v2"       # DCP checkpoints land here (durable — NOT /tmp)
LEDGER_DIR = f"{VOL_ROOT}/ledger_v2"   # per-rank record of trained row ids (proof 4b)
SCRIPT_PATH = "/tmp/resilient_train.py"

import shutil
if dbutils.widgets.get("reset") == "true":
    for d in (CKPT_DIR, LEDGER_DIR):   # only this notebook's own subdirs; token shards are kept
        if os.path.isdir(d):
            shutil.rmtree(d)
            print("reset: removed", d)
print("Token shards :", TOKENS_DIR)
print("Checkpoints  :", CKPT_DIR)
print("Ledger       :", LEDGER_DIR)
print("nproc        :", NPROC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment check
# MAGIC Confirm we're on an 8-GPU H100 node with an FSDP2-capable runtime, and that
# MAGIC `serverless_gpu.data` resolves once the base-python site-packages are bridged onto this
# MAGIC torch-having interpreter (it lives in the base python, torch lives in the ai venv — neither
# MAGIC has both by default).

# COMMAND ----------

import sys, os

import torch
from torch.distributed.fsdp import fully_shard

n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
print(f"visible GPUs: {n_gpu}")
for i in range(n_gpu):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
print("fully_shard (FSDP2) available:", callable(fully_shard))
assert n_gpu == NPROC, f"expected {NPROC} GPUs but see {n_gpu} — attach a GPU_8xH100 (or set nproc)"

# Probe the serverless_gpu.data import through the bridge (same mechanism the torchrun launch uses).
BASE_SP = dbutils.widgets.get("base_site_packages")
_bridged = False
if BASE_SP not in sys.path and os.path.isdir(BASE_SP):
    sys.path.insert(0, BASE_SP); _bridged = True
try:
    import importlib
    _sgc = None
    for _name in ("databricks.air.data", "serverless_gpu.data", "databricks.serverless_gpu.data"):
        try:
            _sgc = importlib.import_module(_name)
            break
        except ImportError:
            continue
    if _sgc is None:
        raise ImportError("none of databricks.air.data / serverless_gpu.data importable")
    print(f"{_name} OK:",
          [n for n in ("UCVolumeDataset", "DataLoader", "UCVolumeWriter", "UCVolumeReader")
           if hasattr(_sgc, n)])
    # Proof 4 needs loader checkpointing (serverless_gpu >= 0.5.21 / databricks-sdk-air >= 0.1.0).
    assert hasattr(_sgc.DataLoader, "load_state_dict"), (
        f"{_name} at {_sgc.__file__} predates DataLoader.state_dict/load_state_dict — proof 4 will fail")
    print("DataLoader checkpointing available: True")
except Exception as e:
    print(f"⚠️ serverless_gpu.data did NOT import ({type(e).__name__}: {e}). "
          f"The launch cell bridges {BASE_SP} onto the ai-venv torchrun; if this persists, "
          f"check the base_site_packages widget against this image.")
finally:
    if _bridged:
        sys.path.remove(BASE_SP)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The training script
# MAGIC Written to `/tmp` and launched by `torchrun` below (torchrun needs a file; a notebook cell
# MAGIC can't be an entrypoint). Read the comments — each best practice is labelled with its doc
# MAGIC section. It's self-contained and egress-free: rank 0 stages deterministic synthetic token
# MAGIC shards into the volume, so you don't need a pre-staged corpus (point `--tokens-dir` at a real
# MAGIC one to stream that instead).

# COMMAND ----------

from pathlib import Path

TRAIN_SCRIPT = r'''
# resilient_train.py — written by resilient_8xh100_notebook.py; launched via torchrun --standalone.
# Demonstrates the AIR performance-and-resiliency doc's best practices; each prints a sentinel that
# is unreachable unless its assertion held. The data-pipeline pieces are a compact copy of
# experiments/dataloading/mid-epoch-resume/ (resumable_data.py, checkpointing.py) — read that
# walkthrough for the why behind each rule.
import argparse, hashlib, json, os, shutil, signal, sys, time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset


# --- AIR SDK data module resolver: databricks.air.data on GPU env v6+, serverless_gpu.data on v5.
#     Loader checkpointing (DataLoader.state_dict/load_state_dict) needs serverless_gpu >= 0.5.21.
def import_air_data():
    import importlib
    if os.environ.get("RESILIENT_LOCAL_FS") == "1":
        # --local pre-flight only: lift the SDK's /Volumes prefix check so the REAL SDK code runs
        # against a local dir. Never set on AIR.
        for m in ("databricks.air.data.fs_layer", "serverless_gpu.utils.fs_layer"):
            try:
                importlib.import_module(m).CachingUCFuseFSLayer._validate_path = lambda self, p: None
            except Exception:  # noqa: BLE001
                pass
    errors = {}
    for path in ("databricks.air.data", "serverless_gpu.data", "databricks.serverless_gpu.data"):
        try:
            mod = importlib.import_module(path)
        except Exception as e:  # noqa: BLE001
            errors[path] = f"{type(e).__name__}: {e}"
            continue
        if not hasattr(mod.DataLoader, "load_state_dict"):
            raise ImportError(f"{path} predates DataLoader checkpointing (need serverless_gpu >= 0.5.21)")
        return mod, path
    raise ImportError(f"AIR SDK data module not found. tried={errors} python={sys.executable}")


# --- synthetic corpus: payload is a pure function of (shard,row,col) — deterministic, egress-free.
#     Each row carries a global row_id so the ledger can prove exactly-once delivery.
def _synth_row(shard, row, seq_len, vocab):
    base = (shard * 1000003 + row * 97) % vocab
    return [(base + col * 31) % vocab for col in range(seq_len)]


def stage_synthetic_shards(tokens_dir, n_shards, rows_per_shard, seq_len, vocab):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(tokens_dir, exist_ok=True)
    for s in range(n_shards):
        path = os.path.join(tokens_dir, f"shard_{s:05d}.parquet")
        if os.path.exists(path):
            continue
        rows = [_synth_row(s, r, seq_len, vocab) for r in range(rows_per_shard)]
        ids = [s * rows_per_shard + r for r in range(rows_per_shard)]
        pq.write_table(pa.table({"row_id": ids, "input_ids": rows}), path)


def read_rows(local_path, start):
    # ParseFn: rows [start:] of one cached shard. Deterministic (the seam shard is re-read on
    # resume). A large-file reader would map `start` to a parquet row group instead of slicing.
    import pyarrow.parquet as pq
    tbl = pq.read_table(local_path, columns=["row_id", "input_ids"])
    ids, toks = tbl.column("row_id").to_pylist(), tbl.column("input_ids").to_pylist()
    for i in range(start, len(ids)):
        yield {"row_id": ids[i], "tokens": torch.tensor(toks[i], dtype=torch.long)}


# === BEST PRACTICE 1 — Load data efficiently (doc: "Load data efficiently to minimize idle GPU").
# UCVolumeDataset yields shard paths, auto-partitioned across ranks x workers (no DistributedSampler)
# and locally cached; the SDK DataLoader prefetches and forces persistent_workers=True.
#
# === BEST PRACTICE 4 — Checkpoint the data pipeline (doc section of that name).
# The SDK DataLoader's state_dict() tracks, per (rank, worker) stream, what was DELIVERED to this
# loop (not what was prefetched) — but at FILE granularity: a shard part-way through would be
# replayed from row 0. ResumableRows adds the row cursor inside the current shard. (It also avoids
# two first-file bugs in the SDK docs' wrapper example: it snapshots the inner state inside the
# worker, after the inner stream yields — never in __init__ or carried over from the last epoch.)
class ResumableRows(IterableDataset):
    def __init__(self, files, parse_file):
        super().__init__()
        self._files, self._parse = files, parse_file
        self._file_start, self._rows, self._resume = None, 0, None

    def __iter__(self):
        skip, self._resume = (self._resume or 0), None
        for local_path in self._files:
            start = dict(self._files.state_dict())   # SDK counts a file delivered once yielded
            assert start.get("version") == 1 and "delivered" in start, start
            start["delivered"] -= 1                  # -> "about to start this file"
            self._file_start = start
            for i, row in enumerate(self._parse(local_path, skip), start=skip):
                self._rows = i + 1                    # count before yield (loader snapshots here)
                yield row
            skip = 0

    def state_dict(self):
        rows = self._resume if self._resume is not None else self._rows
        return {"version": 1, "files": self._file_start, "rows": rows}

    def load_state_dict(self, state):
        if state["files"] is not None:
            self._files.load_state_dict(state["files"])
        self._file_start, self._rows, self._resume = state["files"], state["rows"], state["rows"]


def _ignore_sigterm(worker_id):
    # torchrun forwards SIGTERM to each rank's process group, DataLoader workers included; if they
    # die mid-save the rank raises. Workers still exit with their rank.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def build_loader(args, mod):
    rows = ResumableRows(mod.UCVolumeDataset(args.tokens_dir), read_rows)
    # The SDK loader requires the 'fork' start method; say so explicitly, because the default is
    # 'spawn' inside mp.spawn children (the --local path) and 'forkserver' on Python >= 3.14.
    return mod.DataLoader(rows, batch_size=args.batch, num_workers=args.num_workers,
                          worker_init_fn=_ignore_sigterm, multiprocessing_context="fork")


def gather_data_state(loader, epoch):
    # Collective. DCP keeps ONE copy of a non-tensor value per key, so per-rank loader states under
    # a shared key would collapse to one rank's. Gather all of them into one replicated JSON string.
    everyone = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, {"epoch": epoch, "loader": loader.state_dict()})
    return json.dumps(everyone)


# --- compact GPT-2-style causal LM; FSDP2 fully_shard per block + root; bf16 autocast. -----------
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.ln1(x)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, dim, heads, layers, seq):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def build_fsdp_model(args, device):
    model = GPT(args.vocab, args.dim, args.heads, args.layers, args.seq).to(device)
    if not args.local and device.type == "cuda":
        from torch.distributed.fsdp import fully_shard
        for blk in model.blocks:
            fully_shard(blk)
        fully_shard(model)
    elif dist.get_world_size() > 1:
        # --local: DDP keeps the CPU ranks' replicas identical. Without it each rank trains its own
        # model, and DCP (which saves one copy of a replicated, unsharded tensor) restores rank 0's
        # weights everywhere -> the resume fingerprint can't match.
        model = nn.parallel.DistributedDataParallel(model)
    return model


def lm_loss(model, tokens, device, use_bf16):
    tokens = tokens.to(device)
    x, y = tokens[:, :-1], tokens[:, 1:]
    if use_bf16 and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))


# === BEST PRACTICE 2 — DCP + async save (doc: "Checkpoint with DCP" / "Save asynchronously").
# === BEST PRACTICE 3 — recover from the most recent VALID checkpoint (doc section of that name).
# One DCP save per step_N dir: model + optimizer + app step + every rank's data state, so the
# weights and the data cursor are atomic together (UCVolumeWriter publishes .metadata last).
class Checkpointer:
    def __init__(self, ckpt_dir, mod):
        self.ckpt_dir, self.mod = ckpt_dir, mod
        self._future = None
        self.saves = 0
        # async_save finishes its upload with collectives on a background thread: give them their
        # own gloo group so they can't interleave with the training loop's collectives.
        self.pg = dist.new_group(backend="gloo")
        if dist.get_rank() == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
        dist.barrier()

    def _valid(self):
        out = []
        for name in os.listdir(self.ckpt_dir) if os.path.isdir(self.ckpt_dir) else []:
            if name.startswith("step_") and os.path.exists(os.path.join(self.ckpt_dir, name, ".metadata")):
                try:
                    out.append((int(name.split("_", 1)[1]), os.path.join(self.ckpt_dir, name)))
                except ValueError:
                    pass
        return sorted(out)

    def latest_valid(self):
        # Rank 0 decides and broadcasts: ranks listing FUSE independently can disagree mid-save.
        box = [None]
        if dist.get_rank() == 0:
            v = self._valid()
            box[0] = v[-1] if v else None
        dist.broadcast_object_list(box, src=0, group=self.pg)
        return box[0]

    def wait(self):
        if self._future is not None:
            getattr(self._future, "upload_completion", self._future).result()
            self._future = None

    def async_save(self, step, model, opt, data_json):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        self.wait()                                     # one save in flight at a time
        path = os.path.join(self.ckpt_dir, f"step_{step:09d}")
        if dist.get_rank() == 0 and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)     # torn save of this step from an old attempt
        dist.barrier()
        msd, osd = get_state_dict(model, opt)           # sharded DTensors, model + optim together
        state = {"model": msd, "optim": osd, "app": {"global_step": step}, "data": data_json}
        self._future = dcp.async_save(state, storage_writer=self.mod.UCVolumeWriter(path),
                                      process_group=self.pg)
        self.saves += 1
        return path

    def load(self, model, opt, path):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        msd, osd = get_state_dict(model, opt)
        state = {"model": msd, "optim": osd, "app": {"global_step": -1}, "data": ""}
        dcp.load(state, storage_reader=self.mod.UCVolumeReader(path), process_group=self.pg)
        set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
        return int(state["app"]["global_step"]), state["data"]


class Ledger:
    # (step, epoch, row_ids) per trained batch, flushed WITH each checkpoint — so on disk it is
    # exactly the trajectory a resume continues from. The notebook's verify cell reads it.
    def __init__(self, root, attempt, rank):
        self.dir = os.path.join(root, f"attempt_{attempt}")
        self.rank, self.buf = rank, []
        os.makedirs(self.dir, exist_ok=True)

    def record(self, step, epoch, ids):
        self.buf.append({"step": step, "epoch": epoch, "ids": ids})

    def flush(self, step):
        if self.buf:
            with open(os.path.join(self.dir, f"rank_{self.rank:05d}_upto_{step:09d}.jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in self.buf))
            self.buf = []


def _fingerprint(model, opt):
    from torch.distributed.checkpoint.state_dict import get_state_dict
    msd, osd = get_state_dict(model, opt)
    name = sorted(msd.keys())[0]

    def _g(t):
        return t.full_tensor() if hasattr(t, "full_tensor") else t
    parts = [_g(msd[name]).detach().to(torch.float64).cpu().numpy().tobytes()]
    st = osd.get("state", {}).get(name, {})
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st:
            parts.append(_g(st[k]).detach().to(torch.float64).cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def _p(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def _all(flag, device):
    t = torch.tensor([0.0 if flag else 1.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item() == 0.0


def worker(rank, world, args):
    # dcp.async_save needs a CPU/gloo backend alongside NCCL ("A CPU backend must be enabled for
    # async save"). --local is pure gloo (CPU).
    backend = "gloo" if args.local else "cpu:gloo,cuda:nccl"
    if args.local:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        os.environ["RESILIENT_LOCAL_FS"] = "1"
    dist.init_process_group(backend, rank=rank, world_size=world)
    if not args.local:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cpu" if args.local else "cuda")
    torch.manual_seed(args.seed)                       # deterministic base seeding
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    mod, sdk_path = import_air_data()

    # stage the corpus (rank 0), then wait for FUSE visibility on every rank.
    if rank == 0:
        stage_synthetic_shards(args.tokens_dir, args.shards, args.rows_per_shard,
                               args.seq + 1, args.vocab)
        _p(rank, f"[rank0] staged {args.shards} token shards under {args.tokens_dir}")
    dist.barrier()

    ckpt = Checkpointer(args.ckpt_dir, mod)
    model = build_fsdp_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = build_loader(args, mod)

    # --- resume-aware start: newest valid checkpoint -> weights, optimizer, AND each rank's exact
    #     row position. Must happen before the first iter(loader). ---------------------------------
    step, epoch = 0, 0
    latest = ckpt.latest_valid() if args.auto_resume else None
    if latest is not None:
        step, data_json = ckpt.load(model, opt, latest[1])
        states = json.loads(data_json)
        assert len(states) == world, (f"checkpoint has data state for {len(states)} ranks, world is "
                                      f"{world}: resume on the same GPU count (the loader grid is pinned)")
        loader.load_state_dict(states[rank]["loader"])
        epoch = states[rank]["epoch"]
        _p(rank, f"RESUMED_FROM_STEP={step} rank0_epoch={epoch} (valid ckpt {latest[1]})")
    else:
        _p(rank, "COLD_START")
    attempt = [time.strftime("%Y%m%dT%H%M%S") + f"_{os.getpid()}"] if rank == 0 else [None]
    dist.broadcast_object_list(attempt, src=0)
    ledger = Ledger(args.ledger_dir, attempt[0], rank)

    def save():
        ckpt.wait()                                   # prior upload's collectives done first
        path = ckpt.async_save(step, model, opt, gather_data_state(loader, epoch))
        ledger.flush(step)
        _p(rank, f"[rank0] async DCP checkpoint queued step={step} -> {path}")

    # --- training loop: step budget; each rank cycles its own stream. Ranks hold different numbers
    #     of rows, so they do not finish an epoch on the same step — a shared epoch boundary would
    #     hang or drop data. The epoch counter is per rank and rides in the checkpoint. ------------
    it, batches, last_loss, last_saved = iter(loader), 0, float("nan"), step
    seen_this_epoch = set()
    start_step = step
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch, seen_this_epoch, it = epoch + 1, set(), iter(loader)
            batch = next(it)
        loss = lm_loss(model, batch["tokens"], device, args.bf16)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        last_loss = float(loss.detach()); step += 1; batches += 1
        ids = [int(i) for i in batch["row_id"]]
        seen_this_epoch.update(ids)
        ledger.record(step, epoch, ids)
        if step % max(1, args.log_every) == 0:
            _p(rank, f"step={step} loss={last_loss:.4f} epoch={epoch}")
        # step-based cadence is identical on every rank; a time-based one must be agreed with an
        # all-reduce first (the save is a collective).
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save(); last_saved = step
    if last_saved != step or ckpt.saves == 0:
        save()
    ckpt.wait()                                       # drain the final async save
    pre_fp = _fingerprint(model, opt)
    dist.barrier()

    ok = True

    # === CHECK 1 — efficient streaming via the AIR SDK data module. ===============================
    dl_ok = batches > 0 and sdk_path.endswith(".data")
    if dl_ok:
        _p(rank, f"RESILIENT_DATALOAD_OK batches={batches} loader={sdk_path} num_workers={args.num_workers}")
    else:
        ok = False; _p(rank, f"RESILIENT_DATALOAD_FAIL batches={batches} loader={sdk_path}")

    # === CHECK 2 — async DCP checkpoint to the volume, .metadata present. ========================
    last = ckpt.latest_valid()
    ck_ok = ckpt.saves > 0 and last is not None and last[0] == step
    if ck_ok:
        _p(rank, f"RESILIENT_ASYNC_CKPT_OK saves={ckpt.saves} backend=UCVolumeWriter "
                 f"latest_valid_step={last[0]} (.metadata present)")
    else:
        ok = False; _p(rank, f"RESILIENT_ASYNC_CKPT_FAIL saves={ckpt.saves} latest={last}")

    # === CHECK 3 — recover from the most recent VALID checkpoint (bit-identical, resumes at step).
    m2 = build_fsdp_model(args, device)
    o2 = torch.optim.AdamW(m2.parameters(), lr=args.lr)
    r_step, r_data = ckpt.load(m2, o2, last[1]) if last is not None else (None, None)
    resume_fp = _fingerprint(m2, o2) if last is not None else None
    rs_ok = _all(last is not None and resume_fp == pre_fp and r_step == step, device)
    if rs_ok:
        _p(rank, f"RESILIENT_RESUME_OK fingerprint_match=True resumed_step={r_step} valid_ckpt_step={last[0]}")
    else:
        ok = False
        _p(rank, f"RESILIENT_RESUME_FAIL fp_match={resume_fp == pre_fp} loaded_step={r_step} live_step={step}")

    # === CHECK 4a — the data position rides in the checkpoint, per rank. Restore it into a FRESH
    #     loader (as a resubmitted container would) and pull the next batches: none may replay a row
    #     this rank already trained in its current epoch. (4b, no row skipped or repeated across a
    #     real relaunch, is proven from the ledger by the notebook's verify cell.)
    pl_ok, detail = False, "no checkpoint"
    if r_data:
        mine = json.loads(r_data)[rank]
        fresh = build_loader(args, mod)
        fresh.load_state_dict(mine["loader"])
        nxt = []
        for b in fresh:
            nxt += [int(i) for i in b["row_id"]]
            if len(nxt) >= args.batch * max(1, args.num_workers):
                break
        replay = len(set(nxt) & seen_this_epoch)
        pl_ok = mine["epoch"] == epoch and replay == 0
        detail = (f"restored(step={r_step},epoch={mine['epoch']}) next_rows={len(nxt)} "
                  f"replayed={replay} per_worker={ {w: s['rows'] for w, s in mine['loader']['per_worker'].items()} }")
        del fresh
    pl_ok = _all(pl_ok, device)
    if pl_ok:
        _p(rank, f"RESILIENT_PIPELINE_STATE_OK {detail}")
    else:
        ok = False; _p(rank, f"RESILIENT_PIPELINE_STATE_FAIL {detail}")

    if rank == 0 and ok:
        print(f"RESILIENT_SUITE_COMPLETE proofs=1,2,3,4a (dataload+async_ckpt+resume+pipeline_state) "
              f"steps={start_step}->{step}", flush=True)

    dist.barrier()
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    if rank == 0 and not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir", default=None)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--ledger-dir", default=None)
    p.add_argument("--auto-resume", action="store_true", default=True)
    p.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    p.add_argument("--shards", type=int, default=32)
    p.add_argument("--rows-per-shard", type=int, default=256)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2,
                   help="per rank; pinned into the checkpoint — keep it fixed across relaunches")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local", action="store_true",
                   help="CPU/gloo pre-flight with the real AIR SDK against local dirs")
    p.add_argument("--local-world", type=int, default=2)
    p.add_argument("--master-port", type=int, default=29531)
    args = p.parse_args()

    if args.local:
        args.tokens_dir = args.tokens_dir or "/tmp/resilient_tokens"
        args.ckpt_dir = args.ckpt_dir or "/tmp/resilient_ckpt"
        args.ledger_dir = args.ledger_dir or "/tmp/resilient_ledger"
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(args.local_world, args), nprocs=args.local_world, join=True)
        return 0
    assert args.tokens_dir and args.ckpt_dir and args.ledger_dir, \
        "--tokens-dir, --ckpt-dir and --ledger-dir (UC volume paths) required"
    worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

Path(SCRIPT_PATH).write_text(TRAIN_SCRIPT)
print(f"wrote training script ({len(TRAIN_SCRIPT.splitlines())} lines) -> {SCRIPT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Launch — `torchrun --standalone --nproc_per_node=8`
# MAGIC `--standalone` sets up single-node rendezvous on localhost. We launch the **databricks-ai
# MAGIC venv** torchrun (the one that has `torch`) with the **base-python site-packages bridged onto
# MAGIC `PYTHONPATH`** — that's the one combination where `serverless_gpu.data` (base python) and
# MAGIC `torch` (ai venv) are both importable on env v5 (verified 2026-08-27). torchrun propagates the
# MAGIC env to every rank. Output streams below; look for the four `RESILIENT_*_OK` sentinels and
# MAGIC `RESILIENT_SUITE_COMPLETE`. Keep `nproc` and `num_workers` unchanged between this cell and the
# MAGIC relaunch: the saved data position is only valid on the same (GPUs × workers) grid.

# COMMAND ----------

import subprocess

AI_TORCHRUN = dbutils.widgets.get("ai_torchrun")
BASE_SP = dbutils.widgets.get("base_site_packages")

launch_env = os.environ.copy()
launch_env["PYTHONPATH"] = BASE_SP + ":" + launch_env.get("PYTHONPATH", "")

# Use the ai-venv torchrun if present; otherwise fall back to this interpreter's torch.distributed.run.
if os.path.exists(AI_TORCHRUN):
    launcher = [AI_TORCHRUN]
else:
    print(f"⚠️ {AI_TORCHRUN} not found — falling back to `{sys.executable} -m torch.distributed.run` "
          f"(serverless_gpu.data may not import if this interpreter lacks the bridge target)")
    launcher = [sys.executable, "-m", "torch.distributed.run"]

def run_training(extra_args):
    cmd = launcher + [
        "--standalone", f"--nproc_per_node={NPROC}", "--max-restarts=0", SCRIPT_PATH,
        "--tokens-dir", TOKENS_DIR, "--ckpt-dir", CKPT_DIR, "--ledger-dir", LEDGER_DIR,
        "--num-workers", dbutils.widgets.get("num_workers"),
        "--layers", dbutils.widgets.get("layers"), "--dim", dbutils.widgets.get("dim"),
        "--heads", dbutils.widgets.get("heads"), "--seq", dbutils.widgets.get("seq"),
        "--batch", dbutils.widgets.get("batch"), "--steps", dbutils.widgets.get("steps"),
        "--save-every", dbutils.widgets.get("save_every"), "--shards", dbutils.widgets.get("shards"),
    ] + extra_args
    print("Launching:\n ", " ".join(cmd), "\n" + "=" * 90)
    proc = subprocess.run(cmd, env=launch_env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("----- stderr (torchrun / NCCL) -----\n" + proc.stderr[-4000:])
    print("=" * 90, "\nexit code:", proc.returncode)
    return proc

proc = run_training([])
assert proc.returncode == 0, "training did not exit 0 — read the sentinels / stderr above"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prove auto-resume across a container restart
# MAGIC The valid checkpoints from the run above are on the durable UC volume. Relaunching must
# MAGIC **recover from the newest valid checkpoint** and *continue training* rather than start cold —
# MAGIC this simulates what AIR's `max_retries` does when a job's container is preempted and
# MAGIC restarted. We pass **2× the steps** so there's real work left after resuming (otherwise the
# MAGIC run resumes at the finish line with nothing to train). Look for `RESUMED_FROM_STEP=…` (not
# MAGIC `COLD_START`) and the step counter climbing past the checkpoint.
# MAGIC
# MAGIC With the default widgets the resume point is **mid-epoch and mid-shard**: 32 shards × 256 rows
# MAGIC over 8 GPUs × 2 workers = 16 streams of 512 rows, i.e. 128 steps per GPU-epoch at batch 8, so
# MAGIC step 120 sits 8 steps before the epoch ends, 224 rows into each worker's second shard. The
# MAGIC relaunch must continue from exactly that row on every GPU. (For a hard crash *between*
# MAGIC checkpoints, with a real `max_retries` resubmit on multi-node, use
# MAGIC `workloads/mid-epoch-resume.example.yaml`.)

# COMMAND ----------

STEPS = int(dbutils.widgets.get("steps"))
proc2 = run_training(["--steps", str(STEPS * 2)])   # resume at the checkpoint, then train further
assert proc2.returncode == 0, "resumed run did not exit 0 — read the output above"
assert "RESUMED_FROM_STEP" in proc2.stdout, (
    "expected the relaunch to auto-resume from the volume checkpoint but saw no RESUMED_FROM_STEP — "
    "confirm CKPT_DIR is a durable UC volume path (not /tmp) and that checkpoints were written")
print("\n✅ Relaunch auto-resumed from the last valid checkpoint and trained further.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify proof 4b — every row trained exactly once across the relaunch
# MAGIC Each GPU recorded `(step, epoch, row_ids)` for every batch it trained, flushed **with** each
# MAGIC checkpoint, so the ledger is exactly the trajectory a resume continues from. Across both
# MAGIC launches it must show: contiguous steps on every GPU; no row twice within a GPU-epoch; no row on
# MAGIC two GPUs in the same epoch; and every epoch all GPUs finished covers the whole corpus exactly
# MAGIC once. If the relaunch had restarted the epoch (the old behaviour), `no_dupes` fails; if it had
# MAGIC skipped ahead too far, `exact_cover` fails.

# COMMAND ----------

import json
from collections import defaultdict

import pyarrow.parquet as pq


def verify_ledger(ledger_dir, tokens_dir):
    by_rank = defaultdict(dict)             # rank -> step -> (attempt, epoch, ids); later attempt wins
    attempts = sorted(d for d in os.listdir(ledger_dir) if d.startswith("attempt_"))
    for a in attempts:
        for name in sorted(os.listdir(os.path.join(ledger_dir, a))):
            rank = int(name.split("_")[1])
            with open(os.path.join(ledger_dir, a, name)) as f:
                for line in f:
                    r = json.loads(line)
                    by_rank[rank][r["step"]] = (a, r["epoch"], r["ids"])
    corpus = set()
    for name in sorted(os.listdir(tokens_dir)):
        if name.endswith(".parquet"):
            corpus.update(pq.read_table(os.path.join(tokens_dir, name), columns=["row_id"])
                          .column("row_id").to_pylist())
    ranks, ok = sorted(by_rank), True
    maxes = {r: max(by_rank[r]) for r in ranks}
    contiguous = len(set(maxes.values())) == 1 and all(
        set(by_rank[r]) == set(range(1, maxes[r] + 1)) for r in ranks)
    per, used = defaultdict(list), set()
    for r in ranks:
        for st in sorted(by_rank[r]):
            a, ep, ids = by_rank[r][st]
            per[(r, ep)].extend(ids)
            used.add(a)
    dupes = sum(len(v) - len(set(v)) for v in per.values())
    last_ep = {r: max(ep for rr, ep in per if rr == r) for r in ranks}
    overlaps, covered = 0, {}
    for ep in sorted({ep for _, ep in per}):
        seen = set()
        for r in ranks:
            s_ = set(per.get((r, ep), []))
            overlaps += len(seen & s_)
            seen |= s_
        if all(last_ep[r] > ep for r in ranks):
            covered[ep] = seen == corpus
    checks = {"contiguous": contiguous, "no_dupes": dupes == 0, "disjoint": overlaps == 0,
              "exact_cover": bool(covered) and all(covered.values()), "resumed": len(used) >= 2}
    for k, v in checks.items():
        print(f"LEDGER {k}: {'OK' if v else 'FAIL'}")
    print(f"gpus={len(ranks)} steps={sorted(set(maxes.values()))} attempts={len(used)} "
          f"corpus_rows={len(corpus)} complete_epochs={sorted(covered)} dupes={dupes} overlaps={overlaps}")
    if all(checks.values()):
        print("RESILIENT_PIPELINE_OK exactly-once across the relaunch (no repeats, no skips)")
    return all(checks.values())


assert verify_ledger(LEDGER_DIR, TOKENS_DIR), "ledger verification failed — see the LEDGER lines above"

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this demonstrated (mapped to the doc)
# MAGIC - **1. Load data efficiently** — `RESILIENT_DATALOAD_OK`: shards streamed through
# MAGIC   `serverless_gpu.data.UCVolumeDataset` + `DataLoader`, auto-sharded across ranks×workers with
# MAGIC   no `DistributedSampler` and served from the local cache.
# MAGIC - **2. Checkpoint with DCP + save asynchronously** — `RESILIENT_ASYNC_CKPT_OK`: `dcp.async_save`
# MAGIC   via `UCVolumeWriter`, prior save awaited before the next, `.metadata` written.
# MAGIC - **3. Recover from the most recent valid checkpoint** — `RESILIENT_RESUME_OK` +
# MAGIC   `RESUMED_FROM_STEP`: `find_latest_valid` validates `.metadata`, loads via `UCVolumeReader` +
# MAGIC   `set_state_dict`, params/optimizer bit-identical, training continues at the saved step.
# MAGIC - **4. Checkpoint the data pipeline** — `RESILIENT_PIPELINE_STATE_OK` + `RESILIENT_PIPELINE_OK`:
# MAGIC   each GPU's exact (shard, row) position per DataLoader worker rides in the same DCP save as the
# MAGIC   weights; the relaunch continued from that row and the ledger shows every row of every
# MAGIC   completed epoch trained exactly once. Order across workers after a resume can differ from an
# MAGIC   uninterrupted run (torch's worker round-robin restarts); the *set* per epoch cannot.
# MAGIC
# MAGIC **Pre-flight tip:** the script runs on CPU without AIR, against the **real AIR SDK** code —
# MAGIC `PYTHONPATH=<dir containing databricks/air or serverless_gpu> python /tmp/resilient_train.py
# MAGIC --local --local-world 2 --steps 40` (then rerun with `--steps 80` to resume); `--local` lifts
# MAGIC only the SDK's `/Volumes` path check so local dirs work.
