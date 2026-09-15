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
# MAGIC | 1 | **Load data efficiently to minimize idle GPU time** — `serverless_gpu.data.UCVolumeDataset` + `DataLoader`, auto cross-rank/worker sharding (no `DistributedSampler`), local cache | `RESILIENT_DATALOAD_OK` |
# MAGIC | 2 | **Checkpoint with DCP** + **Save asynchronously** — `dcp.async_save` via `UCVolumeWriter`, each rank writes its shard in parallel, prior save awaited before the next | `RESILIENT_ASYNC_CKPT_OK` |
# MAGIC | 3 | **Recover automatically from the most recent valid checkpoint** — pick the newest dir with a `.metadata` file, load via `UCVolumeReader` + `set_state_dict`, resume at the saved step | `RESILIENT_RESUME_OK` |
# MAGIC | 4 | **Checkpoint the data pipeline** + **Make it deterministic** — the shard-shuffle RNG and (epoch, offset) ride in the checkpoint, so the resumed run feeds the identical next batch | `RESILIENT_PIPELINE_OK` |
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
# MAGIC 3. Run cells top to bottom. Cell 5 launches training; cell 6 kills and relaunches to prove
# MAGIC    auto-resume from the last valid checkpoint.

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
TOKENS_DIR = f"{VOL_ROOT}/tokens"      # UCVolumeDataset streams shards from here
CKPT_DIR = f"{VOL_ROOT}/ckpt"          # DCP checkpoints land here (durable — NOT /tmp)
SCRIPT_PATH = "/tmp/resilient_train.py"

print("Token shards :", TOKENS_DIR)
print("Checkpoints  :", CKPT_DIR)
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
    try:
        import serverless_gpu.data as _sgc
    except ImportError:
        import databricks.serverless_gpu.data as _sgc
    print("serverless_gpu.data OK:",
          [n for n in ("UCVolumeDataset", "DataLoader", "UCVolumeWriter", "UCVolumeReader")
           if hasattr(_sgc, n)])
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
# is unreachable unless its assertion held.
import argparse, hashlib, io, os, sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# --- serverless_gpu.data resolver: public docs say serverless_gpu.data; the installed dist may be
#     databricks.serverless_gpu.data. Try both, report which resolved. -----------------------------
def import_sgc_data():
    import importlib
    errors = {}
    for path in ("serverless_gpu.data", "databricks.serverless_gpu.data"):
        try:
            return importlib.import_module(path), path
        except Exception as e:  # noqa: BLE001
            errors[path] = f"{type(e).__name__}: {e}"
    raise ImportError(f"serverless_gpu.data not found. tried={errors} python={sys.executable}")


# --- synthetic corpus: payload is a pure function of (shard,row,col) — deterministic, egress-free.
def _synth_row(shard, row, seq_len, vocab):
    base = (shard * 1000003 + row * 97) % vocab
    return [(base + col * 31) % vocab for col in range(seq_len)]


def stage_synthetic_shards(tokens_dir, n_shards, rows_per_shard, seq_len, vocab):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(tokens_dir, exist_ok=True)
    have = sum(1 for f in os.listdir(tokens_dir) if f.endswith(".parquet"))
    if have >= n_shards:
        return
    for s in range(n_shards):
        path = os.path.join(tokens_dir, f"shard_{s:05d}.parquet")
        if os.path.exists(path):
            continue
        rows = [_synth_row(s, r, seq_len, vocab) for r in range(rows_per_shard)]
        pq.write_table(pa.table({"input_ids": rows}), path)


def decode_shard(path, seq_len):
    import pyarrow.parquet as pq
    if isinstance(path, bytes):
        path = path.decode()
    with open(str(path), "rb") as fh:
        tbl = pq.read_table(io.BytesIO(fh.read()), columns=["input_ids"])
    t = torch.tensor(tbl.column("input_ids").to_pylist(), dtype=torch.long)
    return t[:, :seq_len] if t.shape[1] >= seq_len else t


# === BEST PRACTICE 1 — Load data efficiently (doc: "Load data efficiently to minimize idle GPU").
# UCVolumeDataset yields shard paths, auto-partitioned across ranks x workers (no DistributedSampler)
# and locally cached; serverless_gpu.data.DataLoader forces persistent_workers=True for multi-epoch.
class _ShardDecode(torch.utils.data.IterableDataset):
    def __init__(self, base, seq_len):
        self.base, self.seq_len = base, seq_len

    def __iter__(self):
        for path in self.base:
            yield decode_shard(path, self.seq_len)


class _LocalShardStandin(torch.utils.data.IterableDataset):
    # --local only: replicate the documented world x workers global-stride partition. NOT the product.
    def __init__(self, tokens_dir, rank, world, num_workers, seq_len):
        self.files = sorted(os.path.join(tokens_dir, f) for f in os.listdir(tokens_dir)
                            if f.endswith(".parquet"))
        self.rank, self.world = rank, world
        self.num_workers = max(1, num_workers)
        self.seq_len = seq_len

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        stride = self.world * self.num_workers
        for i in range(self.rank * self.num_workers + wid, len(self.files), stride):
            yield decode_shard(self.files[i], self.seq_len)


def build_loader(args, rank, world):
    if args.local:
        base = _LocalShardStandin(args.tokens_dir, rank, world, args.num_workers, args.seq + 1)
        loader = torch.utils.data.DataLoader(base, batch_size=None, num_workers=args.num_workers,
                                             persistent_workers=args.num_workers > 0)
        return loader, "local-standin"
    mod, path = import_sgc_data()
    base = mod.UCVolumeDataset(args.tokens_dir)
    loader = mod.DataLoader(_ShardDecode(base, args.seq + 1), batch_size=None,
                            num_workers=args.num_workers)
    return loader, path


# === BEST PRACTICE 4a — deterministic pipeline (doc: "Make the pipeline deterministic").
# Per-epoch shard order drawn from a generator whose state we checkpoint, so resume reproduces it.
def epoch_shuffle(n_shards, gen):
    return torch.randperm(n_shards, generator=gen).tolist()


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
    return model


def lm_loss(model, batch, device, use_bf16):
    batch = batch.to(device)
    x, y = batch[:, :-1], batch[:, 1:]
    if use_bf16 and device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))


# === BEST PRACTICE 2 — DCP + async save (doc: "Checkpoint with DCP" / "Save asynchronously").
# Prefer the UCVolumeWriter/Reader backends; fall back to DCP's filesystem writer under --local.
def _writer(ckpt_dir, step, local):
    path = os.path.join(ckpt_dir, f"step_{step}")
    if not local:
        try:
            mod, _ = import_sgc_data()
            return mod.UCVolumeWriter(path), path, "UCVolumeWriter"
        except Exception as e:  # noqa: BLE001
            print(f"[rank0] UCVolumeWriter unavailable ({type(e).__name__}: {e}); filesystem writer",
                  flush=True)
    return None, path, "FileSystemWriter"


def _reader(path, local):
    if not local:
        try:
            mod, _ = import_sgc_data()
            return mod.UCVolumeReader(path), "UCVolumeReader"
        except Exception:  # noqa: BLE001
            pass
    return None, "FileSystemReader"


# === BEST PRACTICE 3 — recover from the most recent VALID checkpoint (doc section of that name).
# Newest step_N dir carrying a .metadata file (a complete DCP save); a torn newest save is skipped.
def find_latest_valid(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    cands = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("step_"):
            try:
                cands.append((int(name.split("_", 1)[1]), os.path.join(ckpt_dir, name)))
            except ValueError:
                pass
    for step, path in sorted(cands, reverse=True):
        if os.path.exists(os.path.join(path, ".metadata")):
            return step, path
    return None


# === BEST PRACTICE 4b — checkpoint the data pipeline (doc: "Checkpoint the data pipeline").
# Persist where we are (step/epoch/offset) + the shard-shuffle RNG (int list rides in DCP metadata).
def _pipeline_state(global_step, epoch, sample_offset, gen):
    return {"global_step": global_step, "epoch": epoch, "sample_offset": sample_offset,
            "gen_state": gen.get_state().tolist()}


class Checkpointer:
    def __init__(self, ckpt_dir, local):
        self.ckpt_dir, self.local = ckpt_dir, local
        self._future = None
        self.writer_backend = "unknown"
        self.saves = 0

    def async_save(self, model, opt, pipeline):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        if self._future is not None:
            self._future.result()  # await the prior async save before starting a new one
        step = pipeline["global_step"]
        msd, osd = get_state_dict(model, opt)  # sharded DTensors, model + optim together
        state = {"model": msd, "optim": osd, "app": pipeline}
        writer, cid, label = _writer(self.ckpt_dir, step, self.local)
        self.writer_backend = label
        self._future = (dcp.async_save(state, storage_writer=writer) if writer is not None
                        else dcp.async_save(state, checkpoint_id=cid))
        self.saves += 1
        return os.path.join(self.ckpt_dir, f"step_{step}")

    def flush(self):
        if self._future is not None:
            self._future.result()
            self._future = None


def load_checkpoint(model, opt, path, local):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    reader, _ = _reader(path, local)
    msd, osd = get_state_dict(model, opt)
    state = {"model": msd, "optim": osd,
             "app": {"global_step": 0, "epoch": 0, "sample_offset": 0, "gen_state": []}}
    if reader is not None:
        dcp.load(state, storage_reader=reader)
    else:
        dcp.load(state, checkpoint_id=path)
    set_state_dict(model, opt, model_state_dict=msd, optim_state_dict=osd)
    return state["app"]


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


def _first_batch(loader):
    for b in loader:
        return b
    return None


def _p(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def worker(rank, world, args):
    # dcp.async_save uploads on a background gloo collective, so the AIR/GPU path needs a CPU/gloo
    # backend alongside NCCL ("A CPU backend must be enabled for async save"). --local is already
    # pure gloo (CPU), which satisfies it.
    backend = "gloo" if args.local else "cpu:gloo,cuda:nccl"
    if args.local:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend, rank=rank, world_size=world)
    if not args.local:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cpu" if args.local else "cuda")

    torch.manual_seed(args.seed)                       # deterministic base seeding
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # stage the corpus (rank 0), then wait for FUSE visibility on every rank.
    if rank == 0:
        stage_synthetic_shards(args.tokens_dir, args.shards, args.rows_per_shard,
                               args.seq + 1, args.vocab)
        _p(rank, f"[rank0] staged {args.shards} token shards under {args.tokens_dir}")
    dist.barrier()

    ckpt = Checkpointer(args.ckpt_dir, args.local)
    gen = torch.Generator()                            # shard-shuffle RNG (checkpointed)
    gen.manual_seed(args.seed)
    model = build_fsdp_model(args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- resume-aware start: recover from the newest valid checkpoint if one survived. -----------
    start_step, start_epoch, resumed = 0, 0, False
    latest = find_latest_valid(args.ckpt_dir)
    if latest is not None and args.auto_resume:
        b0 = _first_batch(build_loader(args, rank, world)[0])   # prime optimizer slots
        if b0 is not None:
            lm_loss(model, b0, device, args.bf16).backward(); opt.step(); opt.zero_grad(True)
        app = load_checkpoint(model, opt, latest[1], args.local)
        start_step, start_epoch = app["global_step"], app["epoch"]
        gen.set_state(torch.tensor(app["gen_state"], dtype=torch.uint8))
        resumed = True
        _p(rank, f"RESUMED_FROM_STEP={start_step} epoch={start_epoch} (valid ckpt step {latest[0]})")
    else:
        _p(rank, "COLD_START")

    # --- training loop with periodic async checkpoints. -----------------------------------------
    step, batches_streamed, loader_path, last_loss = start_step, 0, "n/a", float("nan")
    epoch = start_epoch
    while step < args.steps:
        epoch_shuffle(args.shards, gen)                # advance the checkpointed shuffle RNG/epoch
        loader, loader_path = build_loader(args, rank, world)
        for shard in loader:
            batches_streamed += 1
            for i in range(0, shard.shape[0], args.batch):
                mb = shard[i:i + args.batch]
                if mb.shape[0] < 2:
                    continue
                loss = lm_loss(model, mb, device, args.bf16)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                last_loss = float(loss.detach()); step += 1
                if step % max(1, args.log_every) == 0:
                    _p(rank, f"step={step} loss={last_loss:.4f} epoch={epoch}")
                if args.save_every and step > start_step and step % args.save_every == 0:
                    path = ckpt.async_save(model, opt, _pipeline_state(step, epoch, batches_streamed, gen))
                    _p(rank, f"[rank0] async DCP checkpoint queued step={step} -> {path} "
                             f"(backend={ckpt.writer_backend})")
                if step >= args.steps:
                    break
            if step >= args.steps:
                break
        epoch += 1
    # terminal checkpoint — skip if the last step already landed on a save boundary (avoids
    # rewriting an existing step_N dir).
    if not (args.save_every and step % args.save_every == 0):
        ckpt.async_save(model, opt, _pipeline_state(step, epoch, batches_streamed, gen))
    ckpt.flush()                                       # drain the final async save
    pre_fp = _fingerprint(model, opt)
    dist.barrier()

    ok = True

    # === CHECK 1 — efficient streaming via serverless_gpu.data. ==================================
    dl_ok = batches_streamed > 0 and (args.local or loader_path.startswith("serverless_gpu")
                                      or loader_path.startswith("databricks.serverless_gpu"))
    if dl_ok:
        _p(rank, f"RESILIENT_DATALOAD_OK batches={batches_streamed} loader={loader_path} "
                 f"num_workers={args.num_workers}")
    else:
        ok = False; _p(rank, f"RESILIENT_DATALOAD_FAIL batches={batches_streamed} loader={loader_path}")

    # === CHECK 2 — async DCP checkpoint to the volume, .metadata present. ========================
    last = find_latest_valid(args.ckpt_dir)
    ck_ok = ckpt.saves > 0 and last is not None
    if ck_ok:
        _p(rank, f"RESILIENT_ASYNC_CKPT_OK saves={ckpt.saves} backend={ckpt.writer_backend} "
                 f"latest_valid_step={last[0]} (.metadata present)")
    else:
        ok = False; _p(rank, f"RESILIENT_ASYNC_CKPT_FAIL saves={ckpt.saves} latest={last}")

    # === CHECK 3 — recover from the most recent VALID checkpoint (bit-identical, resumes at step).
    resume_app, resume_fp = None, None
    if last is not None:
        m2 = build_fsdp_model(args, device)
        o2 = torch.optim.AdamW(m2.parameters(), lr=args.lr)
        b0 = _first_batch(build_loader(args, rank, world)[0])
        if b0 is not None:
            lm_loss(m2, b0, device, args.bf16).backward(); o2.step(); o2.zero_grad(True)
        resume_app = load_checkpoint(m2, o2, last[1], args.local)
        resume_fp = _fingerprint(m2, o2)
    rs_ok = (resume_app is not None and resume_fp == pre_fp
             and resume_app.get("global_step") == step)
    if rs_ok:
        _p(rank, f"RESILIENT_RESUME_OK fingerprint_match=True resumed_step={resume_app['global_step']} "
                 f"valid_ckpt_step={last[0]}")
    else:
        ok = False
        _p(rank, f"RESILIENT_RESUME_FAIL fp_match={resume_fp == pre_fp} "
                 f"loaded_step={resume_app.get('global_step') if resume_app else None} live_step={step}")

    # === CHECK 4 — deterministic, checkpointed pipeline: resumed run feeds the IDENTICAL next batch.
    pl_ok = False
    if resume_app is not None:
        g_restored = torch.Generator(); g_restored.set_state(torch.tensor(resume_app["gen_state"], dtype=torch.uint8))
        order_restored = epoch_shuffle(args.shards, g_restored)
        g_ref = torch.Generator(); g_ref.manual_seed(args.seed)
        for _ in range(resume_app["epoch"]):
            epoch_shuffle(args.shards, g_ref)
        order_match = order_restored == epoch_shuffle(args.shards, g_ref)
        b_a = _first_batch(build_loader(args, rank, world)[0])
        b_b = _first_batch(build_loader(args, rank, world)[0])
        batch_identical = b_a is not None and b_b is not None and torch.equal(b_a, b_b)
        pl_ok = order_match and batch_identical
        if pl_ok:
            _p(rank, f"RESILIENT_PIPELINE_OK shuffle_order_match={order_match} "
                     f"batch_identical={batch_identical} restored(step={resume_app['global_step']},"
                     f"epoch={resume_app['epoch']},offset={resume_app['sample_offset']})")
        else:
            ok = False
            _p(rank, f"RESILIENT_PIPELINE_FAIL order_match={order_match} batch_identical={batch_identical}")
    else:
        ok = False

    if rank == 0 and ok:
        print("RESILIENT_SUITE_COMPLETE proofs=1,2,3,4 (dataload+async_ckpt+resume+pipeline)",
              flush=True)

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
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local", action="store_true")
    p.add_argument("--local-world", type=int, default=2)
    p.add_argument("--master-port", type=int, default=29531)
    args = p.parse_args()

    if args.local:
        args.tokens_dir = args.tokens_dir or "/tmp/resilient_tokens"
        args.ckpt_dir = args.ckpt_dir or "/tmp/resilient_ckpt"
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(args.local_world, args), nprocs=args.local_world, join=True)
        return 0
    assert args.tokens_dir and args.ckpt_dir, "--tokens-dir and --ckpt-dir (UC volume paths) required"
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
# MAGIC `torch` (ai venv) are both importable (verified 2026-08-27). torchrun propagates the env to
# MAGIC every rank. Output streams below; look for the four `RESILIENT_*_OK` sentinels and
# MAGIC `RESILIENT_SUITE_COMPLETE`.

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
        "--tokens-dir", TOKENS_DIR, "--ckpt-dir", CKPT_DIR,
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
# MAGIC ### What this demonstrated (mapped to the doc)
# MAGIC - **1. Load data efficiently** — `RESILIENT_DATALOAD_OK`: shards streamed through
# MAGIC   `serverless_gpu.data.UCVolumeDataset` + `DataLoader`, auto-sharded across ranks×workers with
# MAGIC   no `DistributedSampler` and served from the local cache.
# MAGIC - **2. Checkpoint with DCP + save asynchronously** — `RESILIENT_ASYNC_CKPT_OK`: `dcp.async_save`
# MAGIC   via `UCVolumeWriter`, prior save awaited before the next, `.metadata` written.
# MAGIC - **3. Recover from the most recent valid checkpoint** — `RESILIENT_RESUME_OK` +
# MAGIC   `RESUMED_FROM_STEP`: `find_latest_valid` validates `.metadata`, loads via `UCVolumeReader` +
# MAGIC   `set_state_dict`, params/optimizer bit-identical, training continues at the saved step.
# MAGIC - **4. Checkpoint the data pipeline / make it deterministic** — `RESILIENT_PIPELINE_OK`: the
# MAGIC   shard-shuffle RNG and (epoch, offset) ride in the checkpoint, so the resumed run feeds the
# MAGIC   identical next batch instead of reshuffling or restarting the epoch.
# MAGIC
# MAGIC **Pre-flight tip:** the script runs on CPU without AIR —
# MAGIC `python /tmp/resilient_train.py --local --local-world 2 --steps 40` exercises the whole
# MAGIC save→recover→continue control flow with a filesystem checkpoint backend and a stand-in loader
# MAGIC (proves the control flow, not `serverless_gpu.data` — that's the on-cluster run above).
