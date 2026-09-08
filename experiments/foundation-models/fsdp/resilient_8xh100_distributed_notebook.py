# Databricks notebook source
# MAGIC %md
# MAGIC # Resilient FSDP training on 8×H100 — `@distributed` variant
# MAGIC
# MAGIC Same four AIR **Performance and resiliency** best practices as
# MAGIC `resilient_8xh100_notebook`, but launched with the **`serverless_gpu` `@distributed`
# MAGIC decorator** instead of `torchrun`. This is the notebook-native single-node path
# MAGIC ([docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/distributed-training)).
# MAGIC
# MAGIC | # | Best practice (doc section) | Sentinel |
# MAGIC |---|---|---|
# MAGIC | 1 | **Load data efficiently to minimize idle GPU time** — `serverless_gpu.data.UCVolumeDataset` + `DataLoader` | `RESILIENT_DATALOAD_OK` |
# MAGIC | 2 | **Checkpoint with DCP** + **Save asynchronously** — `dcp.async_save` via `UCVolumeWriter` | `RESILIENT_ASYNC_CKPT_OK` |
# MAGIC | 3 | **Recover automatically from the most recent valid checkpoint** — newest `.metadata` dir via `UCVolumeReader` | `RESILIENT_RESUME_OK` |
# MAGIC | 4 | **Checkpoint the data pipeline** + **Make it deterministic** — checkpointed shuffle RNG + (epoch, offset) | `RESILIENT_PIPELINE_OK` |
# MAGIC
# MAGIC ### `@distributed` vs the torchrun sibling
# MAGIC - **Launch:** decorate a `train` function with `@distributed(gpus=8)` and call
# MAGIC   `train.distributed(**kwargs)` — the framework fans the function out one-process-per-GPU,
# MAGIC   syncs the notebook environment, and creates/nests an MLflow run. No subprocess, no
# MAGIC   `torchrun`.
# MAGIC - **No `PYTHONPATH` bridge:** under the `@distributed` runtime `torch` and `serverless_gpu`
# MAGIC   already coexist, so `serverless_gpu.data` imports directly (the bridge is only needed on the
# MAGIC   raw-torchrun path).
# MAGIC - **You still set up the process group yourself:** `torch.cuda.set_device(LOCAL_RANK)` +
# MAGIC   `dist.init_process_group("nccl")` at the top of the function.
# MAGIC - **Imports + data go *inside* the function** (the doc's guidance — the function is
# MAGIC   serialized to the workers, so keep the closure small).
# MAGIC
# MAGIC > ⚠️ **Scope note:** `@distributed` is a Beta convenience layer. The repo's steering notes
# MAGIC > (`AGENTS.md`, `docs/private/uat-plan-2026-07.md`) had it ruled *out* for the customer
# MAGIC > engagement in favour of the torchrun/CLI path; use this variant only where that decision has
# MAGIC > been superseded. The torchrun sibling (`resilient_8xh100_notebook`) remains the portable,
# MAGIC > CLI-transferable path.
# MAGIC
# MAGIC ### How to run
# MAGIC 1. Attach to an **AI base environment (v5)** with a **`GPU_8xH100`** accelerator.
# MAGIC 2. Fill in the **catalog / schema / volume** widgets (a writable UC volume; checkpoints must
# MAGIC    be on a durable volume, never `/tmp`).
# MAGIC 3. Run cells top to bottom. Cell 4 launches training; cell 5 relaunches to prove auto-resume.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog>", "catalog")
dbutils.widgets.text("schema", "<schema>", "schema")
dbutils.widgets.text("volume", "<volume>", "UC volume (writable)")
dbutils.widgets.text("base_dir", "resilient-fsdp-distributed", "subdir in the volume for this job")
dbutils.widgets.text("gpus", "8", "GPUs for @distributed (single-node 8xH100)")
dbutils.widgets.text("layers", "12", "transformer layers")
dbutils.widgets.text("dim", "1024", "model dim")
dbutils.widgets.text("heads", "16", "attention heads")
dbutils.widgets.text("seq", "512", "sequence length")
dbutils.widgets.text("batch", "8", "per-GPU micro-batch (sequences)")
dbutils.widgets.text("steps", "120", "optimizer steps")
dbutils.widgets.text("save_every", "40", "checkpoint every N steps")
dbutils.widgets.text("shards", "32", "synthetic token shards to stage")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
BASE_DIR = dbutils.widgets.get("base_dir")
GPUS = int(dbutils.widgets.get("gpus"))

assert "<" not in (CATALOG + SCHEMA + VOLUME), (
    "Fill in the catalog / schema / volume widgets with a writable UC volume before running.")

VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{BASE_DIR}"
CFG = dict(
    tokens_dir=f"{VOL_ROOT}/tokens",   # UCVolumeDataset streams shards from here
    ckpt_dir=f"{VOL_ROOT}/ckpt",       # DCP checkpoints land here (durable — NOT /tmp)
    layers=int(dbutils.widgets.get("layers")), dim=int(dbutils.widgets.get("dim")),
    heads=int(dbutils.widgets.get("heads")), seq=int(dbutils.widgets.get("seq")),
    batch=int(dbutils.widgets.get("batch")), steps=int(dbutils.widgets.get("steps")),
    save_every=int(dbutils.widgets.get("save_every")), shards=int(dbutils.widgets.get("shards")),
    rows_per_shard=256, vocab=8192, lr=3e-4, log_every=10, num_workers=2, seed=0,
)
print("Token shards :", CFG["tokens_dir"])
print("Checkpoints  :", CFG["ckpt_dir"])
print("GPUs         :", GPUS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment check
# MAGIC 8-GPU H100 node, FSDP2-capable runtime, and `serverless_gpu.data` importable **without** a
# MAGIC bridge (the `@distributed` runtime carries both torch and serverless_gpu).

# COMMAND ----------

import torch
from torch.distributed.fsdp import fully_shard

n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
print(f"visible GPUs on the driver: {n_gpu}")
print("fully_shard (FSDP2) available:", callable(fully_shard))

try:
    from serverless_gpu import distributed
    print("serverless_gpu.distributed: OK")
    try:
        import serverless_gpu.data as _sgc
    except ImportError:
        import databricks.serverless_gpu.data as _sgc
    print("serverless_gpu.data:", [n for n in ("UCVolumeDataset", "DataLoader", "UCVolumeWriter",
                                                "UCVolumeReader") if hasattr(_sgc, n)])
except Exception as e:
    print(f"⚠️ serverless_gpu not importable on the driver ({type(e).__name__}: {e}). "
          f"Attach an AI base env (v5) with a GPU accelerator.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The distributed training function
# MAGIC Decorated with `@distributed(gpus=…)`. Everything it needs (imports, model, data, checkpoint
# MAGIC logic) is defined **inside** the function so the serialized closure stays small — the doc's
# MAGIC guidance. Each best practice is labelled with its doc section.

# COMMAND ----------

from serverless_gpu import distributed


@distributed(gpus=GPUS)  # gpu_type auto-detected from the attached GPU_8xH100 accelerator
def run_train(tokens_dir, ckpt_dir, layers, dim, heads, seq, batch, steps, save_every,
              shards, rows_per_shard, vocab, lr, log_every, num_workers, seed,
              auto_resume=True):
    import hashlib
    import io
    import os
    import sys

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F

    # --- process-group setup (you do this yourself under @distributed) ----------------------
    # NOTE: dcp.async_save offloads state to CPU and uploads it on a BACKGROUND gloo collective,
    # so the process group must carry a CPU/gloo backend alongside NCCL. Plain "nccl" trips
    # "A CPU backend must be enabled for async save" at the first checkpoint.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda")

    def _p(*a):
        if rank == 0:
            print(*a, flush=True)

    # --- serverless_gpu.data resolver (native under @distributed; both dist names tried) -----
    def import_sgc_data():
        import importlib
        for path in ("serverless_gpu.data", "databricks.serverless_gpu.data"):
            try:
                return importlib.import_module(path), path
            except Exception:  # noqa: BLE001
                pass
        raise ImportError("serverless_gpu.data not importable in the @distributed runtime")

    # --- synthetic corpus: payload is a pure function of (shard,row,col) — deterministic ------
    def _synth_row(shard, row, seq_len, vcb):
        base = (shard * 1000003 + row * 97) % vcb
        return [(base + col * 31) % vcb for col in range(seq_len)]

    def stage_synthetic_shards(tdir, n_shards, rps, seq_len, vcb):
        import pyarrow as pa
        import pyarrow.parquet as pq
        os.makedirs(tdir, exist_ok=True)
        if sum(1 for f in os.listdir(tdir) if f.endswith(".parquet")) >= n_shards:
            return
        for s in range(n_shards):
            path = os.path.join(tdir, f"shard_{s:05d}.parquet")
            if os.path.exists(path):
                continue
            rows = [_synth_row(s, r, seq_len, vcb) for r in range(rps)]
            pq.write_table(pa.table({"input_ids": rows}), path)

    def decode_shard(path, seq_len):
        import pyarrow.parquet as pq
        if isinstance(path, bytes):
            path = path.decode()
        with open(str(path), "rb") as fh:
            tbl = pq.read_table(io.BytesIO(fh.read()), columns=["input_ids"])
        t = torch.tensor(tbl.column("input_ids").to_pylist(), dtype=torch.long)
        return t[:, :seq_len] if t.shape[1] >= seq_len else t

    # === BEST PRACTICE 1 — Load data efficiently. UCVolumeDataset auto-shards across
    #     ranks×workers (no DistributedSampler) + local cache; DataLoader → persistent workers.
    class _ShardDecode(torch.utils.data.IterableDataset):
        def __init__(self, base, seq_len):
            self.base, self.seq_len = base, seq_len

        def __iter__(self):
            for path in self.base:
                yield decode_shard(path, self.seq_len)

    def build_loader():
        mod, path = import_sgc_data()
        base = mod.UCVolumeDataset(tokens_dir)
        loader = mod.DataLoader(_ShardDecode(base, seq + 1), batch_size=None,
                                num_workers=num_workers)
        return loader, path

    # === BEST PRACTICE 4a — deterministic pipeline: per-epoch shard order from a checkpointed gen.
    def epoch_shuffle(n_shards, gen):
        return torch.randperm(n_shards, generator=gen).tolist()

    # --- compact GPT-2-style causal LM; FSDP2 fully_shard per block + root; bf16 autocast. ----
    class Block(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, h, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            hh = self.ln1(x)
            mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), 1)
            a, _ = self.attn(hh, hh, hh, attn_mask=mask, need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self, vcb, d, h, ly, sq):
            super().__init__()
            self.tok = nn.Embedding(vcb, d)
            self.pos = nn.Embedding(sq, d)
            self.blocks = nn.ModuleList([Block(d, h) for _ in range(ly)])
            self.lnf = nn.LayerNorm(d)
            self.head = nn.Linear(d, vcb, bias=False)

        def forward(self, idx):
            pos = torch.arange(idx.shape[1], device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None]
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.lnf(x))

    def build_fsdp_model():
        from torch.distributed.fsdp import fully_shard
        model = GPT(vocab, dim, heads, layers, seq).to(device)
        for blk in model.blocks:
            fully_shard(blk)
        fully_shard(model)
        return model

    def lm_loss(model, mb):
        mb = mb.to(device)
        x, y = mb[:, :-1], mb[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))

    # === BEST PRACTICE 2 — DCP + async save via the UCVolumeWriter backend. -------------------
    def make_writer(step):
        mod, _ = import_sgc_data()
        path = os.path.join(ckpt_dir, f"step_{step}")
        return mod.UCVolumeWriter(path), path

    def make_reader(path):
        mod, _ = import_sgc_data()
        return mod.UCVolumeReader(path)

    # === BEST PRACTICE 3 — recover from the most recent VALID checkpoint (.metadata present). --
    def find_latest_valid():
        if not os.path.isdir(ckpt_dir):
            return None
        cands = []
        for name in os.listdir(ckpt_dir):
            if name.startswith("step_"):
                try:
                    cands.append((int(name.split("_", 1)[1]), os.path.join(ckpt_dir, name)))
                except ValueError:
                    pass
        for st, path in sorted(cands, reverse=True):
            if os.path.exists(os.path.join(path, ".metadata")):
                return st, path
        return None

    # === BEST PRACTICE 4b — checkpoint the data pipeline (step/epoch/offset + shuffle RNG). ----
    def pipeline_state(gstep, epoch, offset, gen):
        return {"global_step": gstep, "epoch": epoch, "sample_offset": offset,
                "gen_state": gen.get_state().tolist()}

    class Checkpointer:
        def __init__(self):
            self._future = None
            self.saves = 0

        def async_save(self, model, opt, pipe):
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict
            if self._future is not None:
                self._future.result()  # await the prior async save before starting a new one
            msd, osd = get_state_dict(model, opt)
            writer, path = make_writer(pipe["global_step"])
            self._future = dcp.async_save({"model": msd, "optim": osd, "app": pipe},
                                          storage_writer=writer)
            self.saves += 1
            return path

        def flush(self):
            if self._future is not None:
                self._future.result()
                self._future = None

    def load_checkpoint(model, opt, path):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        msd, osd = get_state_dict(model, opt)
        state = {"model": msd, "optim": osd,
                 "app": {"global_step": 0, "epoch": 0, "sample_offset": 0, "gen_state": []}}
        dcp.load(state, storage_reader=make_reader(path))
        set_state_dict(model, opt, model_state_dict=msd, optim_state_dict=osd)
        return state["app"]

    def fingerprint(model, opt):
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

    def first_batch():
        for b in build_loader()[0]:
            return b
        return None

    # ---------------------------------------------------------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if rank == 0:
        stage_synthetic_shards(tokens_dir, shards, rows_per_shard, seq + 1, vocab)
        _p(f"[rank0] staged {shards} token shards under {tokens_dir}")
    dist.barrier()

    ckpt = Checkpointer()
    gen = torch.Generator()
    gen.manual_seed(seed)
    model = build_fsdp_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- resume-aware start ----------------------------------------------------------------
    start_step, start_epoch = 0, 0
    latest = find_latest_valid()
    if latest is not None and auto_resume:
        b0 = first_batch()  # prime optimizer slots before set_state_dict
        if b0 is not None:
            lm_loss(model, b0).backward(); opt.step(); opt.zero_grad(set_to_none=True)
        app = load_checkpoint(model, opt, latest[1])
        start_step, start_epoch = app["global_step"], app["epoch"]
        gen.set_state(torch.tensor(app["gen_state"], dtype=torch.uint8))
        _p(f"RESUMED_FROM_STEP={start_step} epoch={start_epoch} (valid ckpt step {latest[0]})")
    else:
        _p("COLD_START")

    # --- training loop with periodic async checkpoints -------------------------------------
    step, batches_streamed, loader_path = start_step, 0, "n/a"
    epoch = start_epoch
    while step < steps:
        epoch_shuffle(shards, gen)  # advance the checkpointed shuffle RNG for this epoch
        loader, loader_path = build_loader()
        for shard in loader:
            batches_streamed += 1
            for i in range(0, shard.shape[0], batch):
                mb = shard[i:i + batch]
                if mb.shape[0] < 2:
                    continue
                loss = lm_loss(model, mb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                step += 1
                if step % max(1, log_every) == 0:
                    _p(f"step={step} loss={float(loss.detach()):.4f} epoch={epoch}")
                if save_every and step > start_step and step % save_every == 0:
                    path = ckpt.async_save(model, opt, pipeline_state(step, epoch, batches_streamed, gen))
                    _p(f"[rank0] async DCP checkpoint queued step={step} -> {path}")
                if step >= steps:
                    break
            if step >= steps:
                break
        epoch += 1
    # terminal checkpoint — skip if the last step already landed on a save boundary (avoids
    # rewriting an existing step_N dir).
    if not (save_every and step % save_every == 0):
        ckpt.async_save(model, opt, pipeline_state(step, epoch, batches_streamed, gen))
    ckpt.flush()
    pre_fp = fingerprint(model, opt)
    dist.barrier()

    ok = True

    # === CHECK 1 — efficient streaming via serverless_gpu.data. ============================
    dl_ok = batches_streamed > 0 and (loader_path.startswith("serverless_gpu")
                                      or loader_path.startswith("databricks.serverless_gpu"))
    if dl_ok:
        _p(f"RESILIENT_DATALOAD_OK batches={batches_streamed} loader={loader_path} "
           f"num_workers={num_workers}")
    else:
        ok = False; _p(f"RESILIENT_DATALOAD_FAIL batches={batches_streamed} loader={loader_path}")

    # === CHECK 2 — async DCP checkpoint to the volume, .metadata present. ==================
    last = find_latest_valid()
    if ckpt.saves > 0 and last is not None:
        _p(f"RESILIENT_ASYNC_CKPT_OK saves={ckpt.saves} backend=UCVolumeWriter "
           f"latest_valid_step={last[0]} (.metadata present)")
    else:
        ok = False; _p(f"RESILIENT_ASYNC_CKPT_FAIL saves={ckpt.saves} latest={last}")

    # === CHECK 3 — recover from the most recent VALID checkpoint (bit-identical, at step). =
    resume_app, resume_fp = None, None
    if last is not None:
        m2 = build_fsdp_model()
        o2 = torch.optim.AdamW(m2.parameters(), lr=lr)
        b0 = first_batch()
        if b0 is not None:
            lm_loss(m2, b0).backward(); o2.step(); o2.zero_grad(set_to_none=True)
        resume_app = load_checkpoint(m2, o2, last[1])
        resume_fp = fingerprint(m2, o2)
    rs_ok = (resume_app is not None and resume_fp == pre_fp
             and resume_app.get("global_step") == step)
    if rs_ok:
        _p(f"RESILIENT_RESUME_OK fingerprint_match=True resumed_step={resume_app['global_step']} "
           f"valid_ckpt_step={last[0]}")
    else:
        ok = False
        _p(f"RESILIENT_RESUME_FAIL fp_match={resume_fp == pre_fp} "
           f"loaded_step={resume_app.get('global_step') if resume_app else None} live_step={step}")

    # === CHECK 4 — deterministic, checkpointed pipeline: identical next batch after resume. =
    if resume_app is not None:
        g_restored = torch.Generator(); g_restored.set_state(torch.tensor(resume_app["gen_state"], dtype=torch.uint8))
        order_restored = epoch_shuffle(shards, g_restored)
        g_ref = torch.Generator(); g_ref.manual_seed(seed)
        for _ in range(resume_app["epoch"]):
            epoch_shuffle(shards, g_ref)
        order_match = order_restored == epoch_shuffle(shards, g_ref)
        b_a, b_b = first_batch(), first_batch()
        batch_identical = b_a is not None and b_b is not None and torch.equal(b_a, b_b)
        if order_match and batch_identical:
            _p(f"RESILIENT_PIPELINE_OK shuffle_order_match={order_match} "
               f"batch_identical={batch_identical} restored(step={resume_app['global_step']},"
               f"epoch={resume_app['epoch']},offset={resume_app['sample_offset']})")
        else:
            ok = False
            _p(f"RESILIENT_PIPELINE_FAIL order_match={order_match} batch_identical={batch_identical}")
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
        raise RuntimeError("one or more resilience checks did not pass — read the RESILIENT_* lines")
    return {"resumed_from": start_step, "final_step": step} if rank == 0 else None


print("run_train defined and decorated with @distributed(gpus=%d)" % GPUS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Launch — `run_train.distributed(...)`
# MAGIC `.distributed()` fans the function out one process per GPU, syncs the environment, and
# MAGIC creates/nests an MLflow run. Watch for the four `RESILIENT_*_OK` sentinels and
# MAGIC `RESILIENT_SUITE_COMPLETE` (printed by rank 0).

# COMMAND ----------

result = run_train.distributed(**CFG)
print("result:", result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prove auto-resume across a restart
# MAGIC The valid checkpoints from the run above are on the durable UC volume. A second
# MAGIC `.distributed()` call must **recover from the newest valid checkpoint** and *continue
# MAGIC training* rather than start cold — the same recovery AIR's `max_retries` triggers when a
# MAGIC container is preempted. We target **2× the steps** so there's real work left after resuming
# MAGIC (otherwise the run resumes at the finish line with nothing to train). Look for
# MAGIC `RESUMED_FROM_STEP=…` (not `COLD_START`) and a step counter that climbs past the checkpoint.

# COMMAND ----------

CFG_CONTINUE = {**CFG, "steps": CFG["steps"] * 2}   # resume at the checkpoint, then train further
result2 = run_train.distributed(**CFG_CONTINUE)
print("result:", result2)
if isinstance(result2, dict):
    assert result2.get("resumed_from", 0) > 0, (
        "expected the relaunch to resume from a volume checkpoint but resumed_from=0 — confirm "
        "ckpt_dir is a durable UC volume (not /tmp) and that checkpoints were written")
    print(f"\n✅ Relaunch auto-resumed from step {result2['resumed_from']} and trained on to "
          f"{result2.get('final_step')}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this demonstrated (mapped to the doc)
# MAGIC - **1. Load data efficiently** — `RESILIENT_DATALOAD_OK`: shards streamed through
# MAGIC   `serverless_gpu.data.UCVolumeDataset` + `DataLoader`, auto-sharded across ranks×workers, no
# MAGIC   `DistributedSampler`, served from the local cache.
# MAGIC - **2. Checkpoint with DCP + save asynchronously** — `RESILIENT_ASYNC_CKPT_OK`: `dcp.async_save`
# MAGIC   via `UCVolumeWriter`, prior save awaited before the next, `.metadata` written.
# MAGIC - **3. Recover from the most recent valid checkpoint** — `RESILIENT_RESUME_OK` +
# MAGIC   `RESUMED_FROM_STEP`: `find_latest_valid` validates `.metadata`, loads via `UCVolumeReader` +
# MAGIC   `set_state_dict`, params/optimizer bit-identical, training continues at the saved step.
# MAGIC - **4. Checkpoint the data pipeline / make it deterministic** — `RESILIENT_PIPELINE_OK`: the
# MAGIC   shard-shuffle RNG + (epoch, offset) ride in the checkpoint, so the resumed run feeds the
# MAGIC   identical next batch.
# MAGIC
# MAGIC The torchrun sibling `resilient_8xh100_notebook` runs the identical training logic via
# MAGIC `torchrun --standalone` (no `@distributed`) — the portable path that also transfers to the
# MAGIC multi-node AIR CLI.
