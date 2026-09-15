# Databricks notebook source
# MAGIC %md
# MAGIC # TabICLv2 pretraining on 8×H100 — resilient checkpointing via DCP + `UCVolumeWriter`
# MAGIC
# MAGIC Runs your **real `tabicl.train`** stage-1 pretraining across a full `GPU_8xH100` node with
# MAGIC `@distributed`, but **replaces TabICL's native `torch.save(...).ckpt` checkpoint layer with
# MAGIC PyTorch Distributed Checkpoint (DCP) writing to a UC volume via `serverless_gpu.data`** — the
# MAGIC mechanism from the AIR **Performance and resiliency**
# MAGIC [guide](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency).
# MAGIC We do it by monkeypatching the upstream `Trainer` checkpoint methods before launching, so the
# MAGIC training loop and numerics are 100% upstream — only *persistence* changes.
# MAGIC
# MAGIC ### Which doc best practices this covers (and the one it can't)
# MAGIC | Doc section | Here | How |
# MAGIC |---|---|---|
# MAGIC | **Checkpoint with DCP** | ✅ | `dcp.save` with `serverless_gpu.data.UCVolumeWriter`; model weights stored as real DCP tensors |
# MAGIC | **Recover from the most recent valid checkpoint** | ✅ | newest `step-N/` dir carrying a `.metadata` file, loaded via `UCVolumeReader` |
# MAGIC | **Checkpoint the data pipeline / make it deterministic** | ✅ | `curr_step` is persisted in the checkpoint; TabICL's `seed()` reseeds priors by `curr_step`, so a resumed run continues the synthetic-data stream instead of regenerating from step 0 |
# MAGIC | **Save asynchronously** (`dcp.async_save`) | ❌ | TabICL saves from **rank 0 only** (its `save_checkpoint` is under a `master_process` guard) and inits an **NCCL-only** process group; `async_save` needs an all-ranks collective + a CPU/gloo backend. Documented as a follow-up (un-guard the save to all ranks + add a gloo group). We use **synchronous** `dcp.save(no_dist=True)`. |
# MAGIC | **Load data efficiently** (`UCVolumeDataset`) | — | N/A: TabICL **generates synthetic priors on the fly** (`graph_scm`), it doesn't stream a corpus from a volume. |
# MAGIC
# MAGIC ### Why `no_dist=True`
# MAGIC Under DDP the model is **replicated** on every rank, so rank 0 already holds the full state.
# MAGIC A rank-0-only `dcp.save(..., no_dist=True)` writes the complete checkpoint with no collective —
# MAGIC which is exactly what fits TabICL's rank-0-only save. On load, all 8 ranks each read the full
# MAGIC checkpoint independently (`no_dist=True`) and land identical replicas.
# MAGIC
# MAGIC ### Launch: `@distributed` (Beta)
# MAGIC > ⚠️ Same scope caveat as the sibling notebooks: `@distributed` is single-node and the repo's
# MAGIC > steering notes default the customer engagement to the torchrun/CLI path
# MAGIC > (`AGENTS.md`, `docs/private/uat-plan-2026-07.md`). The DCP monkeypatch below is launcher-agnostic —
# MAGIC > it works identically under `torchrun`.
# MAGIC
# MAGIC ### How to run
# MAGIC 1. Attach an **AI base environment (v5)** + **`GPU_8xH100`**; make `tabicl` importable
# MAGIC    (vendored-wheels recipe: `docs/cookbook/install-packages-from-a-volume.md`).
# MAGIC 2. Fill the **catalog / schema / volume** widgets (writable UC volume) and `repo_root`.
# MAGIC 3. Run top to bottom. Cell 4 pretrains cold to `max_steps`; cell 5 relaunches at 2× steps and
# MAGIC    must **resume from the DCP checkpoint** (look for `DCP_RESUME step=…`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

import os

dbutils.widgets.text("gpus", "8", "GPUs for @distributed (8 = full GPU_8xH100 node)")
dbutils.widgets.text("repo_root", "/Workspace/Shared/databricks-air-lab", "repo root synced in the workspace")
dbutils.widgets.text("catalog", "<catalog>", "catalog")
dbutils.widgets.text("schema", "<schema>", "schema")
dbutils.widgets.text("volume", "<volume>", "UC volume (writable)")
dbutils.widgets.text("base_dir", "tabicl-8xh100-dcp", "subdir in the volume for this run")
# run size / throughput knobs
dbutils.widgets.text("max_steps", "80", "optimizer steps (short, readable; full stage-1 = 500,000)")
dbutils.widgets.text("save_temp_every", "20", "checkpoint (DCP) every N steps")
dbutils.widgets.text("micro_batch_size", "4", "micro-batch — must DIVIDE batch_size_per_gp (=4) so a micro-batch stays within one prior group (same seq_len/train_size); 8 spans two groups and tabicl rejects it")
dbutils.widgets.text("batch_size", "64", "datasets per optimizer step (upstream stage-1)")
dbutils.widgets.text("n_jobs", "16", "CPU prior-gen workers per rank")
dbutils.widgets.text("lr", "8e-4", "learning rate (upstream stage-1)")
dbutils.widgets.dropdown("dtype", "float32", ["float32", "bfloat16"], "compute dtype (upstream = float32)")

GPUS = int(dbutils.widgets.get("gpus"))
REPO_ROOT = dbutils.widgets.get("repo_root")
TABICL_DIR = os.path.join(REPO_ROOT, "experiments/foundation-models/tabicl")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
BASE_DIR = dbutils.widgets.get("base_dir")
assert "<" not in (CATALOG + SCHEMA + VOLUME), (
    "Fill in the catalog / schema / volume widgets with a writable UC volume before running "
    "(checkpoints must be on a durable volume, never /tmp).")
CKPT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{BASE_DIR}/ckpt"

CFG = dict(
    tabicl_dir=TABICL_DIR,
    ckpt_dir=CKPT_DIR,
    max_steps=int(dbutils.widgets.get("max_steps")),
    save_temp_every=int(dbutils.widgets.get("save_temp_every")),
    micro_batch_size=int(dbutils.widgets.get("micro_batch_size")),
    batch_size=int(dbutils.widgets.get("batch_size")),
    n_jobs=int(dbutils.widgets.get("n_jobs")),
    lr=float(dbutils.widgets.get("lr")),
    dtype=dbutils.widgets.get("dtype"),
)
print("GPUs        :", GPUS)
print("tabicl dir  :", TABICL_DIR)
print("checkpoints :", CKPT_DIR, "(DCP dirs on a durable UC volume)")
print("steps       :", CFG["max_steps"], "| DCP every", CFG["save_temp_every"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment check
# MAGIC 8-GPU H100 node, `serverless_gpu` importable, `tabicl` importable, and the upstream
# MAGIC `Trainer.save_checkpoint / load_checkpoint / get_latest_checkpoint / manage_checkpoint`
# MAGIC methods present (the monkeypatch targets — if upstream renamed them this cell says so).

# COMMAND ----------

import sys

import torch

n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
print(f"visible GPUs: {n_gpu}")
assert n_gpu == GPUS, f"expected {GPUS} GPUs but see {n_gpu} — attach a GPU_8xH100 (or set gpus)"

try:
    from serverless_gpu import distributed  # noqa: F401
    import serverless_gpu.data as _sgc
    print("serverless_gpu.distributed + data:",
          [n for n in ("UCVolumeWriter", "UCVolumeReader") if hasattr(_sgc, n)])
except Exception as e:  # noqa: BLE001
    print(f"⚠️ serverless_gpu not importable ({type(e).__name__}: {e}) — attach an AI base env (v5) + GPU.")

assert os.path.isdir(TABICL_DIR), f"tabicl_dir not found: {TABICL_DIR} — set repo_root"
if TABICL_DIR not in sys.path:
    sys.path.insert(0, TABICL_DIR)
try:
    import tabicl  # noqa: F401
    from tabicl.train import _run as _trun
    have = [m for m in ("save_checkpoint", "load_checkpoint", "get_latest_checkpoint",
                        "manage_checkpoint") if hasattr(_trun.Trainer, m)]
    print("tabicl OK; Trainer methods to patch present:", have)
    assert len(have) == 4, "upstream Trainer checkpoint API changed — the monkeypatch needs updating"
except Exception as e:  # noqa: BLE001
    print(f"⚠️ tabicl NOT importable / API changed ({type(e).__name__}: {e}).")
    print("   → vendor tabicl's wheels: docs/cookbook/install-packages-from-a-volume.md . Do not proceed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. The distributed pretraining function (with the DCP checkpoint monkeypatch)
# MAGIC `@distributed`, everything inside the function (closure serialized to the workers). Before
# MAGIC `runpy.run_module("tabicl.train")` we swap the four `Trainer` checkpoint methods to route
# MAGIC through DCP + `UCVolumeWriter`/`UCVolumeReader`. `tabicl.train` inits its own DDP group, so we
# MAGIC only set the device (no `init_process_group` here).

# COMMAND ----------

from serverless_gpu import distributed


@distributed(gpus=GPUS)  # gpu_type auto-detected from the attached GPU_8xH100 accelerator
def pretrain(tabicl_dir, ckpt_dir, max_steps, save_temp_every, micro_batch_size, batch_size,
             n_jobs, lr, dtype):
    import io
    import os
    import runpy
    import shutil
    import sys

    import torch

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)   # do NOT init the PG — tabicl.train inits its own DDP group

    def _p(*a):
        if rank == 0:
            print(*a, flush=True)

    if tabicl_dir not in sys.path:
        sys.path.insert(0, tabicl_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    # wandb → MLflow shim is a LOCAL repo file (tabicl_dir); the @distributed workers may not have
    # the workspace repo path mounted. It's optional for the DCP checkpoint demo — if it's not
    # importable here, run with --wandb_log False so tabicl.train never touches wandb.
    wandb_log = "False"
    try:
        from wandb_mlflow_shim import install
        install()
        wandb_log = "True"
        _p("[launch] wandb→MLflow shim installed; tabicl metrics → MLflow")
    except Exception as e:  # noqa: BLE001
        _p(f"[launch] wandb_mlflow_shim unavailable ({type(e).__name__}: {e}) — "
           f"running with --wandb_log False; DCP checkpoint demo is unaffected")

    # -------------------------------------------------------------------------------------
    # DCP CHECKPOINT MONKEYPATCH — swap tabicl Trainer's torch.save/.ckpt I/O for torch DCP +
    # serverless_gpu.data.UCVolumeWriter/Reader. Model weights ride as real DCP tensors; the
    # optimizer + scheduler + curr_step ride as one opaque blob (replicated under DDP, so no
    # collective is needed just to shard them). Save is rank-0-only + no_dist (matches upstream's
    # master_process guard); load is per-rank + no_dist (each replica reads the full checkpoint).
    # `events` (closure-captured, per process) records what happened for the rank-0 report.
    # -------------------------------------------------------------------------------------
    events = {"saves": [], "resumed_from": None, "backend": None}

    def _sgc():
        try:
            from serverless_gpu.data import UCVolumeWriter, UCVolumeReader
        except ImportError:
            from databricks.serverless_gpu.data import UCVolumeWriter, UCVolumeReader
        return UCVolumeWriter, UCVolumeReader

    def _install_dcp_checkpointing():
        import torch.distributed.checkpoint as dcp
        from tabicl.train import _run as trun

        def _dir(self, step):
            return os.path.join(self.config.checkpoint_dir, f"step-{step}")

        def save_checkpoint(self, name):  # called on rank 0 only (master_process guard in train())
            step = self.curr_step
            d = _dir(self, step)
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            buf = io.BytesIO()
            torch.save({"optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        "curr_step": step, "config": self.model_config}, buf)
            state = {"model": self.raw_model.state_dict(), "extra": buf.getvalue()}
            UCVolumeWriter, _ = _sgc()
            dcp.save(state, storage_writer=UCVolumeWriter(d), no_dist=True)  # sync; see notebook header
            events["saves"].append(step)
            events["backend"] = "UCVolumeWriter"
            print(f"DCP_CKPT_SAVE step={step} -> {d} (UCVolumeWriter, no_dist)", flush=True)

        def get_latest_checkpoint(self):
            cd = self.config.checkpoint_dir
            if not os.path.isdir(cd):
                return None
            cands = []
            for n in os.listdir(cd):
                p = os.path.join(cd, n)
                if n.startswith("step-") and os.path.isdir(p) and os.path.exists(os.path.join(p, ".metadata")):
                    try:
                        cands.append((int(n.split("-")[1]), p))
                    except ValueError:
                        pass
            return max(cands)[1] if cands else None

        def load_checkpoint(self):  # called on ALL ranks in __init__; each reads independently
            path = getattr(self.config, "checkpoint_path", None) or get_latest_checkpoint(self)
            if not path or not os.path.isdir(path) or not os.path.exists(os.path.join(path, ".metadata")):
                print("No DCP checkpoint found, starting from scratch.", flush=True)
                return
            _, UCVolumeReader = _sgc()
            state = {"model": self.raw_model.state_dict(), "extra": b""}
            dcp.load(state, storage_reader=UCVolumeReader(path), no_dist=True)
            self.raw_model.load_state_dict(state["model"])
            if getattr(self.config, "only_load_model", False):
                print(f"Loaded model weights only from {path}", flush=True)
                return
            extra = torch.load(io.BytesIO(state["extra"]), map_location=self.config.device,
                               weights_only=False)
            self.optimizer.load_state_dict(extra["optimizer_state"])
            self.scheduler.load_state_dict(extra["scheduler_state"])
            self.curr_step = extra["curr_step"]
            events["resumed_from"] = self.curr_step
            print(f"DCP_RESUME step={self.curr_step} <- {path}", flush=True)

        def manage_checkpoint(self):  # delete oldest TEMP checkpoint DIRECTORIES (not .ckpt files)
            cd = self.config.checkpoint_dir
            temps = []
            for n in os.listdir(cd):
                p = os.path.join(cd, n)
                if n.startswith("step-") and os.path.isdir(p):
                    try:
                        step = int(n.split("-")[1])
                        if step % self.config.save_perm_every != 0:
                            temps.append((step, p))
                    except ValueError:
                        pass
            temps.sort()
            for _, p in temps[:max(0, len(temps) - self.config.max_checkpoints)]:
                shutil.rmtree(p, ignore_errors=True)

        trun.Trainer.save_checkpoint = save_checkpoint
        trun.Trainer.get_latest_checkpoint = get_latest_checkpoint
        trun.Trainer.load_checkpoint = load_checkpoint
        trun.Trainer.manage_checkpoint = manage_checkpoint
        return "UCVolumeWriter/UCVolumeReader"

    backend = _install_dcp_checkpointing()
    _p(f"[launch] world={world}; DCP checkpoint backend = {backend}; ckpt_dir = {ckpt_dir}")

    # --- build tabicl.train argv: knobs on top, pinned upstream stage-1 recipe below -----------
    argv = [
        "tabicl.train",
        "--wandb_log", wandb_log, "--device", "cuda", "--dtype", str(dtype),
        "--np_seed", "42", "--torch_seed", "42",
        "--max_steps", str(max_steps), "--batch_size", str(batch_size),
        "--micro_batch_size", str(micro_batch_size), "--lr", str(lr), "--muon", "True",
        "--n_jobs", str(n_jobs), "--prior_device", "cpu",
        "--max_features", "100", "--max_classes", "10", "--max_seq_len", "1024",
        # checkpointing → our DCP monkeypatch honors these exactly as upstream would
        "--checkpoint_dir", ckpt_dir,
        "--save_temp_every", str(save_temp_every), "--save_perm_every", str(max_steps),
        "--max_checkpoints", "3", "--only_load_model", "False",
        # --- pinned upstream stage-1 recipe (scripts/train_v2_clf_stage1.sh) -------------------
        "--beta1", "0.9", "--weight_decay", "0.01", "--use_cautious_wd", "False",
        "--scheduler", "cosine_with_restarts", "--warmup_proportion", "0.01",
        "--cosine_num_cycles", "1", "--cosine_amplitude_decay", "1", "--cosine_lr_end", "1e-7",
        "--gradient_clipping", "10.0", "--prior_type", "graph_scm", "--batch_size_per_gp", "4",
        "--min_features", "1", "--min_train_size", "0.3", "--max_train_size", "0.9",
        "--seq_len_per_gp", "True", "--graph_noise", "False",
        "--filter_unpredictable_graphs", "True", "--filter_unpredictable_datasets", "True",
        "--allow_act_warping", "False", "--min_n_nodes", "2", "--max_n_nodes", "32",
        "--cauchy_dag_offset", "0.0", "--embed_dim", "128",
        "--col_num_blocks", "3", "--col_nhead", "8", "--col_num_inds", "128",
        "--col_affine", "False", "--col_feature_group", "same", "--col_feature_group_size", "3",
        "--col_target_aware", "True", "--col_ssmax", "True",
        "--row_num_blocks", "3", "--row_nhead", "8", "--row_num_cls", "4",
        "--row_rope_base", "100000", "--row_rope_interleaved", "False",
        "--icl_num_blocks", "12", "--icl_nhead", "8", "--icl_ssmax", "True",
        "--ssmax_type", "qassmax-mlp-elementwise",
        "--ff_factor", "2", "--norm_first", "True", "--zero_init", "False",
    ]
    sys.argv = argv

    # --- run the real trainer (blocks until max_steps; all ranks participate in DDP) -----------
    train_error = None
    try:
        runpy.run_module("tabicl.train", run_name="__main__", alter_sys=True)
    except Exception as e:  # noqa: BLE001 — capture so rank 0 can still emit the report
        train_error = e

    if rank != 0:
        if train_error is not None:
            raise train_error
        return None

    # ---- rank 0: acceptance report + sentinel ----
    latest = None
    if os.path.isdir(ckpt_dir):
        dirs = [(int(n.split("-")[1]), os.path.join(ckpt_dir, n))
                for n in os.listdir(ckpt_dir)
                if n.startswith("step-") and os.path.isdir(os.path.join(ckpt_dir, n))
                and os.path.exists(os.path.join(ckpt_dir, n, ".metadata"))
                and n.split("-")[1].isdigit()]
        latest = max(dirs)[0] if dirs else None

    c1 = world == 8
    c2 = train_error is None
    c3 = len(events["saves"]) > 0 and latest is not None       # DCP checkpoints with .metadata
    print("\n" + "=" * 78)
    print("ACCEPTANCE REPORT — TabICL pretrain with DCP/UCVolume checkpointing")
    print("=" * 78)
    print(f"CHECK 1  full 8×H100 node ................. {'PASS' if c1 else 'FAIL'} (world={world})")
    print(f"CHECK 2  tabicl.train ran to completion .. {'PASS' if c2 else 'FAIL'}"
          + ("" if c2 else f" ({type(train_error).__name__}: {train_error})"))
    print(f"CHECK 3  DCP checkpoints written (.metadata) {'PASS' if c3 else 'FAIL'} "
          f"(steps={events['saves']}, latest_valid={latest}, backend={events['backend']})")
    resumed = events["resumed_from"]
    print(f"INFO     resumed_from ..................... {resumed} "
          f"({'COLD_START' if not resumed else 'recovered via UCVolumeReader'})")
    ok = c1 and c2 and c3
    print("-" * 78)
    if ok:
        print(f"VERDICT: ACCEPTED — TabICLv2 stage-1 pretrained on 8×H100 with DCP/UCVolume checkpoints.")
        print(f"TABICL_DCP_PRETRAIN_OK world={world} steps={max_steps} dcp_saves={len(events['saves'])} "
              f"latest_valid={latest} resumed_from={resumed} backend={events['backend']}")
    else:
        print("VERDICT: NOT ACCEPTED — read the failing CHECK above.")

    if train_error is not None:
        raise train_error
    if not ok:
        raise RuntimeError("one or more checks failed — see the ACCEPTANCE REPORT above")
    return {"world": world, "steps": max_steps, "dcp_saves": events["saves"],
            "latest_valid": latest, "resumed_from": resumed}


print("pretrain defined and decorated with @distributed(gpus=%d)" % GPUS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. (optional) Reset — wipe prior DCP checkpoints for a clean COLD_START
# MAGIC Run once if the volume already has checkpoints from an earlier run (otherwise the launch
# MAGIC below will resume from them instead of starting cold).

# COMMAND ----------

# dbutils.fs.rm(CFG["ckpt_dir"].replace("/Volumes", "dbfs:/Volumes"), recurse=True); print("cleared", CFG["ckpt_dir"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Launch — cold start → train to `max_steps`
# MAGIC Writes DCP checkpoints (`step-N/` dirs with `.metadata`) to the UC volume via `UCVolumeWriter`.
# MAGIC Rank 0 prints `DCP_CKPT_SAVE step=…` per save and ends with `TABICL_DCP_PRETRAIN_OK`.

# COMMAND ----------

result = pretrain.distributed(**CFG)
print("\nresult:", result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Prove resilient recovery — relaunch at 2× steps
# MAGIC The DCP checkpoints are on the durable volume. Relaunching with more steps must **recover the
# MAGIC latest valid checkpoint** (via `UCVolumeReader`, `.metadata`-validated) and continue — TabICL's
# MAGIC own `load_checkpoint` (now DCP-backed) restores model+optimizer+scheduler+`curr_step`, and its
# MAGIC `seed(curr_step)` keeps the synthetic-data stream continuous. Look for `DCP_RESUME step=…`.

# COMMAND ----------

CFG_CONTINUE = {**CFG, "max_steps": CFG["max_steps"] * 2}
result2 = pretrain.distributed(**CFG_CONTINUE)
print("\nresult:", result2)
if isinstance(result2, dict):
    assert result2.get("resumed_from"), (
        "expected the relaunch to resume from a DCP checkpoint but resumed_from is empty — confirm "
        "ckpt_dir is a durable UC volume (not /tmp) and that cell 4 wrote checkpoints")
    print(f"\n✅ Relaunch recovered from DCP checkpoint at step {result2['resumed_from']} and "
          f"continued to {result2['steps']}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this demonstrated (mapped to the doc)
# MAGIC - **Checkpoint with DCP** — `TABICL_DCP_PRETRAIN_OK` + `DCP_CKPT_SAVE`: `dcp.save` via
# MAGIC   `UCVolumeWriter`, model weights as real DCP tensors, on a durable UC volume.
# MAGIC - **Recover from the most recent valid checkpoint** — `DCP_RESUME`: newest `step-N/` with a
# MAGIC   `.metadata` file, loaded via `UCVolumeReader`, restoring model+optimizer+scheduler+step.
# MAGIC - **Checkpoint the data pipeline / determinism** — `curr_step` persisted in the checkpoint;
# MAGIC   TabICL's `seed(curr_step)` reseeds the synthetic priors so the resumed run continues the data
# MAGIC   stream rather than regenerating from step 0.
# MAGIC - **Not covered:** `dcp.async_save` (TabICL's rank-0-only save + NCCL-only PG — see header) and
# MAGIC   `UCVolumeDataset` (TabICL generates priors on the fly, no corpus to stream).
# MAGIC
# MAGIC The training loop, model, and numerics are 100% upstream `tabicl.train`; only the checkpoint
# MAGIC **persistence layer** was swapped, by monkeypatching four `Trainer` methods.
