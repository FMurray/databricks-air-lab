# Databricks notebook source
# MAGIC %md
# MAGIC # TabICLv2 pretraining on 8×H100 — `@distributed` notebook (3-stage curriculum)
# MAGIC
# MAGIC Runs your **real `tabicl.train`** across **all 8 GPUs of one `GPU_8xH100` node**, launched with the
# MAGIC `serverless_gpu` **`@distributed`** decorator — **from an interactive notebook, no Job and no CLI**.
# MAGIC That last part is the point: it's the multi-GPU path you can use *today* while the
# MAGIC AI-environment-in-Jobs selectability gap is being fixed.
# MAGIC
# MAGIC ### The 3-stage curriculum (pick with the `stage` widget)
# MAGIC TabICL is an **in-context learner**: the *rows* of a synthetic table are its input sequence. The v2
# MAGIC recipe pretrains in three stages of growing table size, each chaining from the previous stage's
# MAGIC checkpoint (recipe deltas verbatim from `scripts/train_v2_clf_stage{1,2,3}.sh`):
# MAGIC
# MAGIC | Stage | Rows/table (seq len) | Steps (full) | FA3 | GPU utilization |
# MAGIC |---|---|---|---|---|
# MAGIC | **1** | ≤ 1,024 | 500,000 | off | **low & expected** — 27M-param model, ms-scale steps, host-bound |
# MAGIC | **2** | 400 – 10,240 | 40,000 | on | climbs — attention grows with rows |
# MAGIC | **3** | 400 – 60,000 | 10,000 | on | **saturates** — 60k-row attention; the OOM / "needs-B300?" case |
# MAGIC
# MAGIC **Why utilization is a stage property, not a knob:** attention cost is **quadratic in rows**. Stage 1
# MAGIC finishes each step in milliseconds and idles on CPU prior-gen + Muon host-sync — so *no setting*
# MAGIC saturates 8×H100 at stage 1. The GPUs genuinely peg at stages 2/3, where the sequences are long
# MAGIC enough to bury the cards in attention FLOPs. **So the KPI is `steps/s` / `datasets/s`, not util %.**
# MAGIC
# MAGIC ### One MLflow run per stage
# MAGIC Each launch is one run (`tabicl-clf-stage{N}-8xh100`, tagged `tabicl_stage=N`) under the notebook's
# MAGIC experiment, logging that stage's `ce`/`accuracy`/`prior_time`/`train_time` + throughput/util — so
# MAGIC stage 1/2/3 sit side by side as comparable runs.
# MAGIC
# MAGIC ### How `@distributed` differs from your `torchrun` runs
# MAGIC - You decorate a `train()` function with `@distributed(gpus=8)` and call `train.distributed(**kwargs)`.
# MAGIC   The framework fans it out **one process per GPU**, syncs the notebook env, and creates/nests an
# MAGIC   MLflow run. No `torchrun`, no subprocess.
# MAGIC - Inside the function we do exactly what `train_with_mlflow.py` does under `torchrun`: install the
# MAGIC   wandb→MLflow shim, set `sys.argv`, and `runpy.run_module("tabicl.train")`. `tabicl.train` reads
# MAGIC   `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from the env (the decorator sets them, same as `torchrun`) and
# MAGIC   initializes its **own** DDP process group — so we do **not** init one here.
# MAGIC - Imports + the arg set live **inside** the function (the closure is serialized to the workers).
# MAGIC
# MAGIC > ⚠️ **Scope note.** `@distributed` is a Beta convenience layer and is **single-node only** (this is
# MAGIC > the full 8×H100 node — the biggest single-node shape). For multi-node (16+ GPUs) you'd move to the
# MAGIC > `torchrun`/CLI path; the repo's steering notes (`AGENTS.md`, `docs/private/uat-plan-2026-07.md`)
# MAGIC > default the customer engagement to torchrun/CLI. Use this variant for the notebook-native,
# MAGIC > single-node-8×H100 case that you can run without Jobs/CLI access.
# MAGIC
# MAGIC ### How to run
# MAGIC 1. Attach to an **AI base environment (v5)** with a **`GPU_8xH100`** accelerator.
# MAGIC 2. Make sure `tabicl` is importable in the env (see the **Environment check** cell — if you hit the
# MAGIC    pip-install issue, use the vendored-wheels recipe: `docs/cookbook/install-packages-from-a-volume.md`).
# MAGIC    Stages 2/3 also need **FlashAttention-3** in the env (the env check reports it).
# MAGIC 3. Pick the **`stage`** widget (start at 1) and fill in **catalog / schema / volume** (a *writable*
# MAGIC    UC volume — checkpoints must be durable, never `/tmp`). No repo sync needed — the shim is inlined.
# MAGIC 4. Run cells top to bottom. To walk the curriculum, re-run the launch cell with `stage`=1 → 2 → 3;
# MAGIC    each is its own MLflow run and stage N auto-loads stage N-1's weights from the volume.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC The **`stage`** widget selects the faithful per-stage recipe (seq len, lr, micro-batch, FA3, grad
# MAGIC clip — verbatim from `scripts/train_v2_clf_stage{1,2,3}.sh`, soda-inria/tabicl@main). The other
# MAGIC widgets are **cross-cutting platform knobs** (gpus, prior_device, n_jobs, dtype, muon) that don't
# MAGIC change the recipe. `max_steps` is a **smoke length** — the full recipe is 500k/40k/10k by stage.

# COMMAND ----------

# --- WHAT TO RUN: the curriculum stage sets the faithful recipe (seq len, lr, micro-batch, FA3) ---
# GPU utilization is a FUNCTION OF THE STAGE, not a knob: stage 1 (≤1,024-row tables) is a small,
# host-bound workload that CANNOT saturate 8×H100 (27M-param model, ms-scale steps); stages 2/3 grow
# the in-context sequence to 10k/60k rows, where attention FLOPs (quadratic in rows) actually peg the
# GPUs. Recipe deltas below are verbatim from soda-inria/tabicl@main scripts/train_v2_clf_stage{1,2,3}.sh.
dbutils.widgets.dropdown("stage", "1", ["1", "2", "3"],
    "curriculum stage — 1: ≤1,024 rows (light/host-bound); 2: 400–10,240 (FA3); 3: 400–60,000 (FA3, GPU-heavy)")
dbutils.widgets.text("gpus", "8", "GPUs for @distributed (8 = full GPU_8xH100 node)")
dbutils.widgets.text("max_steps", "200",
    "SMOKE step count (full recipe: stage1=500,000 / stage2=40,000 / stage3=10,000). 200 ≈ warmup + readable window")

# --- platform / experiment knobs (cross-cutting; NOT part of the faithful recipe) --------------
# SPEED LEVERS for a step that's compute-bound at ~40% util: upstream runs effective fp32 (--amp is
# on but --dtype float32 ⇒ a no-op autocast), so H100 tensor cores sit idle and matmuls are slow.
# TF32 is already enabled by tabicl, so it is NOT a lever. The two levers that are:
dbutils.widgets.dropdown("dtype", "float32", ["float32", "bfloat16"],
    "TOP LEVER: bfloat16 autocast engages H100 tensor cores (fp32 barely uses them). fp32 master weights kept; bf16 needs no grad-scaler. Validate loss tracks vs fp32")
dbutils.widgets.dropdown("model_compile", "False", ["False", "True"],
    "torch.compile(dynamic=True): fuses the many small col/row/icl kernels → removes launch-overhead gaps (one-time compile warmup; strongest with bfloat16)")
dbutils.widgets.dropdown("muon", "True", ["True", "False"],
    "Muon optimizer — keep True to match the published optimization procedure")
dbutils.widgets.text("n_jobs", "16", "CPU prior-gen workers PER RANK — prior-gen is <0.001s/step (NOT the bottleneck); leave generous")
dbutils.widgets.dropdown("prior_device", "cpu", ["cpu", "cuda"],
    "keep cpu: prior-gen isn't the bottleneck, and cuda hangs under DDP (each spawn worker opens its own CUDA context)")
dbutils.widgets.dropdown("stage3_recompute", "off (upstream)", ["off (upstream)", "on (if OOM)"],
    "stage-3 ONLY: gradient checkpointing — turn on if the 60,000-row tables OOM (trades compute for memory)")

# --- durable storage (per-stage checkpoints land in <base_dir>/stage{N}; stage N+1 chains from N) ---
dbutils.widgets.text("catalog", "<catalog>", "catalog")
dbutils.widgets.text("schema", "<schema>", "schema")
dbutils.widgets.text("volume", "<volume>", "UC volume (writable)")
dbutils.widgets.text("base_dir", "tabicl-8xh100-pretrain", "subdir in the volume for this pretraining run")

# COMMAND ----------

import os

GPUS = int(dbutils.widgets.get("gpus"))
STAGE = int(dbutils.widgets.get("stage"))
MAX_STEPS = int(dbutils.widgets.get("max_steps"))
BATCH_SIZE = 64  # upstream: constant across all three stages

# Faithful per-stage recipe deltas (soda-inria/tabicl@main scripts/train_v2_clf_stage{1,2,3}.sh).
# Everything NOT here is identical across stages and is pinned in the training function.
STAGE_SHAPES = {
    1: dict(lr=8e-4, micro_batch_size=4, batch_size_per_gp=4, min_seq_len=None, max_seq_len=1024,
            log_seq_len=False, min_train_size=0.3, max_train_size=0.9, gradient_clipping=10.0,
            use_flash_attn3=False, recompute=None),   # FA3 off; ≤1,024-row tables
    2: dict(lr=1e-4, micro_batch_size=1, batch_size_per_gp=1, min_seq_len=400, max_seq_len=10240,
            log_seq_len=True, min_train_size=0.79, max_train_size=0.81, gradient_clipping=10.0,
            use_flash_attn3=True, recompute=None),     # FA3 on; up to 10,240-row tables
    3: dict(lr=2e-5, micro_batch_size=1, batch_size_per_gp=1, min_seq_len=400, max_seq_len=60000,
            log_seq_len=True, min_train_size=0.79, max_train_size=0.81, gradient_clipping=1.0,
            use_flash_attn3=True, recompute=False),    # FA3 on; up to 60,000-row tables (the OOM/B300 case)
}
SHAPE = dict(STAGE_SHAPES[STAGE])
if STAGE == 3 and dbutils.widgets.get("stage3_recompute").startswith("on"):
    SHAPE["recompute"] = True  # gradient checkpointing on if the 60k-row tables OOM

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
BASE_DIR = dbutils.widgets.get("base_dir")
assert "<" not in (CATALOG + SCHEMA + VOLUME), (
    "Fill in the catalog / schema / volume widgets with a writable UC volume before running "
    "(checkpoints must be on a durable volume, never /tmp).")
VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{BASE_DIR}"
CKPT_DIR = f"{VOL_ROOT}/stage{STAGE}"                              # this stage's checkpoints
PREV_CKPT_DIR = f"{VOL_ROOT}/stage{STAGE - 1}" if STAGE > 1 else None  # chain-from source (weights only)

# --- batching invariant (fail fast HERE, before launching 8 GPUs) -----------------------------
# per-rank batch (batch_size/gpus) must tile into prior groups, and a micro-batch may not span two
# groups (they carry different seq_len/train_size — tabicl validate_micro_batch). All three stage
# shapes satisfy this (s1 4/4, s2 1/1, s3 1/1); the guard catches any hand-edit that breaks it.
assert BATCH_SIZE % GPUS == 0, f"batch_size ({BATCH_SIZE}) must be divisible by gpus ({GPUS})."
PER_RANK = BATCH_SIZE // GPUS
assert PER_RANK % SHAPE["batch_size_per_gp"] == 0, (
    f"per-rank batch ({PER_RANK}) must be a multiple of batch_size_per_gp ({SHAPE['batch_size_per_gp']}).")
assert SHAPE["batch_size_per_gp"] % SHAPE["micro_batch_size"] == 0, (
    f"micro_batch_size ({SHAPE['micro_batch_size']}) must evenly divide batch_size_per_gp "
    f"({SHAPE['batch_size_per_gp']}) — a micro-batch cannot span two prior groups.")

# All simple types so the closure stays cheap to serialize to the 8 workers.
CFG = dict(
    stage=STAGE,
    ckpt_dir=CKPT_DIR,
    prev_ckpt_dir=PREV_CKPT_DIR,
    run_name=f"tabicl-clf-stage{STAGE}-8xh100",
    max_steps=MAX_STEPS,
    batch_size=BATCH_SIZE,
    n_jobs=int(dbutils.widgets.get("n_jobs")),
    dtype=dbutils.widgets.get("dtype"),
    model_compile=dbutils.widgets.get("model_compile"),
    muon=dbutils.widgets.get("muon"),
    prior_device=dbutils.widgets.get("prior_device"),
    max_features=100,   # upstream: constant across stages
    max_classes=10,     # upstream: constant across stages
    save_temp_every=max(1, MAX_STEPS // 2),  # ≥2 checkpoints in a short smoke run
    **SHAPE,            # lr, micro_batch_size, batch_size_per_gp, min/max_seq_len, log_seq_len, train sizes, grad clip, fa3, recompute
)

print(f"STAGE {STAGE}  seq_len≤{SHAPE['max_seq_len']}  lr={SHAPE['lr']}  micro_batch={SHAPE['micro_batch_size']}  "
      f"FA3={SHAPE['use_flash_attn3']}  (recompute={SHAPE['recompute']})")
print("checkpoints   :", CKPT_DIR, "(durable UC volume)")
if PREV_CKPT_DIR:
    print("chains from   :", PREV_CKPT_DIR, "(stage weights only, on first launch)")
print(f"batching      : batch_size={BATCH_SIZE} → {PER_RANK}/rank, group={SHAPE['batch_size_per_gp']}, "
      f"micro_batch={SHAPE['micro_batch_size']}  (invariants OK)")
print("platform      :", f"gpus={GPUS} dtype={CFG['dtype']} model_compile={CFG['model_compile']} "
      f"muon={CFG['muon']} n_jobs={CFG['n_jobs']} prior_device={CFG['prior_device']}")
print("steps (smoke) :", MAX_STEPS, "|  full recipe:", {1: 500000, 2: 40000, 3: 10000}[STAGE])
if SHAPE["use_flash_attn3"]:
    print("⚠️ stage needs FlashAttention-3 (Hopper) — see the env check; if absent, stage 2/3 will error.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment check
# MAGIC Confirm this is an 8-GPU H100 node, that `serverless_gpu.distributed` is importable, and that
# MAGIC **`tabicl` resolves**. If `tabicl` is missing this is the pip-install issue — vendor the wheels
# MAGIC into a UC volume and add them to the env (`docs/cookbook/install-packages-from-a-volume.md`).

# COMMAND ----------

import sys

import torch

n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
print(f"visible GPUs: {n_gpu}")
for i in range(n_gpu):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
assert n_gpu == GPUS, (
    f"expected {GPUS} GPUs but see {n_gpu} — attach a GPU_8xH100 accelerator (or set the gpus widget)")

try:
    from serverless_gpu import distributed  # noqa: F401
    print("serverless_gpu.distributed: OK")
except Exception as e:  # noqa: BLE001
    print(f"⚠️ serverless_gpu not importable ({type(e).__name__}: {e}) — attach an AI base env (v5) + GPU.")

# tabicl must be importable in the env. The wandb→MLflow shim is INLINED in the training cell, so
# no repo files need to be synced — this notebook is standalone.
try:
    import tabicl  # noqa: F401
    print("tabicl importable: OK  (version:", getattr(tabicl, "__version__", "unknown"), ")")
except Exception as e:  # noqa: BLE001
    print(f"⚠️ tabicl NOT importable ({type(e).__name__}: {e}).")
    print("   → This is the pip-install issue. Vendor tabicl's wheels into a UC volume and add them to")
    print("     the env: docs/cookbook/install-packages-from-a-volume.md . Do NOT proceed until this is OK.")

# FlashAttention-3 — required by stages 2 & 3 (--use_flash_attn3 True). Stage 1 runs without it.
_fa3 = next((m for m in ("flash_attn_interface", "flash_attn_3", "flash_attn") if __import__(
    "importlib").util.find_spec(m) is not None), None)
print(f"FlashAttention-3: {'found (' + _fa3 + ')' if _fa3 else 'NOT found'}  "
      f"— {'ok for any stage' if _fa3 else 'stage 1 fine; stages 2/3 will error until FA3 is installed'}")
print("wandb→MLflow shim: inlined in the training function (standalone — no external shim files)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The distributed pretraining function
# MAGIC Decorated with `@distributed(gpus=…)`. It replicates the `train_with_mlflow.py` torchrun
# MAGIC entrypoint — shim → `sys.argv` → `runpy.run_module("tabicl.train")` — names a per-stage MLflow
# MAGIC run, chains from the previous stage's checkpoint, and adds a rank-0 GPU-util sampler. The
# MAGIC stage-varying args come from `SHAPE`; everything else is the pinned cross-stage recipe.

# COMMAND ----------

from serverless_gpu import distributed


@distributed(gpus=GPUS)  # gpu_type auto-detected from the attached GPU_8xH100 accelerator
def pretrain(stage, ckpt_dir, prev_ckpt_dir, run_name, max_steps, batch_size, batch_size_per_gp,
             micro_batch_size, n_jobs, lr, dtype, model_compile, muon, prior_device, use_flash_attn3,
             max_features, max_classes, min_seq_len, max_seq_len, log_seq_len, min_train_size,
             max_train_size, gradient_clipping, recompute, save_temp_every):
    import os
    import re
    import runpy
    import subprocess
    import sys
    import threading
    import time
    import types

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # Set our device; do NOT init the process group — tabicl.train inits its own DDP group,
    # and a second init_process_group would error ("trying to initialize the default PG twice").
    torch.cuda.set_device(local_rank)

    def _p(*a):
        if rank == 0:
            print(*a, flush=True)

    _p(f"[launch] world={world} gpus={world} (RANK/LOCAL_RANK/WORLD_SIZE from @distributed)")

    os.makedirs(ckpt_dir, exist_ok=True)  # upstream writes wand_id.txt at Trainer init, before mkdir

    # --- INLINED wandb → MLflow shim (keeps this notebook standalone — no repo files needed) ----
    # tabicl.train does a module-level `import wandb` and uses exactly: wandb.init(...).id,
    # wandb.log(metrics, step=), wandb.finish(), wandb.define_metric(). We plant a fake `wandb`
    # module backed by the ambient MLflow run @distributed created. Faithful port of the repo's
    # verified wandb_mlflow_shim.py + MLflowLogger (attach-never-end + param chunking/sanitizing).
    # Installed on EVERY rank so `import wandb` succeeds everywhere; upstream only calls init/log
    # on rank 0, so only rank 0 actually writes to MLflow.
    def install_wandb_mlflow_shim():
        if getattr(sys.modules.get("wandb"), "__mlflow_shim__", False):
            return  # idempotent
        bad = re.compile(r"[^A-Za-z0-9_\-. /]")  # MLflow key charset

        class _Logger:
            def __init__(self):
                self.run_id = None

            def setup(self, config):
                import mlflow
                if not os.environ.get("MLFLOW_TRACKING_URI"):
                    mlflow.set_tracking_uri("databricks")
                # Attach precedence: ambient active run (the one @distributed nested) wins; else
                # start_run() resumes MLFLOW_RUN_ID if set, otherwise opens a fresh run. We never
                # end it (finish() is a no-op) so tabicl's fit-internal finish can't kill the run.
                active = mlflow.active_run()
                self.run_id = (active or mlflow.start_run()).info.run_id
                try:
                    from mlflow.utils.validation import (MAX_PARAM_VAL_LENGTH,
                                                         MAX_PARAMS_TAGS_PER_BATCH)
                except ImportError:
                    MAX_PARAM_VAL_LENGTH, MAX_PARAMS_TAGS_PER_BATCH = 500, 100
                cfg = config if isinstance(config, dict) else (vars(config) if config else {})
                clean = {bad.sub("_", str(k)): str(v)[:MAX_PARAM_VAL_LENGTH] for k, v in cfg.items()}
                items = list(clean.items())
                for i in range(0, len(items), MAX_PARAMS_TAGS_PER_BATCH):  # ~112 params > 100/batch
                    mlflow.log_params(dict(items[i:i + MAX_PARAMS_TAGS_PER_BATCH]))

            def log_metrics(self, metrics, step=None):
                import mlflow
                numeric = {bad.sub("_", str(k)): float(v) for k, v in metrics.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
                if numeric:
                    mlflow.log_metrics(numeric, step=step)

        class _ShimRun:
            def __init__(self, logger):
                self._logger = logger

            @property
            def id(self):  # persisted to checkpoint_dir/wand_id.txt; passed back as id= on resume
                return self._logger.run_id if self._logger else "wandb-disabled"

            def log(self, metrics, step=None):
                if self._logger:
                    self._logger.log_metrics(metrics, step=step)

            def finish(self):
                pass  # never end the ambient run — @distributed owns its lifecycle

        state = {"run": None}

        def _init(project=None, name=None, id=None, config=None, mode=None, **_ignored):  # noqa: A002
            if mode == "disabled":
                state["run"] = _ShimRun(None)
                return state["run"]
            logger = _Logger()
            logger.setup(config)
            state["run"] = _ShimRun(logger)
            return state["run"]

        def _log(metrics, step=None):
            if state["run"]:
                state["run"].log(metrics, step=step)

        def _finish():
            if state["run"]:
                state["run"].finish()

        mod = types.ModuleType("wandb")
        mod.__mlflow_shim__ = True
        mod.init = _init
        mod.log = _log
        mod.finish = _finish
        mod.define_metric = lambda *a, **k: None  # no-op; MLflow steps cover this
        mod.run = None
        sys.modules["wandb"] = mod

    install_wandb_mlflow_shim()

    # --- name this stage's MLflow run on rank 0 (separate run per stage, one experiment) --------
    # Start the run OURSELVES with a stage-specific name + tags, before tabicl's wandb.init fires;
    # the shim then attaches to this active run (never ends it), so the ~112 TrainConfig params and
    # per-step ce/accuracy/prior_time/train_time all land here. All stages share the notebook's
    # ambient experiment, so stage 1/2/3 show up as three comparable runs under it.
    owns_run = False
    if rank == 0:
        import mlflow
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("databricks")
        if mlflow.active_run() is None:
            mlflow.start_run(run_name=run_name)
            owns_run = True
        else:
            mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tags({"tabicl_stage": str(stage), "gpus": str(world),
                         "seq_len_max": str(max_seq_len), "micro_batch_size": str(micro_batch_size),
                         "prior_device": str(prior_device), "dtype": str(dtype),
                         "model_compile": str(model_compile), "muon": str(muon)})

    # --- rank-0 GPU-utilization sampler: the "is it going brrr" evidence -----------------------
    # nvidia-smi samples the whole node (all GPUs) so we only run it on rank 0. The AIR system-metrics
    # gauge is unreliable on short jobs (reads 0% — sampling artifact, see NOTES); this in-node poll is
    # the trustworthy signal.
    samples = []  # (t, gpu_index, util%, mem_used_MiB)
    stop = threading.Event()

    def _sampler():
        q = "index,utilization.gpu,memory.used"
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                t = time.time()
                for line in out.strip().splitlines():
                    idx, util, mem = (x.strip() for x in line.split(","))
                    samples.append((t, int(idx), float(util), float(mem)))
            except Exception:  # noqa: BLE001 — a sampling miss must never touch training
                pass
            stop.wait(2.0)

    sampler = None
    if rank == 0:
        sampler = threading.Thread(target=_sampler, daemon=True)
        sampler.start()

    # --- chaining: on the FIRST launch of a stage>1, load the previous stage's WEIGHTS only. ----
    # On a later launch this stage's own ckpt_dir already holds step-*.ckpt and tabicl auto-resumes
    # from it with FULL optimizer/scheduler state — so we must NOT pass --checkpoint_path then (it
    # would override the resume). All ranks compute this identically from the shared UC volume.
    def _latest_ckpt(d):
        if not d or not os.path.isdir(d):
            return None
        best = None
        for f in os.listdir(d):
            if f.startswith("step-") and f.endswith(".ckpt"):
                try:
                    n = int(f[len("step-"):-len(".ckpt")])
                except ValueError:
                    continue
                if best is None or n > best[0]:
                    best = (n, os.path.join(d, f))
        return best[1] if best else None

    resume_args = []
    if _latest_ckpt(ckpt_dir) is None and prev_ckpt_dir is not None:
        prev = _latest_ckpt(prev_ckpt_dir)
        if prev:
            resume_args = ["--checkpoint_path", prev, "--only_load_model", "True"]
            _p(f"[chain] stage {stage}: loading weights from {prev} (first launch)")
        else:
            _p(f"[chain] WARNING stage {stage}: no stage-{stage - 1} checkpoint under {prev_ckpt_dir} "
               f"— starting from RANDOM init. Run the earlier stage first for the real curriculum.")

    # --- build tabicl.train argv: stage-driven values on top, pinned cross-stage recipe below ---
    argv = [
        "tabicl.train",
        "--wandb_log", "True",          # upstream emits metrics only with this on; shim → MLflow
        "--device", "cuda",
        "--dtype", str(dtype),
        "--np_seed", "42", "--torch_seed", "42",
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--micro_batch_size", str(micro_batch_size),    # 4 (s1) / 1 (s2,s3)
        "--batch_size_per_gp", str(batch_size_per_gp),   # 4 (s1) / 1 (s2,s3)
        "--lr", str(lr),                                 # 8e-4 / 1e-4 / 2e-5 by stage
        "--muon", str(muon),
        "--model_compile", str(model_compile),           # torch.compile(dynamic=True) — kernel fusion
        "--n_jobs", str(n_jobs),                         # CPU prior-gen workers per rank
        "--prior_device", str(prior_device),
        "--use_flash_attn3", str(use_flash_attn3),       # stage 2/3 = True (needs FA3 in the env)
        "--gradient_clipping", str(gradient_clipping),   # 10.0 (s1,s2) / 1.0 (s3)
        "--max_features", str(max_features),
        "--max_classes", str(max_classes),
        "--max_seq_len", str(max_seq_len),               # 1024 / 10240 / 60000 by stage
        "--min_train_size", str(min_train_size),         # 0.3 (s1) / 0.79 (s2,s3)
        "--max_train_size", str(max_train_size),         # 0.9 (s1) / 0.81 (s2,s3)
        "--checkpoint_dir", ckpt_dir,
        "--save_temp_every", str(save_temp_every),
        "--save_perm_every", str(max_steps),             # one permanent ckpt at the end
        # --- pinned cross-stage recipe (identical in the stage 1/2/3 scripts) ------------------
        "--beta1", "0.9", "--weight_decay", "0.01", "--use_cautious_wd", "False",
        "--scheduler", "cosine_with_restarts", "--warmup_proportion", "0.01",
        "--cosine_num_cycles", "1", "--cosine_amplitude_decay", "1", "--cosine_lr_end", "1e-7",
        "--prior_type", "graph_scm", "--min_features", "1",
        "--seq_len_per_gp", "True", "--graph_noise", "False",
        "--filter_unpredictable_graphs", "True", "--filter_unpredictable_datasets", "True",
        "--allow_act_warping", "False", "--min_n_nodes", "2", "--max_n_nodes", "32",
        "--cauchy_dag_offset", "0.0",
        "--embed_dim", "128",
        "--col_num_blocks", "3", "--col_nhead", "8", "--col_num_inds", "128",
        "--col_affine", "False", "--col_feature_group", "same", "--col_feature_group_size", "3",
        "--col_target_aware", "True", "--col_ssmax", "True",
        "--row_num_blocks", "3", "--row_nhead", "8", "--row_num_cls", "4",
        "--row_rope_base", "100000", "--row_rope_interleaved", "False",
        "--icl_num_blocks", "12", "--icl_nhead", "8", "--icl_ssmax", "True",
        "--ssmax_type", "qassmax-mlp-elementwise",
        "--ff_factor", "2", "--norm_first", "True", "--zero_init", "False",
    ]
    # stage-conditional args (only stages 2/3 set these; stage 1 uses upstream defaults)
    if min_seq_len is not None:
        argv += ["--min_seq_len", str(min_seq_len)]
    if log_seq_len:
        argv += ["--log_seq_len", "True"]
    if recompute is not None:
        argv += ["--recompute", str(recompute)]   # stage 3 = False (grad-ckpt off) unless OOM override
    argv += resume_args
    sys.argv = argv
    _p(f"[launch] stage {stage} argv head:", " ".join(argv[1:28]))

    # --- run the real trainer (blocks until max_steps; all ranks participate in DDP) -----------
    t0 = time.time()
    train_error = None
    try:
        runpy.run_module("tabicl.train", run_name="__main__", alter_sys=True)
    except Exception as e:  # noqa: BLE001 — capture so rank 0 can still emit the report
        train_error = e
    wall = time.time() - t0

    # --- stop the sampler and summarize (rank 0 only) ------------------------------------------
    if rank == 0 and sampler is not None:
        stop.set()
        sampler.join(timeout=5)

    if rank != 0:
        # Non-zero ranks: surface a failure, otherwise stay quiet (rank 0 owns the report).
        if train_error is not None:
            raise train_error
        return None

    # ---- rank 0: THROUGHPUT is the headline; GPU util is context (shape-dependent) ----
    per_gpu = {}
    for _, idx, util, mem in samples:
        per_gpu.setdefault(idx, []).append((util, mem))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    all_util = [u for _, _, u, _ in samples]
    rank0_util = _mean([u for _, i, u, _ in samples if i == 0])            # GPU 0 = rank 0 (serial logging/ckpt)
    worker_util = _mean([u for _, i, u, _ in samples if i != 0])          # GPUs 1..N = pure training
    node_util = _mean(all_util)
    pegged = 100.0 * sum(1 for u in all_util if u >= 80) / len(all_util) if all_util else 0.0
    steps_per_s = max_steps / wall if wall > 0 else 0.0
    datasets_per_s = batch_size * steps_per_s   # batch_size = datasets/optimizer-step (global)

    print("\n" + "=" * 78)
    print(f"STAGE {stage} RESULT — THROUGHPUT (the metric that matters), then GPU util (context)")
    print("=" * 78)
    print(f"wall {wall:.0f}s for {max_steps} steps  →  {steps_per_s:.3f} steps/s, "
          f"{datasets_per_s:,.0f} datasets/s across {world} GPUs")
    print("  (includes cold-start + prior warmup; read steady-state s/it, prior_time, train_time in MLflow)")
    print("-" * 78)
    print(f"{'GPU':>3} {'samples':>8} {'mean_util%':>11} {'max_util%':>10} {'mean_mem_GiB':>13} {'max_mem_GiB':>12}")
    for idx in sorted(per_gpu):
        us = [u for u, _ in per_gpu[idx]]
        ms = [m / 1024.0 for _, m in per_gpu[idx]]
        tag = " (rank0)" if idx == 0 else ""
        print(f"{idx:>3} {len(us):>8} {_mean(us):>11.1f} {max(us):>10.0f} "
              f"{_mean(ms):>13.1f} {max(ms):>12.1f}{tag}")
    print("-" * 78)
    print(f"GPU util — workers (GPU 1..{world - 1}): {worker_util:.1f}%   rank-0 (GPU 0): {rank0_util:.1f}%   "
          f"node: {node_util:.1f}%   (≥80%: {pegged:.0f}% of samples)")
    print("  rank 0 reads lower because it carries the serial MLflow logging + checkpoint I/O while the")
    print("  workers spin-wait at the all-reduce (NCCL spin counts as 100% util — so 'workers pegged,")
    print("  rank-0 low' is the healthy signature, not imbalance).")
    if stage == 1:
        print("  NOTE stage 1 (≤1,024-row tables, 27M-param model) is host-bound — low util is EXPECTED;")
        print("       optimize steps/s, not util. Utilization climbs at stage 2/3 (10k/60k-row attention).")
    else:
        print(f"  stage {stage}: large in-context sequences (≤{max_seq_len} rows) → attention FLOPs should"
              " actually keep the GPUs busy; low util here would be a real finding worth chasing.")

    # ---- log the throughput/util metrics to THIS stage's MLflow run ----
    try:
        import mlflow
        if mlflow.active_run() is not None:
            mlflow.log_metrics({"steps_per_s": steps_per_s, "datasets_per_s": datasets_per_s,
                                "wall_s": wall, "gpu_util_workers": worker_util,
                                "gpu_util_rank0": rank0_util, "gpu_util_node": node_util,
                                "gpu_pct_pegged": pegged})
    except Exception as e:  # noqa: BLE001
        _p(f"(metric logging skipped: {type(e).__name__}: {e})")

    # ---- rank 0: acceptance report + sentinel ----
    ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")) if os.path.isdir(ckpt_dir) else []
    print("\n" + "=" * 78)
    print("ACCEPTANCE REPORT")
    print("=" * 78)
    c1 = world == 8
    c2 = train_error is None
    c3 = len(ckpts) > 0
    c4 = len(all_util) > 0   # util is MEASURED (not a pass bar — it's shape-dependent, see above)
    print(f"CHECK 1  launch = full 8×H100 node ....... {'PASS' if c1 else 'FAIL'} (world={world})")
    print(f"CHECK 2  stage {stage} ran to completion ...... {'PASS' if c2 else 'FAIL'}"
          + ("" if c2 else f" ({type(train_error).__name__}: {train_error})"))
    print(f"CHECK 3  checkpoints written ............. {'PASS' if c3 else 'FAIL'} ({ckpts})")
    print(f"CHECK 4  GPU utilization measured ........ {'PASS' if c4 else 'FAIL'} "
          f"({len(all_util)} samples; workers {worker_util:.0f}% — context, not a bar)")
    verdict_ok = c1 and c2 and c3   # util is reported, never gates the verdict
    print("-" * 78)
    if verdict_ok:
        print(f"VERDICT: ACCEPTED — TabICLv2 stage {stage} on {world}×H100: "
              f"{steps_per_s:.3f} steps/s, workers {worker_util:.0f}% util, {len(ckpts)} ckpt(s).")
        print(f"TABICL_8XH100_PRETRAIN_OK stage={stage} world={world} steps={max_steps} "
              f"steps_per_s={steps_per_s:.3f} datasets_per_s={datasets_per_s:.0f} "
              f"gpu_util_workers={worker_util:.1f} gpu_util_rank0={rank0_util:.1f} "
              f"seq_len_max={max_seq_len} wall_s={wall:.0f} ckpts={len(ckpts)}")
    else:
        print("VERDICT: NOT ACCEPTED — read the failing CHECK above.")

    # close THIS stage's run so the next stage is a distinct run (only if we opened it)
    try:
        import mlflow
        if verdict_ok:
            mlflow.set_tag("verdict", "ACCEPTED")
        if owns_run and mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:  # noqa: BLE001
        pass

    if train_error is not None:
        raise train_error
    if not verdict_ok:
        raise RuntimeError("one or more acceptance checks failed — see the ACCEPTANCE REPORT above")

    return {"stage": stage, "world": world, "steps": max_steps, "steps_per_s": steps_per_s,
            "datasets_per_s": datasets_per_s, "gpu_util_workers": worker_util,
            "gpu_util_rank0": rank0_util, "wall_s": wall, "checkpoints": ckpts}


print("pretrain defined and decorated with @distributed(gpus=%d)" % GPUS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch — `pretrain.distributed(...)`
# MAGIC Fans the function out one process per GPU. This launch is **one MLflow run for the selected
# MAGIC `stage`** (named `tabicl-clf-stage{N}-8xh100`, tagged `tabicl_stage=N`), under the notebook's
# MAGIC experiment. To run the full curriculum: set `stage=1`, run this cell; then `stage=2`, run again
# MAGIC (it auto-loads stage-1's weights from the volume); then `stage=3`. Three comparable runs, one
# MAGIC experiment. `tabicl.train`'s tqdm streams below; rank 0 prints throughput, util, and the sentinel.

# COMMAND ----------

result = pretrain.distributed(**CFG)
print("\nresult:", result)
if isinstance(result, dict):
    print(f"\n✅ Stage {result['stage']} on {result['world']}×H100 — {result['steps_per_s']:.3f} steps/s "
          f"({result['datasets_per_s']:,.0f} datasets/s), workers {result['gpu_util_workers']:.0f}% util "
          f"(rank0 {result['gpu_util_rank0']:.0f}%), {result['wall_s']:.0f}s, "
          f"{len(result['checkpoints'])} checkpoint(s). Bump the `stage` widget for the next stage.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the result — and where GPU utilization actually comes from
# MAGIC
# MAGIC **The headline is throughput, not utilization.** `steps/s` and `datasets/s` are what move
# MAGIC wall-clock; GPU-util % is a *consequence* of the stage's shape, not a dial you turn.
# MAGIC
# MAGIC **Utilization by stage (this is the whole point of adding 2 & 3):**
# MAGIC - **Stage 1 (≤1,024-row tables)** — a 27M-param model over short in-context sequences finishes
# MAGIC   each step in milliseconds, then waits on CPU prior-gen + Muon's host-side sync. **Low GPU util
# MAGIC   is expected and correct here** — 8×H100 is over-provisioned for stage 1. The win is data-parallel
# MAGIC   throughput (more datasets/step across 8 GPUs), not a pegged gauge.
# MAGIC - **Stage 2 (→10,240 rows)** / **Stage 3 (→60,000 rows)** — attention is **quadratic in rows**, so
# MAGIC   the per-step GPU work explodes and the cards actually saturate. Stage 3 at 60k rows / fp32 is the
# MAGIC   memory-stress case (the "does it need B300?" question); if it OOMs, flip `stage3_recompute` on.
# MAGIC
# MAGIC **Rank-0 vs workers:** GPU 0 (rank 0) reads lower because it does the serial MLflow logging +
# MAGIC checkpoint I/O while GPUs 1–7 spin-wait at the all-reduce — and NCCL's spin counts as 100% util.
# MAGIC So "workers pegged, rank-0 low" is the healthy signature, and the **workers** number is the honest
# MAGIC utilization for this shape.
# MAGIC
# MAGIC **Per-step breakdown lives in MLflow:** `prior_time` / `train_time` / `ce` / `accuracy`, per stage
# MAGIC run (via the inlined shim). That's where you separate CPU-gen cost from GPU-compute cost.
# MAGIC
# MAGIC **Speed levers when the step is compute-bound at low util (the stage-1 case):** the model runs
# MAGIC *effective fp32* out of the box (`--amp` on but `--dtype float32` = no-op autocast), so H100 tensor
# MAGIC cores idle. In order of leverage:
# MAGIC 1. **`dtype=bfloat16`** — engages the tensor cores; bf16 keeps fp32 dynamic range (no grad-scaler)
# MAGIC    and fp32 master weights, so it's a mild, standard mixed-precision change. Validate the loss
# MAGIC    curve tracks your fp32 run for a few hundred steps before committing a week to it.
# MAGIC 2. **`model_compile=True`** — `torch.compile(dynamic=True)` fuses the many small col/row/icl
# MAGIC    kernels and removes launch-overhead gaps; strongest *combined* with bfloat16. One-time warmup.
# MAGIC
# MAGIC Run three tagged runs — fp32 / bf16 / bf16+compile — and compare `steps_per_s` across them (all
# MAGIC under this experiment). TF32 is already on, so it's not a lever. `prior_device=cuda` is moot here
# MAGIC (prior-gen is <0.001s) and hangs under DDP — leave it `cpu`.
# MAGIC
# MAGIC **Scaling past one node (16+ GPUs):** `@distributed` is single-node. Multi-node is `torchrun` + the
# MAGIC AIR CLI (`workloads/`), where the repo measured **97.7% weak-scaling efficiency 8→16 GPU** over
# MAGIC EFA/GDRDMA — the path if stage-3 needs more than one node.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Status
# MAGIC **Stage 1 ran green on 8×H100** (`@distributed` launch, inlined shim → MLflow, real `tabicl.train`);
# MAGIC low GPU util at stage 1 is the workload, not a defect (see the result cell). **Stages 2 & 3 are
# MAGIC pre-registered, not yet run** — recipe deltas are verbatim from upstream
# MAGIC `scripts/train_v2_clf_stage{2,3}.sh`. First-run risks for 2/3:
# MAGIC 1. **FlashAttention-3** must be present in the AI env (both stages pass `--use_flash_attn3 True`).
# MAGIC    The env-check cell reports whether it's found; if not, install/vendor it first.
# MAGIC 2. **Stage 3 memory** — 60,000-row attention at fp32 is the OOM/"needs-B300" case; if it OOMs, set
# MAGIC    the `stage3_recompute` widget to on (gradient checkpointing).
# MAGIC 3. **Chaining** — stage 2/3 load the prior stage's weights only on first launch; confirm the
# MAGIC    `[chain] loading weights from …` line appears (not the RANDOM-init warning).
# MAGIC
# MAGIC Record results per stage in `experiments/foundation-models/NOTES.md` (each stage is its own MLflow
# MAGIC run under the notebook's experiment, tagged `tabicl_stage=N`).
