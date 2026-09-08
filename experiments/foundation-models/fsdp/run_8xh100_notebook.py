# Databricks notebook source
# MAGIC %md
# MAGIC # FSDP on 8×H100 from a notebook — *without* the `@distributed` decorator
# MAGIC
# MAGIC This notebook drives an **8-GPU FSDP2 training job on a single 8×H100 node** using plain
# MAGIC `torch.distributed`, launched by **`torchrun`** — the exact elastic-launch path you already use
# MAGIC on MLR / bare metal. We are **deliberately not using `serverless_gpu`'s `@distributed`**
# MAGIC decorator: it's a Beta convenience layer and is single-node-only anyway, so raw `torchrun` is
# MAGIC both the portable option and the one whose behavior transfers straight to the multi-node CLI
# MAGIC path. **Your existing torchrun-based training code runs here unchanged.**
# MAGIC
# MAGIC **What you get out of it:** a real per-GPU throughput number (tokens/s/GPU) on one node. That
# MAGIC number is the **baseline** for the 2-node CLI run (`workloads/fsdp-scaling.example.yaml`), where
# MAGIC the ratio between the two is the **weak-scaling efficiency** — i.e. what crossing the node
# MAGIC boundary onto the EFA fabric actually costs you.
# MAGIC
# MAGIC ### How to run
# MAGIC 1. Attach this notebook to an **AI base environment (v5)** with a **`GPU_8xH100`** accelerator.
# MAGIC    (v4 silently breaks GPU-job storage egress — use v5.)
# MAGIC 2. Make sure the repo is synced to the workspace (default path below is the `/Shared` mirror).
# MAGIC 3. Run all cells top to bottom. The launch cell prints an acceptance report ending in a
# MAGIC    `VERDICT:` line and an `FSDP_THROUGHPUT …` data line.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC `repo_root` is where the repo is synced in the workspace; the benchmark script is a sibling of
# MAGIC this notebook. `nproc` should equal the node's GPU count (8 on `GPU_8xH100`). Shape defaults to
# MAGIC the ~1.6B-param benchmark model — big enough that the GPUs are busy and FSDP's collectives are a
# MAGIC real fraction of each step (a toy model would measure launch overhead, not scaling).

# COMMAND ----------

dbutils.widgets.text("repo_root", "/Workspace/Shared/databricks-air-lab", "repo root in workspace")
dbutils.widgets.text("nproc", "8", "GPUs on this node (nproc_per_node)")
dbutils.widgets.text("layers", "16", "transformer layers")
dbutils.widgets.text("dim", "2048", "model dim")
dbutils.widgets.text("heads", "16", "attention heads")
dbutils.widgets.text("seq", "1024", "sequence length")
dbutils.widgets.text("batch", "8", "PER-GPU batch")
dbutils.widgets.text("warmup", "10", "untimed warmup steps")
dbutils.widgets.text("iters", "40", "timed steps")

import os

REPO_ROOT = dbutils.widgets.get("repo_root")
NPROC = int(dbutils.widgets.get("nproc"))
SCRIPT = os.path.join(REPO_ROOT, "experiments/foundation-models/fsdp/bench_fsdp_scaling.py")

# Fallback: derive the script path from THIS notebook's location (script is a sibling), in case
# the repo is synced somewhere other than the default /Shared mirror.
if not os.path.exists(SCRIPT):
    try:
        nb = (dbutils.notebook.entry_point.getDbutils().notebook()
              .getContext().notebookPath().get())
        cand = "/Workspace" + os.path.join(os.path.dirname(nb), "bench_fsdp_scaling.py")
        if os.path.exists(cand):
            SCRIPT = cand
    except Exception as e:  # noqa: BLE001
        print("notebook-relative fallback unavailable:", e)

print("Script:", SCRIPT, "exists =", os.path.exists(SCRIPT))
assert os.path.exists(SCRIPT), (
    f"benchmark script not found at {SCRIPT} — set the repo_root widget to where the repo is "
    "synced in this workspace")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment check
# MAGIC Confirm we're on an 8-GPU H100 node with an FSDP2-capable runtime before spending the step time.

# COMMAND ----------

import torch
from torch.distributed.fsdp import fully_shard

n_gpu = torch.cuda.device_count()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  nccl {torch.cuda.nccl.version()}")
print(f"visible GPUs: {n_gpu}")
for i in range(n_gpu):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
print("fully_shard (FSDP2) available:", callable(fully_shard))

assert n_gpu == NPROC, (
    f"expected {NPROC} GPUs but see {n_gpu} — attach a GPU_8xH100 accelerator (or set nproc)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch: `torchrun --standalone --nproc_per_node=8`
# MAGIC `--standalone` sets up single-node rendezvous on localhost (no MASTER_ADDR needed); `torchrun`
# MAGIC then spawns one rank per GPU. We invoke it as `python -m torch.distributed.run` so it resolves
# MAGIC against this notebook's own interpreter regardless of PATH. Output (including the acceptance
# MAGIC report) streams below.

# COMMAND ----------

import sys
import subprocess

cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--standalone", f"--nproc_per_node={NPROC}",
    SCRIPT,
    "--layers", dbutils.widgets.get("layers"),
    "--dim", dbutils.widgets.get("dim"),
    "--heads", dbutils.widgets.get("heads"),
    "--seq", dbutils.widgets.get("seq"),
    "--batch", dbutils.widgets.get("batch"),
    "--warmup", dbutils.widgets.get("warmup"),
    "--iters", dbutils.widgets.get("iters"),
    "--mp",
]
print("Launching:\n ", " ".join(cmd), "\n" + "=" * 90)

proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
if proc.stderr:
    print("----- stderr (torchrun / NCCL) -----")
    print(proc.stderr[-4000:])
print("=" * 90, "\nexit code:", proc.returncode)
assert proc.returncode == 0, "benchmark did not exit 0 — read the acceptance report / stderr above"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the baseline, and the command to scale to 2 nodes
# MAGIC The `FSDP_THROUGHPUT` line carries `per_gpu_tokens_per_s` — this single-node figure is the
# MAGIC baseline. Feed it to the CLI multi-node run so its report computes the weak-scaling efficiency
# MAGIC directly. (Multi-node is **CLI-only** — the notebook decorator path is single-node regardless.)

# COMMAND ----------

import re

m = re.search(r"per_gpu_tokens_per_s=(\d+(?:\.\d+)?)", proc.stdout)
per_gpu = float(m.group(1)) if m else None
if per_gpu is None:
    print("Could not parse per_gpu_tokens_per_s from output — check the FSDP_THROUGHPUT line above.")
else:
    print(f"Single-node (world={NPROC}) baseline: {per_gpu:,.0f} tokens/s/GPU\n")
    print("Now run the 2-node scaling job from the CLI (16 GPUs), passing this baseline so the")
    print("run reports weak-scaling efficiency = per-GPU@16 / per-GPU@8 (the EFA node-crossing tax):\n")
    print(f"  air run -f workloads/fsdp-scaling.example.yaml -p <profile> \\")
    print(f"    --override command='… bench_fsdp_scaling.py … --baseline-per-gpu-tps {per_gpu:.0f}'\n")
    print("See workloads/fsdp-scaling.example.yaml — it documents the exact 1-node vs 2-node series.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this demonstrated
# MAGIC - **8 GPUs driven from a notebook with plain `torchrun`** — no `@distributed`, no code changes
# MAGIC   to a standard torchrun training script; FSDP2 shards params/grads/optimizer across the node.
# MAGIC - A **defensible per-GPU throughput** number (warmup excluded, slowest-rank-bounded), logged to
# MAGIC   the run's MLflow so it outlives the notebook session.
# MAGIC - The **hand-off to multi-node**: same script, same launch semantics, scaled out via the CLI —
# MAGIC   where the efficiency number tells you whether adding nodes is worth it for *this* model shape.
