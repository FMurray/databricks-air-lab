# Databricks notebook source
# MAGIC %md
# MAGIC # UAT check: reserved-pool readiness (one-notebook entrypoint)
# MAGIC Answers "are the 20 dedicated 8xH100 nodes ready?" in one Run-all, using only the
# MAGIC validated CLI-from-notebook path (UAT #17): vendored `air` CLI + context-token auth,
# MAGIC submissions from the FUSE-mirrored repo, verification by MLflow **receipts** (never
# MAGIC `--dry-run` — air 1.0.0 dry-run validates config only and would be a false green).
# MAGIC
# MAGIC Phase A — per-node burn sweep: `pool_nodes` × (1 node, 8xH100) gpu-burn submissions
# MAGIC in parallel; receipts prove burn=PASS, 8 GPUs each, and **distinct nodes by GPU UUID**.
# MAGIC Phase B — fabric probe: one ctypes-NCCL all-reduce across `fabric_nodes` nodes
# MAGIC (sentinel + busbw; 0 skips).
# MAGIC
# MAGIC Shape: **CPU** (this notebook only submits and verifies). Cost gate: `confirm_pool`
# MAGIC must be `yes` — the default run is a no-cost SKIP so a naive Run-all can't take the
# MAGIC pool during a shared window. Coordinate in the team channel before a full sweep.

# COMMAND ----------

# MAGIC %pip install --quiet --no-index --find-links /Workspace/Shared/databricks-air-lab/uat/wheels databricks-air

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("confirm_pool", "no", "confirm_pool: yes = really take the pool (coordinate first)")
dbutils.widgets.text("pool_nodes", "20", "pool_nodes: burn submissions (1 node x 8xH100 each)")
dbutils.widgets.text("burn_seconds", "300", "burn_seconds per node (A1 acceptance used 900)")
dbutils.widgets.text("fabric_nodes", "2", "fabric_nodes: NCCL all-reduce span (0 = skip phase B)")

import json, os, re, shutil, subprocess, sys, time

REPO = "/Workspace/Shared/databricks-air-lab"
BURN_YAML = "workloads/gpu-burn.example.yaml"
FABRIC_YAML = "workloads/nccl-allreduce-v5.example.yaml"
BURN_EXPERIMENT = "air-lab-gpu-burn"
FABRIC_EXPERIMENT = "air-lab-nccl-allreduce"

results, summary = {}, {}
def record(name, ok, detail=""):
    results[name] = {"ok": (None if ok is None else bool(ok)), "detail": str(detail)[:300]}
    print({True: "✅", False: "❌", None: "⏭️"}[results[name]["ok"]], name, "—", str(detail)[:300])

CONFIRM = dbutils.widgets.get("confirm_pool").strip().lower() == "yes"
POOL_NODES = int(dbutils.widgets.get("pool_nodes"))
BURN_SECONDS = int(dbutils.widgets.get("burn_seconds"))
FABRIC_NODES = int(dbutils.widgets.get("fabric_nodes"))

if not CONFIRM:
    record("pool_readiness", None,
           "SKIPPED — set confirm_pool=yes to run. This check submits up to "
           f"{POOL_NODES}x(8xH100) burns + a {FABRIC_NODES}-node fabric probe: pool-scale, "
           "coordinate in the team channel first.")
    dbutils.notebook.exit(json.dumps({"check": "pool-readiness", "results": results}))

# COMMAND ----------

# auth + CLI (the #17-validated path)
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
AIR = shutil.which("air") or os.path.join(os.path.dirname(sys.executable), "air")
v = subprocess.run([AIR, "--version"], capture_output=True, text=True, timeout=120)
record("cli_ready", v.returncode == 0, (v.stdout.strip().splitlines() or ["?"])[-1][:80])

from databricks.sdk import WorkspaceClient  # ships with the vendored CLI
w = WorkspaceClient()

def submit(yaml_path, overrides=()):
    cmd = [AIR, "run", "--json", "-f", yaml_path]
    if overrides:
        cmd += ["--override", *overrides]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    m = re.search(r'"?run_id"?\D*(\d{8,})', r.stdout + r.stderr)
    return (m.group(1) if m else None), (r.stdout + r.stderr)[-200:]

def run_state(run_id):
    rr = w.api_client.do("GET", f"/api/2.2/jobs/runs/get?run_id={run_id}")
    s = rr.get("state", {})
    return s.get("life_cycle_state"), s.get("result_state"), s.get("state_message", "")

def poll_all(run_ids, deadline_s):
    states = {rid: (None, None, "") for rid in run_ids}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        pending = [rid for rid, (lc, _, _) in states.items()
                   if lc not in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED")]
        if not pending:
            break
        for rid in pending:
            states[rid] = run_state(rid)
        done = sum(1 for lc, _, _ in states.values() if lc in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"))
        print(f"{time.strftime('%H:%M:%S')} terminal {done}/{len(run_ids)}")
        if done < len(run_ids):
            time.sleep(45)
    return states

def mlflow_receipt(experiment, job_run_id):
    """Receipts, not run status: admission-refused jobs leave MLflow runs stuck RUNNING
    with empty params (docs/06 / node-acceptance NOTES)."""
    import mlflow
    from mlflow.tracking import MlflowClient
    c = MlflowClient()
    exp = c.get_experiment_by_name(f"/Users/{ctx.userName().get()}/{experiment}") \
        or c.get_experiment_by_name(experiment)
    if exp is None:
        return None
    hits = c.search_runs([exp.experiment_id],
                         f"tags.`mlflow.databricks.jobRunID` = '{job_run_id}'", max_results=1)
    return hits[0] if hits else None

# COMMAND ----------

# Phase A: burn sweep — submit POOL_NODES single-node burns (sequential submits ~7s each,
# runs execute in parallel; the per-shape node quota admits up to its max, refusals are
# classified below rather than failing the whole check opaquely)
burn_ids, burn_notes = [], {}
for i in range(POOL_NODES):
    rid, tail = submit(BURN_YAML, (
        "compute.accelerator_type=GPU_8xH100", "compute.num_accelerators=8",
        "env_variables.EXPECT_GPUS=8", f"env_variables.BURN_SECONDS={BURN_SECONDS}"))
    if rid:
        burn_ids.append(rid)
    else:
        burn_notes[f"submit_{i}"] = tail
    print(f"submit {i+1}/{POOL_NODES}: {rid or 'FAILED'}")
record("burn_submitted", len(burn_ids) == POOL_NODES,
       f"{len(burn_ids)}/{POOL_NODES} submissions accepted {burn_notes or ''}")

states = poll_all(burn_ids, deadline_s=BURN_SECONDS + 1500)
succ = [rid for rid, (lc, rs, _) in states.items() if rs == "SUCCESS"]
quota_refused = [rid for rid, (_, rs, msg) in states.items()
                 if rs not in ("SUCCESS", None) and "GPU quota" in msg]
other_failed = [rid for rid, (lc, rs, msg) in states.items()
                if rs not in ("SUCCESS", None) and "GPU quota" not in msg]
record("burn_runs_terminal", True,
       f"SUCCESS={len(succ)} quota_refused={len(quota_refused)} other_failed={len(other_failed)}")

# COMMAND ----------

# Phase A verdict from receipts: burn=PASS on every SUCCESS run + node distinctness by UUID
uuid_sets, passes = [], 0
for rid in succ:
    r = mlflow_receipt(BURN_EXPERIMENT, rid)
    p = r.data.params if r else {}
    if p.get("burn") == "PASS" and p.get("gpus_visible") == "8":
        passes += 1
        uuid_sets.append(frozenset(p["gpu_uuids"].split(",")))
    else:
        record(f"receipt_{rid}", False, f"params={dict(list(p.items())[:4])}")
all_uuids = set().union(*uuid_sets) if uuid_sets else set()
distinct_nodes = len(all_uuids) // 8
summary.update(nodes_requested=POOL_NODES, nodes_pass=passes, distinct_nodes=distinct_nodes,
               quota_refusals=len(quota_refused), other_failures=len(other_failed))
record("burn_all_pass", passes == len(succ) and len(succ) > 0,
       f"{passes} PASS receipts / {len(succ)} SUCCESS runs")
record("nodes_distinct", distinct_nodes == passes,
       f"{len(all_uuids)} distinct GPU UUIDs -> {distinct_nodes} distinct nodes "
       f"(reuse means quota < requested or scheduler re-pin)")
record("pool_coverage", distinct_nodes >= POOL_NODES,
       f"{distinct_nodes}/{POOL_NODES} pool nodes touched this sweep")

# COMMAND ----------

# Phase B: fabric probe — one ctypes-NCCL all-reduce spanning FABRIC_NODES nodes
if FABRIC_NODES < 2:
    record("fabric_probe", None, "SKIPPED — fabric_nodes < 2")
else:
    frid, tail = submit(FABRIC_YAML, (
        "compute.accelerator_type=GPU_8xH100", f"compute.num_accelerators={FABRIC_NODES * 8}"))
    if not frid:
        record("fabric_probe", False, f"submit failed: {tail}")
    else:
        fstates = poll_all([frid], deadline_s=2100)
        _, rs, msg = fstates[frid]
        r = mlflow_receipt(FABRIC_EXPERIMENT, frid)
        p = r.data.params if r else {}
        m = r.data.metrics if r else {}
        ok = rs == "SUCCESS" and p.get("probe_sentinel") == "MULTINODE_NCCL_V5_OK" \
            and p.get("world_size") == str(FABRIC_NODES * 8)
        busbw = m.get("busbw_gbps")
        summary.update(fabric_nodes=FABRIC_NODES, fabric_busbw_gbps=busbw)
        record("fabric_probe", ok,
               f"run {frid} -> {rs}; sentinel={p.get('probe_sentinel')} "
               f"world={p.get('world_size')} busbw={busbw} GB/s (smoke-grade) {msg[:80]}")

# COMMAND ----------

pool_ready = all(r["ok"] for n, r in results.items()
                 if r["ok"] is not None and n not in ("burn_runs_terminal",))
summary["pool_ready"] = pool_ready
print(("✅ POOL READY " if pool_ready else "❌ NOT READY ") + json.dumps(summary))
dbutils.notebook.exit(json.dumps({"check": "pool-readiness", "results": results,
                                  "summary": summary}))
