"""FSDP2 throughput + weak-scaling benchmark on AIR — the scaling-numbers workload.

Sibling to `train_fsdp.py`. That trainer proves FSDP is *correct* (sharding / reduce-scatter /
convergence / checkpoint) on a deliberately tiny shape; it says nothing about *speed*. This
benchmark answers the question a GPU power user actually asks: **how fast, and does it scale when
I cross the node boundary?** It is deliberately self-contained (copies the model + renderer, no
sibling import) and egress-free (`dependencies: []`, random on-device data) so it runs the same
way from a notebook and from the CLI.

WHAT IT MEASURES — weak scaling (fixed per-GPU batch; global batch grows with the world):
  * steady-state step time (warmup steps skipped), taken as the MAX across ranks — a step is not
    finished until the slowest rank's collectives complete, so the slow rank bounds throughput.
  * tokens/s (global) and tokens/s/GPU. Under ideal weak scaling tokens/s/GPU is CONSTANT as the
    world grows; the drop from 8 GPUs (1 node, NVLink only) to 16 GPUs (2 nodes, adds the EFA
    inter-node hop) IS the fabric tax — the headline scaling number.
  * TFLOP/s/GPU (DERIVED: ~6·N_params·tokens/s/GPU), a compute-intensity sanity figure.

HOW THE SCALING NUMBER IS FORMED — run the SAME script at two shapes via the SAME launch path:
  1. 1 node  (world=8)  → records tokens/s/GPU. This is the baseline.
  2. 2 nodes (world=16) → given `--baseline-per-gpu-tps <that number>`, reports weak-scaling
     efficiency = per_gpu@16 / per_gpu@8 and PASS/FAILs it against `--min-efficiency`.
Comparing notebook@8 vs CLI@16 would conflate the launch path with the scale; the defensible
series is CLI@8 vs CLI@16 (see workloads/fsdp-scaling.example.yaml). The NOTEBOOK run is the
"drive 8 GPUs without @distributed" proof and also emits an 8-GPU number to seed the baseline.

NO `@distributed` DECORATOR anywhere: this is raw `torch.distributed` launched by `torchrun`
(RANK / WORLD_SIZE / LOCAL_RANK from the env). The same code runs under torchrun on MLR, in a
notebook `%sh`/subprocess cell, and on the AIR CLI multi-node path — @distributed is the
notebook-only convenience layer we are deliberately not using.

Pre-flight (single-host, CPU/gloo, tiny shape — proves it launches + emits numbers, NOT a
throughput figure): `python3 bench_fsdp_scaling.py --local`
On AIR it is launched by torchrun (see workloads/fsdp-scaling.example.yaml).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback as _tb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


# ==========================================================================================
# Acceptance report — CANONICAL renderer copied verbatim from the acceptance-report skill
# (.claude/skills/acceptance-report/references/renderer.py). COPIED, never imported: each AIR
# YAML snapshots only its own experiment dir, so a shared-module import vanishes at runtime.
# Everything below WORKLOAD is byte-identical to that source of truth.
# ==========================================================================================
WORKLOAD = "FSDP THROUGHPUT / WEAK SCALING"

# Status enum — exactly these five (see format spec §"Status enum").
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
NA = "N/A-at-this-scale"

import signal          # noqa: E402 — kept next to _receipt, part of the copied renderer block
import textwrap        # noqa: E402
from dataclasses import dataclass   # noqa: E402
from datetime import datetime, timezone   # noqa: E402


@dataclass
class Check:
    """One acceptance check. `status` is one of the five enum values; `traceback` is retained
    (never swallowed) and fenced under the verdict when the run has any FAIL."""
    name: str
    status: str
    measured: str
    threshold: str
    what_why: str
    sufficient: str
    likely_means: str = ""
    traceback: str = ""


def _fail_from_exc(name, threshold, what_why, likely_means, exc) -> Check:
    """Turn an exception into a FAIL record (principle 1: record, don't re-raise) so the report
    still renders and the verdict/exit code can be derived from it. Trace is kept verbatim."""
    return Check(name=name, status=FAIL, measured=f"raised {type(exc).__name__}: {exc}",
                 threshold=threshold, what_why=what_why,
                 sufficient="A raised exception means the property could not be established.",
                 likely_means=likely_means, traceback="".join(_tb.format_exception(exc)))


def _wrap(text: str, indent: str = "               ") -> str:
    """Wrap a long field to ~92 cols, hanging-indented under its dotted label."""
    return textwrap.fill(text, width=96, initial_indent="", subsequent_indent=indent)


def _receipt(checks: "list[Check]", verdict: str, exit_code: int, test_id: str) -> None:
    """Dual-sink the verdict into MLflow params (the durable leg). stdout is the report's
    primary sink but it depends on the env's log delivery and expires with job-run retention
    (format spec §"Preconditions") — the receipt makes an absent stdout report disambiguable:
    receipt present = logs didn't ship; receipt absent = the run died before the verdict.
    Client API bound to MLFLOW_RUN_ID (never start_run — resuming the launcher-owned run
    fails silently on the job plane); alarm-guarded so a blocked tracking call can't hang
    the run; skips cleanly when MLFLOW_RUN_ID is unset (local)."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.log_param(run_id, "acceptance_verdict", verdict)
        client.log_param(run_id, "acceptance_exit", exit_code)
        if test_id:
            client.log_param(run_id, "acceptance_test_id", test_id)
        for i, c in enumerate(checks, 1):
            client.log_param(run_id, f"acceptance_check_{i}", f"{c.status} — {c.name}"[:490])
    except Exception as e:                                 # noqa: BLE001 — receipt is best-effort
        print(f"acceptance receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def render_report(checks: "list[Check]", run_id: str, profile: str, shape: str,
                  scope: str, runtime: str, sentinels: str, test_id: str = "") -> int:
    """Render every check identically and DERIVE the exit code last. Returns the exit code:
    any FAIL ⇒ 1; BLOCKED / SKIPPED / N/A alone ⇒ 0. Verdict is generated from scope + statuses
    so a run cannot claim a proof it did not perform (smoke ⇒ capped at ACCEPTED WITH CAVEATS).
    `test_id` is the UAT results-registry id (utils/verification/results/registry.py) — the
    join key shared by the registry row, the sheet row, and the MLflow receipt."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    W = 70
    out = []
    out.append("=" * 20 + f" {WORKLOAD} ACCEPTANCE REPORT " + "=" * 20)
    out.append(f"Run {run_id}   Profile {profile}   Shape {shape} ( {scope} )")
    out.append(f"Runtime {runtime}   When {when}")
    out.append("")
    out.append(_wrap("Attests to what rank 0 observed. On multi-node the CLI streams node 0 "
                     "only (`air logs <id> --node N`). If this report is absent, treat it as a "
                     "failure.", indent="  "))
    out.append("-" * W)

    has_fail = False
    for i, c in enumerate(checks, 1):
        if c.status == FAIL:
            has_fail = True
        out.append(f"CHECK {i} — {c.name}")
        out.append(f"  Status ....... {c.status}")
        out.append(f"  Measured ..... {c.measured}   Threshold: {c.threshold}")
        out.append(f"  What & why ... {_wrap(c.what_why)}")
        out.append(f"  Sufficient ... {_wrap(c.sufficient)}")
        out.append("-" * W)

    # Verdict — derived from scope + statuses (never a parallel narrative).
    softs = [c for c in checks if c.status in (BLOCKED, SKIPPED, NA)]
    if has_fail:
        verdict, exit_code = "NOT ACCEPTED", 1
        vline = "One or more checks did not clear their threshold at this shape."
    elif scope == "smoke" or softs:
        verdict, exit_code = "ACCEPTED WITH CAVEATS", 0
        capped = "smoke scope (single-process): distributed properties are vacuous here" \
            if scope == "smoke" else \
            "some checks were blocked / skipped / not applicable at this scale"
        vline = f"Every check that ran passed, but {capped} — see the rows above."
    else:
        verdict, exit_code = "ACCEPTED", 0
        vline = f"All checks passed at {shape}."
    out.append(f"VERDICT: {verdict}")
    out.append(f"  {vline}   Sentinels: {sentinels}   Test-id: {test_id or '-'}   "
               f"Exit: {exit_code}")

    # On FAIL — plain English first, then the raw trace (format spec §"On FAIL"). Never swallowed.
    if has_fail:
        out.append("")
        out.append("WHAT THIS LIKELY MEANS")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL:
                out.append(_wrap(f"CHECK {i} failed: {c.measured} did not meet "
                                 f"{c.threshold}. {c.likely_means}", indent="  "))
        out.append("")
        out.append("FOR SUPPORT — raw traceback")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL and c.traceback:
                out.append(f"  [CHECK {i} — {c.name}]")
                out.append(c.traceback.rstrip())

    print("\n" + "\n".join(out), flush=True)
    # Receipt AFTER the print: report delivery is priority one; the receipt is the durable leg.
    _receipt(checks, verdict, exit_code, test_id)
    return exit_code


# ==========================================================================================
# Model — a small in-code decoder transformer, dialed bigger than train_fsdp.py's correctness
# toy so the GPUs are actually busy and FSDP's all-gather / reduce-scatter are a real fraction
# of the step (otherwise "scaling" measures launch overhead, not the fabric). Attention uses a
# plain masked SDPA-free MultiheadAttention (NO FlashAttention) on purpose: the point is a
# CONSISTENT kernel across world sizes so the 8→16 ratio is apples-to-apples, not a peak-MFU
# number. Data is random tokens generated on-device each step (no host copy, no egress) — we
# time the step, we do not check convergence, so RNG content is irrelevant.
# ==========================================================================================
class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, vocab: int, dim: int, heads: int, layers: int, seq: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.nf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.nf(x))


def build_and_shard(args, mesh, device) -> nn.Module:
    """Build on meta then per-block fully_shard + one top-level call, so each rank materializes
    only its 1/world slice (a multi-B model would host-OOM if built whole on one process first).
    Wrapping strategy matches train_fsdp.py's (per-block + top-level)."""
    with torch.device("meta"):
        m = Transformer(args.vocab, args.dim, args.heads, args.layers, args.seq)
    kw = {"mesh": mesh}
    if args.mp:
        kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in m.blocks:
        fully_shard(blk, **kw)
    fully_shard(m, **kw)
    m.to_empty(device=device.type if device.type != "cpu" else "cpu")
    for p in m.parameters():
        with torch.no_grad():
            p.normal_(0, 0.02)
    return m


def count_params_full(model) -> int:
    """Global parameter count (gather sharded DTensors). Collective — call on every rank."""
    return sum(p.full_tensor().numel() if hasattr(p, "full_tensor") else p.numel()
               for p in model.parameters())


def rand_batch(args, device) -> torch.Tensor:
    """Random token grid (batch, seq) on-device. Content is irrelevant to a timing measurement."""
    return torch.randint(0, args.vocab, (args.batch, args.seq), device=device, dtype=torch.long)


def step_loss(model, tokens) -> torch.Tensor:
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    logits = model(inp)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


# ==========================================================================================
# Timing — warmup untimed, then `iters` timed steps. Each step is synchronized so perf_counter
# brackets real GPU work (CUDA is async). The reported per-step time is the MAX across ranks
# (all-reduce MAX of each rank's median): a distributed step is only done when the slowest rank
# finishes its collectives, so the slow rank — not rank 0 — bounds throughput.
# ==========================================================================================
def measure(model, opt, args, device, world) -> "dict":
    def one_step():
        tok = rand_batch(args, device)
        opt.zero_grad(set_to_none=True)
        step_loss(model, tok).backward()
        opt.step()

    for _ in range(args.warmup):                           # untimed: allocator/cudnn/NCCL warmup
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    step_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)

    my_p50 = statistics.median(step_ms)                    # this rank's typical step
    # Slowest rank bounds the step. all-reduce MAX so every rank agrees on the reported number.
    t = torch.tensor([my_p50], device=device)
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    step_s = t.item() / 1e3

    tokens_per_step = args.batch * args.seq * world        # weak scaling: per-rank batch fixed
    tokens_per_s = tokens_per_step / step_s
    per_gpu_tps = tokens_per_s / world
    samples_per_s = (args.batch * world) / step_s
    peak_gb = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    return {
        "step_ms": t.item(), "step_ms_local_p50": my_p50,
        "tokens_per_s": tokens_per_s, "per_gpu_tps": per_gpu_tps,
        "samples_per_s": samples_per_s, "peak_gb": peak_gb,
    }


def mlflow_log(metrics: "dict", params: "dict") -> None:
    """Best-effort MLflow receipt (rank 0). Same client-API pattern as train_fsdp.py: bind to the
    AIR-injected MLFLOW_RUN_ID, never start_run; alarm-guarded; no-op locally."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        c = MlflowClient()
        for k, v in params.items():
            c.log_param(run_id, k, v)
        for k, v in metrics.items():
            c.log_metric(run_id, k, float(v))
    except Exception as e:                                 # noqa: BLE001 — receipt is best-effort
        print(f"mlflow receipt FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


# ==========================================================================================
# Worker — one rank. Builds the sharded model, measures throughput, records checks, emits the
# data line (FSDP_THROUGHPUT) always and the pass-gated sentinel (FSDP_SCALING_OK) only when the
# efficiency bar is cleared. Renders the report on rank 0 and derives the exit code last.
# ==========================================================================================
def worker(rank: int, world: int, args):
    backend = "gloo" if args.local else "nccl"
    mesh_dev = "cpu" if args.local else "cuda"
    if args.local:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend, rank=rank, world_size=world)
    if not args.local:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cpu" if args.local else "cuda")
    mesh = init_device_mesh(mesh_dev, (world,))            # explicit mesh dodges a macOS auto-detect bug

    nccl = None
    try:
        nccl = torch.cuda.nccl.version()
    except Exception:                                      # noqa: BLE001 — no CUDA locally
        pass
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    runtime_str = f"torch {torch.__version__}, nccl {nccl}, cuda {torch.version.cuda}, {gpu_name}"
    if rank == 0:
        print(f"FSDP_VERSIONS torch={torch.__version__} nccl={nccl} cuda={torch.version.cuda} "
              f"world={world} device={mesh_dev} gpu={gpu_name}", flush=True)

    checks: "list[Check]" = []
    sentinels: "list[str]" = []
    nparams = 0
    m = None

    # --- Distributed launch sanity: world size resolved, backend up, all ranks in the group. ---
    launch_ok = world == int(os.environ.get("WORLD_SIZE", world)) and dist.get_world_size() == world
    checks.append(Check(
        name="The job launched across the expected number of GPUs",
        status=PASS if launch_ok else FAIL,
        measured=f"process group world size = {dist.get_world_size()} on backend '{backend}'",
        threshold=f"world size == {world} (whole 8xH100 nodes: 8 → 1 node, 16 → 2 nodes)",
        what_why="Throughput per GPU is only meaningful if every GPU you paid for actually joined "
                 "the job. torchrun must launch one rank per GPU and all ranks must form one "
                 "process group before the first collective.",
        sufficient="A process group whose size equals the requested GPU count means the launch "
                   "fanned out correctly; a smaller size means torchrun fell back to fewer ranks "
                   "or a node dropped out, and any 'scaling' number below would be a fiction.",
        likely_means="torchrun did not launch the expected ranks — check --nnodes/--nproc_per_node "
                     "wiring against the injected NUM_NODES/LOCAL_WORLD_SIZE, and confirm node 1 "
                     "with `air logs <id> --node 1`."))

    # --- Throughput measurement (the workload). Wrapped: OOM/runtime faults become a FAIL row. ---
    m_metrics = None
    try:
        m = build_and_shard(args, mesh, device)
        nparams = count_params_full(m)                     # collective
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        m_metrics = measure(m, opt, args, device, world)
        tps = m_metrics["tokens_per_s"]
        finite = tps == tps and tps not in (float("inf"), float("-inf")) and tps > 0
        tflops_per_gpu = 6 * nparams * m_metrics["per_gpu_tps"] / 1e12   # derived (~6N/token fwd+bwd)
        if rank == 0:
            print(f"FSDP_THROUGHPUT world={world} params={nparams} "
                  f"tokens_per_s={tps:.0f} per_gpu_tokens_per_s={m_metrics['per_gpu_tps']:.0f} "
                  f"samples_per_s={m_metrics['samples_per_s']:.1f} step_ms={m_metrics['step_ms']:.1f} "
                  f"peak_gb={m_metrics['peak_gb']:.2f} tflops_per_gpu={tflops_per_gpu:.1f}",
                  flush=True)
        checks.append(Check(
            name="Steady-state training throughput was measured",
            status=PASS if finite else FAIL,
            measured=f"{tps:,.0f} tokens/s total, {m_metrics['per_gpu_tps']:,.0f} tokens/s/GPU, "
                     f"{m_metrics['step_ms']:.1f} ms/step (slowest rank), "
                     f"~{tflops_per_gpu:.1f} TFLOP/s/GPU derived, peak {m_metrics['peak_gb']:.1f} GB",
            threshold="a finite, positive tokens/s over the timed window (warmup excluded)",
            what_why="This is the raw speed number: how many tokens the whole job trains per "
                     "second and, divided by GPU count, per GPU. The per-GPU figure is what stays "
                     "flat under ideal scaling, so it is the quantity the efficiency check compares.",
            sufficient=f"A finite per-GPU throughput over {args.iters} timed steps (after "
                       f"{args.warmup} warmup steps) is a real measurement; a NaN/inf or a raised "
                       "CUDA OOM is a FAIL and no scaling claim can rest on it.",
            likely_means="The timed loop raised (often CUDA OOM at this shape) — lower --batch or "
                         "--seq, or --dim/--layers, and retry; send this report and FSDP_VERSIONS."))
        sentinels.append("FSDP_THROUGHPUT_MEASURED" if finite else "FSDP_THROUGHPUT_FAILED")
    except Exception as e:                                 # noqa: BLE001 — record, don't raise
        checks.append(_fail_from_exc(
            "Steady-state training throughput was measured",
            "a finite, positive tokens/s over the timed window",
            "The raw tokens/s number is the basis for every scaling figure.",
            "The timed loop raised before producing a number — most often CUDA OOM at this shape "
            "(lower --batch/--seq/--dim) or an FSDP/NCCL mismatch (check FSDP_VERSIONS).", e))
        sentinels.append("FSDP_THROUGHPUT_FAILED")

    # --- Weak-scaling efficiency vs the single-node baseline (only when a baseline is supplied). ---
    if m_metrics is not None and args.baseline_per_gpu_tps > 0:
        eff = m_metrics["per_gpu_tps"] / args.baseline_per_gpu_tps
        eff_ok = eff >= args.min_efficiency
        if rank == 0 and eff_ok:
            print(f"FSDP_SCALING_OK world={world} per_gpu_tokens_per_s={m_metrics['per_gpu_tps']:.0f} "
                  f"baseline={args.baseline_per_gpu_tps:.0f} efficiency={eff:.3f} "
                  f"min={args.min_efficiency:.2f}", flush=True)
            sentinels.append("FSDP_SCALING_OK")
        checks.append(Check(
            name="Weak-scaling efficiency holds when crossing the node boundary",
            status=PASS if eff_ok else FAIL,
            measured=f"per-GPU throughput {m_metrics['per_gpu_tps']:,.0f} vs single-node baseline "
                     f"{args.baseline_per_gpu_tps:,.0f} tokens/s/GPU → efficiency {eff:.1%}",
            threshold=f"efficiency ≥ {args.min_efficiency:.0%} (per-GPU throughput retained at "
                      f"world={world} relative to the 1-node baseline)",
            what_why="Under weak scaling each GPU keeps the same per-GPU batch, so ideal hardware "
                     "would hold per-GPU throughput constant as you add nodes. The shortfall is the "
                     "cost of FSDP's cross-node all-gather / reduce-scatter over the EFA fabric — "
                     "the single most important number for deciding whether multi-node is worth it.",
            sufficient=f"Retaining ≥{args.min_efficiency:.0%} of the 1-node per-GPU rate means the "
                       "inter-node fabric is not the bottleneck at this shape; a large shortfall is a "
                       "real finding (comm-bound — raise batch/seq, or the model is too small to hide "
                       "the collective).",
            likely_means="Per-GPU throughput dropped sharply across the node boundary — usually the "
                         "model/batch is too small to overlap the inter-node collective, or the EFA "
                         "path is degraded (compare against the multinode-probe busbw receipt)."))
    elif m_metrics is not None:
        checks.append(Check(
            name="Weak-scaling efficiency holds when crossing the node boundary",
            status=SKIPPED,
            measured="no --baseline-per-gpu-tps supplied (this run only records absolute throughput)",
            threshold=f"efficiency ≥ {args.min_efficiency:.0%} vs the 1-node baseline",
            what_why="Scaling efficiency is a RATIO of this run's per-GPU throughput to the 1-node "
                     "baseline, so it needs the baseline number from the single-node run.",
            sufficient="Deliberately not computed: run the 1-node shape first, then pass its "
                       "per-GPU tokens/s as --baseline-per-gpu-tps to this multi-node run.",
            likely_means=""))

    if world > 1:
        dist.barrier()

    mem = m_metrics or {}
    if rank == 0:
        mlflow_log(
            metrics={k: mem.get(k, 0.0) for k in
                     ("tokens_per_s", "per_gpu_tps", "samples_per_s", "step_ms", "peak_gb")},
            params={"world": world, "params": nparams,
                    "shape": f"L{args.layers}-d{args.dim}-h{args.heads}-s{args.seq}-b{args.batch}",
                    "mixed_precision": args.mp,
                    "baseline_per_gpu_tps": args.baseline_per_gpu_tps})

    exit_code = 0
    if rank == 0:
        scope = "smoke" if world == 1 else "acceptance"
        shape = f"world={world}, {mesh_dev}, L{args.layers}-d{args.dim}-s{args.seq}-b{args.batch}"
        try:
            run_id = (os.environ.get("MLFLOW_RUN_ID")
                      or os.environ.get("MLFLOW_RUN_NAME") or "local")
            exit_code = render_report(
                checks, run_id=run_id,
                profile=("local-cpu" if args.local else "air"),
                shape=shape, scope=scope, runtime=runtime_str,
                sentinels=" ".join(sentinels), test_id="fsdp-scaling")
        except Exception:                                  # noqa: BLE001 — never lose the verdict
            _tb.print_exc()
            exit_code = 1

    dist.destroy_process_group()
    if rank == 0 and exit_code:
        sys.exit(exit_code)
    return exit_code


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true",
                   help="single-host CPU/gloo pre-flight; spawns --local-world procs (tiny shape)")
    p.add_argument("--local-world", type=int, default=2, help="world size for --local (≥2 non-vacuous)")
    p.add_argument("--master-port", type=int, default=29521)
    # model / task shape — defaults are the AIR H100 benchmark shape (~1.6B params).
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8, help="PER-GPU batch (weak scaling: global = batch·world)")
    p.add_argument("--mp", action="store_true", help="MixedPrecisionPolicy (bf16 compute / fp32 reduce)")
    # timing
    p.add_argument("--warmup", type=int, default=10, help="untimed warmup steps")
    p.add_argument("--iters", type=int, default=40, help="timed steps (median taken)")
    # scaling
    p.add_argument("--baseline-per-gpu-tps", type=float, default=0.0,
                   help="1-node per-GPU tokens/s; when set, the run reports weak-scaling efficiency")
    p.add_argument("--min-efficiency", type=float, default=0.80,
                   help="efficiency bar for the scaling check (health bar, not a hard SLA)")
    args = p.parse_args()

    if args.local:
        # Tiny shape so the CPU pre-flight is fast — this proves launch + measurement wiring, it is
        # NOT a throughput figure (CPU/gloo). Overridden only if the caller passed bigger values.
        if args.layers == 16 and args.dim == 2048:
            args.layers, args.dim, args.heads, args.seq, args.batch = 2, 128, 4, 64, 4
        args.warmup, args.iters = 2, 5
        import torch.multiprocessing as mp
        try:
            mp.spawn(worker, args=(args.local_world, args), nprocs=args.local_world, join=True)
        except Exception:                                  # noqa: BLE001
            _tb.print_exc()
            return 1
        return 0
    else:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        return worker(rank, world, args) or 0


if __name__ == "__main__":
    sys.exit(main())
