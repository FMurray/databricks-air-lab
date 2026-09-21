"""DDP training loop on AIR

DDP keeps a FULL replica of the model on every rank and all-reduces (averages) gradients during backward —
the exact opposite of FSDP's shard-and-reduce-scatter.

  1. DDP_REPLICATION_OK    — every rank holds the WHOLE model, and the replicas are bit-identical
                             across ranks (DDP kept them in sync). This is the property that
                             distinguishes DDP from FSDP, and the reason a state-dominated model
                             OOMs under DDP where it fits under FSDP (open-q #17 baseline).
  2. DDP_ALLREDUCE_OK      — DDP's gradient *all-reduce* is numerically correct vs a single-process
                             reference over the same global batch. Unlike FSDP's reduce-scatter,
                             the reduced grad is a FULL tensor, identical on every rank — no gather
                             is needed to compare.
  3. DDP_TRAIN_OK          — a real forward→backward→optimizer step converges (loss down).
  4. DDP_CKPT_RESUME_OK    — a plain single-writer checkpoint saves + resumes bit-for-bit.

Completion lines:
  DDP_TRAIN_COMPLETE    — Proofs 1+2+3 (replication + all-reduce + convergence).
  DDP_SUITE_COMPLETE    — all four (adds the checkpoint proof).

Design:
  * Self-contained + egress-free: depends on preinstalled `torch` only (`dependencies: []`),
    builds its model in-code, and generates synthetic deterministic data on-device from a
    closed-form formula (no RNG in data, no downloads).
  * Determinism is split: DATA is closed-form (pure function of global step), MODEL INIT uses RNG
    (init on CPU under a fixed seed then `.to(device)`). DDP additionally broadcasts rank 0's
    params at construction, so every rank starts from the identical replica regardless.
  * fp32 master params. `--mp` wraps the forward in bf16 autocast for the training-loop speedup
    only (DDP has no MixedPrecisionPolicy — that is FSDP-specific); Proof 2 is always fp32.

Pre-flight (single-host, CPU/gloo, real DDP replication via 2 spawned procs):
    python3 train_ddp.py --local
On AIR it is launched by torchrun (see workloads/ddp-multinode.example.yaml).

==========================================================================================
TRAINING-CRITICAL NCCL / CUDA / PyTorch FLAGS (what to know before you scale this up)
==========================================================================================
This tiny example runs fine on defaults. A REAL multinode model does not — the flags below decide
whether a fault fails fast or hangs the whole job, whether you hit false OOMs, and whether NCCL
uses the fast EFA/RDMA path. Names/defaults are for PyTorch >= 2.2 (the TORCH_NCCL_* rename); the
runtime build is printed on the DDP_VERSIONS line, verify against it. source: the
Databricks AIR "Multi-node LLM fine-tuning with FSDP" doc (docs.databricks.com/.../cli/examples/
multinode-llm-sft)

── AIR pre-injects these — LEAVE THEM ──
  NCCL_DEBUG=INFO            Your RDMA receipt, and the way to confirm EFA is actually in use. EFA is not
                            classic InfiniBand, so the Mellanox-style /sys/class/infiniband/.../counters/
                            byte stats don't apply, and in practice a workload has no reliable RDMA
                            traffic-counter surface. So the proof that traffic used RDMA (not a TCP
                            fallback) is the NCCL log lines: `NET/OFI Selected provider is efa … efa-direct
                            (found 32 nics)` + per-channel `via NET/Libfabric/<nic>/GDRDMA`. Keep it on
                            for multinode.
  NCCL_IB_TIMEOUT=22, NCCL_CUMEM_ENABLE=0, AWS_OFI_NCCL_VERSION=v1.15.0   EFA-tuned by the platform.
  FI_PROVIDER / NCCL_IB_HCA   The EFA DATA-plane transport is auto-selected by aws-ofi-nccl — do NOT
                            hand-set; a wrong value forces socket fallback and collapses inter-node
                            bandwidth (measured ~359 GB/s busbw on EFA → tens of GB/s on TCP).

── Recommended to SET (the official AIR FSDP example sets this; it is NOT auto-injected) ──
  NCCL_SOCKET_IFNAME=eth0    Pins NCCL's CONTROL-plane / rendezvous socket to eth0 so cross-node
                            bootstrap is reliable. Control plane ONLY — it does NOT move the EFA/GDRDMA
                            DATA plane onto TCP, so unlike FI_PROVIDER/NCCL_IB_HCA above it costs no
                            bandwidth. (Set in the AIR command/env, e.g. before torchrun.)

── Resilience: does a hung/faulted rank fail fast, or deadlock all ranks? (know / set) ──
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1   Default on (>=2.2 name; was NCCL_ASYNC_ERROR_HANDLING). The
                            ProcessGroupNCCL watchdog ABORTS the process on a collective error/timeout
                            instead of every rank blocking forever. Keep on. (TORCH_NCCL_BLOCKING_WAIT=1
                            is the older, higher-overhead synchronous alternative — mutually exclusive.)
  init_process_group(timeout=…)   CODE arg, not an env var (NCCL default 10 min). Raise it when the
                            first iteration is slow (torch.compile warmup, big checkpoint I/O) so a
                            legitimately-slow collective isn't killed as a hang. This trainer uses the
                            default; add `timeout=timedelta(minutes=N)` in worker() for long first steps.
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC   (~480s default) If the watchdog thread itself stalls, a monitor
                            kills the process so AIR `max_retries` can resubmit — pairs with --auto-resume.

── Debug a hang / desync (turn ON only while investigating — real overhead) ──
  TORCH_DISTRIBUTED_DEBUG=DETAIL   Checks every rank issues the same collectives in the same order and
                            logs DDP unused params.
  TORCH_NCCL_TRACE_BUFFER_SIZE=2000 + TORCH_NCCL_DUMP_ON_TIMEOUT=1   NCCL flight recorder: on a timeout,
                            dumps the last N collectives per rank so you can see WHICH rank/collective
                            stalled. Off by default; AIR's own multinode configs use =2000.
  NCCL_DEBUG_SUBSYS=INIT,NET,COLL   Narrows the verbose NCCL_DEBUG=INFO firehose to the subsystems you need.

── Memory / CPU: avoid false OOMs and CPU oversubscription ──
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   Cheapest fix for "OOM with free memory left" from
                            allocator fragmentation (variable batch/seq; FSDP all-gather transients).
                            (>=2.5 also accepts device-agnostic PYTORCH_ALLOC_CONF; other keys:
                            max_split_size_mb, garbage_collection_threshold.)
  OMP_NUM_THREADS           torchrun sets it to 1 (with a warning) if unset; set it to vCPUs/nproc so
                            dataloader/CPU ops don't oversubscribe and serialize. (Our TabICL runs found
                            the CPU stage was 83% of wall-clock — CPU threading is not a footnote.)
  CUDA_VISIBLE_DEVICES      torchrun + torch.cuda.set_device(LOCAL_RANK) bind the device (this trainer
                            already does this). Do NOT set it per-rank by hand under torchrun.

── DDP constructor flags (in code — wrap_ddp() here uses defaults; tune for real models) ──
  gradient_as_bucket_view=True   Grads alias the reduction buckets instead of a 2nd copy (~1 model-size
                            of memory saved). A good default for real training.
  static_graph=True         When the graph is identical every iteration: enables reduction optimizations,
                            permits activation checkpointing under DDP, and avoids the unused-param scan.
  find_unused_parameters=True   ONLY if some params legitimately get no grad some steps. Adds a per-step
                            graph traversal and is the usual source of the "finished reduction" crash on
                            a skipped step. Prefer fixing the skipped step over enabling this.
  bucket_cap_mb (25) / broadcast_buffers   Bigger buckets = fewer/larger all-reduces (more comm/compute
                            overlap, higher transient memory). broadcast_buffers=False if you have no
                            buffers (e.g. BatchNorm stats) to sync each forward.

This script's OWN args (--local, --steps, --batch is PER-RANK, --ckpt-dir must be a UC Volume not
/tmp, --mp, --proof4, --memprobe, env DDP_FAIL_AT_STEP for the max_retries test) are in `--help`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import textwrap
import traceback as _tb
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


INIT_SEED = 1234  # model-init RNG seed (data uses NO RNG)

# Synthetic task — next token is a closed-form modular function of a window of prior tokens.
# Pure function of (global_step, row): resume regenerates the identical batch for step N.
TASK_VOCAB = 64
TASK_WINDOW = 2
TASK_COEFFS = (1, 1)
TASK_BIAS = 0
SEED_A, SEED_B, SEED_C = 131, 17, 5

EXPECTED_LOSS_CEILING = (
    4.10  # < ln(VOCAB)=4.159 (uniform-init step-0); pinned from --local
)
LOSS_DROP_MARGIN = 0.05
REDUCE_TOL = 2e-4  # Proof 2 fp32 grad tolerance
SMOOTH_WINDOW = 20


WORKLOAD = "DDP TRAINING"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
NA = "N/A-at-this-scale"


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
    """Turn an exception into a FAIL record (record, don't re-raise) so the report still renders."""
    return Check(
        name=name,
        status=FAIL,
        measured=f"raised {type(exc).__name__}: {exc}",
        threshold=threshold,
        what_why=what_why,
        sufficient="A raised exception means the property could not be established.",
        likely_means=likely_means,
        traceback="".join(_tb.format_exception(exc)),
    )


def _wrap(text: str, indent: str = "               ") -> str:
    return textwrap.fill(text, width=96, initial_indent="", subsequent_indent=indent)


def _receipt(checks: "list[Check]", verdict: str, exit_code: int, test_id: str) -> None:
    """Dual-sink the verdict into MLflow params (the durable leg). stdout is primary but expires
    with job-run retention; the receipt makes an absent stdout report disambiguable. Client API
    bound to MLFLOW_RUN_ID (never start_run), alarm-guarded, skips cleanly when unset (local)."""
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
            client.log_param(
                run_id, f"acceptance_check_{i}", f"{c.status} — {c.name}"[:490]
            )
    except Exception as e:  # noqa: BLE001 — receipt is best-effort
        print(f"acceptance receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def render_report(
    checks: "list[Check]",
    run_id: str,
    profile: str,
    shape: str,
    scope: str,
    runtime: str,
    sentinels: str,
    test_id: str = "",
) -> int:
    """Render every check identically and DERIVE the exit code last. Any FAIL ⇒ 1; BLOCKED /
    SKIPPED / N/A alone ⇒ 0. Verdict is generated from scope + statuses so a run cannot claim a
    proof it did not perform (smoke ⇒ capped at ACCEPTED WITH CAVEATS)."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    W = 70
    out = []
    out.append("=" * 20 + f" {WORKLOAD} ACCEPTANCE REPORT " + "=" * 20)
    out.append(f"Run {run_id}   Profile {profile}   Shape {shape} ( {scope} )")
    out.append(f"Runtime {runtime}   When {when}")
    out.append("")
    out.append(
        _wrap(
            "Attests to what rank 0 observed. On multi-node the CLI streams node 0 "
            "only (`air logs <id> --node N`). If this report is absent, treat it as a "
            "failure.",
            indent="  ",
        )
    )
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

    softs = [c for c in checks if c.status in (BLOCKED, SKIPPED, NA)]
    if has_fail:
        verdict, exit_code = "NOT ACCEPTED", 1
        vline = "One or more checks did not clear their threshold at this shape."
    elif scope == "smoke" or softs:
        verdict, exit_code = "ACCEPTED WITH CAVEATS", 0
        capped = (
            "smoke scope (single-process): distributed properties are vacuous here"
            if scope == "smoke"
            else "some checks were blocked / skipped / not applicable at this scale"
        )
        vline = f"Every check that ran passed, but {capped} — see the rows above."
    else:
        verdict, exit_code = "ACCEPTED", 0
        vline = f"All checks passed at {shape}."
    out.append(f"VERDICT: {verdict}")
    out.append(
        f"  {vline}   Sentinels: {sentinels}   Test-id: {test_id or '-'}   "
        f"Exit: {exit_code}"
    )

    if has_fail:
        out.append("")
        out.append("WHAT THIS LIKELY MEANS")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL:
                out.append(
                    _wrap(
                        f"CHECK {i} failed: {c.measured} did not meet "
                        f"{c.threshold}. {c.likely_means}",
                        indent="  ",
                    )
                )
        out.append("")
        out.append("FOR SUPPORT — raw traceback")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL and c.traceback:
                out.append(f"  [CHECK {i} — {c.name}]")
                out.append(c.traceback.rstrip())

    print("\n" + "\n".join(out), flush=True)
    _receipt(checks, verdict, exit_code, test_id)
    return exit_code


# ==========================================================================================
# Synthetic task — closed-form, no RNG, pure function of global step. Identical to train_fsdp.py.
# ==========================================================================================
def synth_batch(
    step: int, row_start: int, row_count: int, seq: int, device
) -> torch.Tensor:
    """Token grid of shape (row_count, seq), long. Deterministic in (step, row); no RNG anywhere,
    so the CPU-derived loss ceiling transfers and step N regenerates identically on resume."""
    gid = (
        step * 1_000_003
        + row_start
        + torch.arange(row_count, device=device, dtype=torch.long)
    ).unsqueeze(1)  # (R,1)
    x = torch.empty(row_count, seq, dtype=torch.long, device=device)
    for i in range(TASK_WINDOW):
        x[:, i] = (gid.squeeze(1) * SEED_A + i * SEED_B + SEED_C) % TASK_VOCAB
    for t in range(TASK_WINDOW, seq):
        acc = torch.full((row_count,), TASK_BIAS, dtype=torch.long, device=device)
        for j, c in enumerate(TASK_COEFFS):
            acc = acc + c * x[:, t - 1 - j]
        x[:, t] = acc % TASK_VOCAB
    return x


def batch_loss(model, tokens: torch.Tensor, mp: bool = False) -> torch.Tensor:
    """Next-token cross-entropy, mean over all tokens (matches the reduction convention so a
    mean-of-per-rank-means equals the global mean when local batches are equal-sized).

    `mp` wraps the forward in bf16 autocast — the DDP analog of FSDP's MixedPrecisionPolicy. The
    loss is upcast back to fp32 by cross_entropy's reduction; master params stay fp32."""
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    ctx = (
        torch.autocast(device_type=inp.device.type, dtype=torch.bfloat16)
        if mp and inp.device.type == "cuda"
        else nullcontext()
    )
    with ctx:
        logits = model(inp)  # (B, L-1, V)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


# ==========================================================================================
# Model — a small in-code decoder transformer. Identical to train_fsdp.py (so curves compare).
# ==========================================================================================
class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        L = x.size(1)
        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.n1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class TinyTransformer(nn.Module):
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


def build_model(args, device) -> nn.Module:
    """CPU-init-under-fixed-seed then .to(device) — identical params on every rank. DDP also
    broadcasts rank 0's params at construction, so this is belt-and-suspenders for DDP."""
    torch.manual_seed(INIT_SEED)
    m = TinyTransformer(TASK_VOCAB, args.dim, args.heads, args.layers, args.seq)
    return m.to(device)


def wrap_ddp(model: nn.Module, args, local_rank: int, world: int) -> nn.Module:
    """Wrap once — the whole model is replicated on every rank (contrast FSDP's per-block
    fully_shard). `device_ids` binds the replica to this rank's GPU; omitted on CPU/gloo (--local).
    At world=1 there is nothing to parallelize, so return the bare model."""
    if world == 1:
        return model
    if args.local:
        return DDP(model)  # CPU/gloo — no device_ids
    return DDP(model, device_ids=[local_rank])


def _module(model: nn.Module) -> nn.Module:
    """The underlying replica (strips the DDP wrapper's `module.` indirection)."""
    return model.module if isinstance(model, DDP) else model


# ==========================================================================================
# Proof 1 — replication is real (DDP_REPLICATION_OK). The INVERSE of FSDP's sharding proof:
# every rank must hold the WHOLE model, and the replicas must be bit-identical across ranks.
# ==========================================================================================
def proof1_replication(model, world: int, device, opt) -> Check:
    """Two things, both the defining property of DDP: (a) per-rank PARAMETER STORAGE is the FULL
    model (not full/world) — the P+G+2P envelope on every GPU that makes DDP the OOM baseline for
    open-q #17; (b) the replicas are IDENTICAL across ranks (DDP broadcast + kept them in sync).
    We verify (b) by broadcasting rank 0's params and asserting max|local − rank0| == 0."""
    module = _module(model)
    params = list(module.parameters())
    full = sum(p.numel() for p in params)  # every rank holds all of them
    local = full  # replicated — not full/world

    # (b) cross-rank identity: broadcast rank 0's copy, take the largest elementwise diff, then
    # MAX-reduce so rank 0's report reflects the worst rank. Bit-identical fp32 ⇒ exactly 0.0.
    max_diff = 0.0
    for p in params:
        ref = p.detach().clone()
        dist.broadcast(ref, src=0)
        max_diff = max(max_diff, (p.detach() - ref).abs().max().item())
    md = torch.tensor([max_diff], device=device)
    dist.all_reduce(md, op=dist.ReduceOp.MAX)
    max_diff = md.item()

    # Full training-state envelope (bytes) on the persistent replicated state after one opt step.
    esz = params[0].element_size()
    local_param_b = local * esz
    local_grad_b = sum(p.grad.numel() for p in params if p.grad is not None) * esz
    local_optim_b = 0
    for st in opt.state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                local_optim_b += st[k].numel() * st[k].element_size()
    full_state_b = (full + full + 2 * full) * 4  # (P + G + 2·P optim) fp32, per rank
    local_state_b = local_param_b + local_grad_b + local_optim_b
    mem_gb = (
        (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
    )

    vacuous = world == 1
    ok = vacuous or max_diff == 0.0
    if dist.get_rank() == 0 and ok and not vacuous:
        print(
            f"DDP_REPLICATION_OK world={world} full={full} local={local} replica_max_diff=0 "
            f"mem={mem_gb:.2f}GB state_local={local_state_b / 2**30:.4f}GB "
            f"state_full={full_state_b / 2**30:.4f}GB "
            f"[param={local_param_b} grad={local_grad_b} optim={local_optim_b} bytes] "
            f"(full model on every rank — this is the FSDP-vs-DDP memory contrast, open-q #17)",
            flush=True,
        )

    status = NA if vacuous else (PASS if ok else FAIL)
    measured = (
        f"per-rank storage = {local_state_b / 2**30:.4f} GB (full model on every rank); "
        f"cross-rank replica max-diff = {max_diff:.3e} (params {local}/rank)"
    )
    return Check(
        name="The model is replicated and identical on every GPU",
        status=status,
        measured=measured,
        threshold="cross-rank replica max-diff == 0 (every rank holds the same full model)",
        what_why="This is the property that makes DDP different from FSDP: every GPU keeps a full "
        "copy of the model, gradients and Adam moments, and DDP keeps those copies in "
        "lock-step. It is also why a large model that fits under FSDP will run out of "
        "memory under DDP — each GPU pays the full P+G+2P state cost.",
        sufficient="A zero cross-rank difference means the replicas really are identical; a "
        "non-zero diff means the ranks drifted apart and any averaged gradient is "
        "meaningless. At world=1 there is nothing to replicate, so this is N/A.",
        likely_means="Replicas diverged — most often DDP was not actually wrapping the model, or "
        "a rank initialized from a different seed. Send the DDP_VERSIONS line and "
        "this report.",
    )


# ==========================================================================================
# Proof 2 — gradient all-reduce is correct (DDP_ALLREDUCE_OK).
# ==========================================================================================
def proof2_allreduce(args, device, world: int, local_rank: int) -> Check:
    """Build a DDP model and a single-process reference from an IDENTICAL init (same seed, same
    process). One backward, fp32, autocast off, mean-reduction loss on a global batch split across
    ranks. DDP all-reduces (averages) grads during backward, so after backward every rank already
    holds the FULL averaged gradient — NO gather is needed (the key simplification vs FSDP's
    reduce-scatter, which leaves each rank a shard). Assert max|g − g_ref| < REDUCE_TOL vs the
    reference computed single-process over the same global batch, after the FIRST backward."""
    rank = dist.get_rank()
    local_b = args.reduce_batch
    global_b = local_b * world

    # Reference: unwrapped, fp32, full global batch, single process.
    ref = build_model(args, device)
    ref_tokens = synth_batch(0, 0, global_b, args.seq, device)
    ref.zero_grad(set_to_none=True)
    batch_loss(ref, ref_tokens).backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}

    # DDP model from the SAME seed → identical init; fp32, no autocast.
    dm = wrap_ddp(build_model(args, device), args, local_rank, world)
    my_tokens = synth_batch(0, rank * local_b, local_b, args.seq, device)
    dm.zero_grad(set_to_none=True)
    batch_loss(dm, my_tokens).backward()  # DDP all-reduces grads in the backward

    max_diff = 0.0
    for n, p in _module(dm).named_parameters():
        max_diff = max(max_diff, (p.grad - ref_grads[n]).abs().max().item())

    vacuous = world == 1
    ok = max_diff < REDUCE_TOL
    if rank == 0 and ok and not vacuous:
        print(
            f"DDP_ALLREDUCE_OK world={world} grad_diff={max_diff:.3e} tol={REDUCE_TOL:.1e} "
            f"global_batch={global_b}",
            flush=True,
        )

    status = NA if vacuous else (PASS if ok else FAIL)
    return Check(
        name="Gradients are combined correctly across GPUs (all-reduce)",
        status=status,
        measured=f"largest gradient difference vs a single-process reference = {max_diff:.3e} "
        f"(global batch {global_b})",
        threshold=f"max gradient difference < {REDUCE_TOL:.1e}",
        what_why="Each GPU computes gradients on its own slice of the batch; DDP must average them "
        "so every rank ends up with the correct combined gradient. If this is even "
        "slightly wrong, training looks like it runs but silently learns the wrong "
        "thing — the hardest kind of bug to notice.",
        sufficient=f"Matching a trusted single-process gradient to within {REDUCE_TOL:.1e} means "
        "the all-reduce is numerically correct, not just non-crashing. A failure shows "
        "as a difference above the tolerance. At world=1 the all-reduce is a no-op, so "
        "this is N/A, not a pass.",
        likely_means="The gradient all-reduce produced wrong values — possibly a runtime/NCCL "
        "mismatch or a DDP wrapping bug. Do NOT trust training numbers from this run; "
        "send this report and the DDP_VERSIONS line to support.",
    )


# ==========================================================================================
# Proof 3 — the loop trains (DDP_TRAIN_OK).
# ==========================================================================================
def train_loop(model, opt, sched, args, device, world, start_step, mlf, ckpt_dir):
    """Run steps [start_step, args.steps). Emits Proof 1 after the first optimizer step (moments
    allocated), logs the loss curve, optionally checkpoints, and (if DDP_FAIL_AT_STEP is set)
    hard-exits to exercise platform resume (open-q #10). Returns (train_check, replication_check).
    A non-finite loss is recorded as a FAIL (no raise) so the report renders."""
    rank = dist.get_rank()
    losses = []
    replication_check = None
    fail_at = int(os.environ.get("DDP_FAIL_AT_STEP", "-1"))
    save_every = args.save_every

    train_name = "The training loop runs a real step and the loss goes down"
    train_thresh = (
        f"final-window mean below step-0 by ≥ {LOSS_DROP_MARGIN} AND below "
        f"ceiling {EXPECTED_LOSS_CEILING}"
    )
    train_what = (
        "A full forward → backward → optimizer step on the replicated model, repeated, "
        "must actually reduce the loss. This is the end-to-end proof that a model can "
        "be trained on this platform, not just that the pieces initialize."
    )
    train_likely = (
        "Loss did not fall as expected. If the all-reduce check passed, this is almost "
        "always tuning/precision (learning rate, steps, mixed precision) — NOT a "
        "platform fault; do not report 'DDP doesn't work on AIR'. Re-triage K/LR "
        "before escalating."
    )

    step0_loss = None
    for step in range(start_step, args.steps):
        tokens = synth_batch(step, rank * args.batch, args.batch, args.seq, device)
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(model, tokens, mp=args.mp)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        lv = loss.item()
        losses.append(lv)
        if step0_loss is None:
            step0_loss = lv

        # Proof 1 fires once, after the first step so Adam moments exist for the envelope.
        if replication_check is None:
            replication_check = proof1_replication(model, world, device, opt)

        if lv != lv or abs(lv) == float("inf"):  # NaN/±inf guard — record, don't raise
            train_check = Check(
                name=train_name,
                status=FAIL,
                measured=f"non-finite loss at step {step}: {lv}",
                threshold=train_thresh,
                what_why=train_what,
                sufficient="A finite, falling loss is required; NaN/inf means the step diverged.",
                likely_means="The loss became NaN or inf — usually too high a learning rate or a "
                "mixed-precision overflow. Lower --lr or drop --mp and retry.",
            )
            return train_check, replication_check

        if rank == 0:
            if mlf:
                mlf.log_metric("train_loss", lv, step=step)
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                print(f"[step {step}] loss={lv:.4f}", flush=True)

        if save_every and ckpt_dir and step > start_step and step % save_every == 0:
            save_checkpoint(model, opt, sched, step, ckpt_dir, world, device)

        if fail_at >= 0 and step == fail_at and start_step == 0:
            # Whole-rank hard exit to let the PLATFORM's max_retries decide (torchrun
            # --max-restarts=0 in the YAML isolates this from torchrun's elastic restart).
            # Guarded on start_step == 0 so the fault fires ONCE, on the cold run; the resumed
            # attempt (start_step > 0 via --auto-resume) skips it and completes.
            print(
                f"[rank{rank}] DDP_FORCED_EXIT at step {step} (open-q #10 max_retries test)",
                flush=True,
            )
            sys.exit(137)

    win = min(SMOOTH_WINDOW, len(losses))
    final = sum(losses[-win:]) / win
    drop = step0_loss - final
    ok = (final < step0_loss - LOSS_DROP_MARGIN) and (final < EXPECTED_LOSS_CEILING)
    if rank == 0:
        if ok:
            print(
                f"DDP_TRAIN_OK step0={step0_loss:.4f} final={final:.4f} drop={drop:.4f} "
                f"ceiling={EXPECTED_LOSS_CEILING} steps={args.steps}",
                flush=True,
            )
        if mlf:
            mlf.log_metric("final_window_loss", final)
            mlf.log_param("expected_loss_ceiling", EXPECTED_LOSS_CEILING)

    train_check = Check(
        name=train_name,
        status=PASS if ok else FAIL,
        measured=f"loss step0={step0_loss:.4f} → final-window={final:.4f} (drop {drop:.4f}) "
        f"over {args.steps} steps",
        threshold=train_thresh,
        what_why=train_what,
        sufficient=f"A drop of {drop:.4f} below the {LOSS_DROP_MARGIN} margin and a final loss "
        f"under the {EXPECTED_LOSS_CEILING} ceiling means the optimizer step is doing "
        "real work. A failure looks like a flat or rising loss curve.",
        likely_means=train_likely,
    )
    return train_check, replication_check


# ==========================================================================================
# Proof 4 — checkpoint + resume is correct (DDP_CKPT_RESUME_OK). Plain single-writer torch.save:
# the model is replicated, so rank 0 owns one authoritative copy — no sharded DCP, no DTensor
# gather, no collective desync trap. Every rank loads the identical file on resume.
# ==========================================================================================
def _solo_probe_write(path: str, timeout_s: int = 10) -> bool:
    """Write+fsync+remove a tiny sentinel, guarded by a per-process alarm. Proves permission/403
    (BR-2's failure mode), not capacity/throughput for a real multi-GB checkpoint."""

    def _timeout(signum, frame):
        raise TimeoutError(f"probe write to {path} exceeded {timeout_s}s")

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout_s)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"probe")
            f.flush()
            os.fsync(f.fileno())
        os.remove(path)
        return True
    except Exception as e:  # noqa: BLE001 — any failure ⇒ probe fail
        print(
            f"[rank{dist.get_rank()}] probe write FAILED: {type(e).__name__}: {e}",
            flush=True,
        )
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def save_checkpoint(model, opt, sched, step, ckpt_dir, world, device) -> bool:
    """Single-writer save: rank 0 probes the path (BR-2), broadcasts the success flag so EVERY
    rank branches identically, then rank 0 alone writes the replicated state. A shared barrier
    keeps the ranks aligned. Unlike FSDP's collective DCP save there is no per-rank shard, so this
    cannot desync a collective — the only guard needed is the shared branch on the probe flag."""
    rank = dist.get_rank()
    probe = (
        _solo_probe_write(os.path.join(ckpt_dir, ".probe_rank0")) if rank == 0 else True
    )
    flag = torch.tensor([1.0 if probe else 0.0], device=device)
    dist.broadcast(flag, src=0)  # every rank sees rank 0's probe result
    if flag.item() < 1.0:
        if rank == 0:
            print(
                f"DDP_CKPT_PROBE_FAILED step={step} — blocked-on-BR-2 (UC-volume 403); "
                f"skipping save",
                flush=True,
            )
        return False
    if rank == 0:
        path = os.path.join(ckpt_dir, f"step_{step}.pt")
        state = {
            "model": _module(model).state_dict(),
            "optim": opt.state_dict(),
            "step": step,
        }
        if sched is not None:
            state["sched"] = sched.state_dict()
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(state, path)
        print(f"[rank0] checkpoint save OK step={step} → {path}", flush=True)
    dist.barrier()
    return True


def _fingerprint(module, opt, fp_name: str) -> str:
    """Bit-exact hash of one fixed param + its Adam moments. DDP does NOT reshard params (contrast
    FSDP2, which needs the name-keyed get_state_dict workaround), so the live parameter object is
    identity-stable and `opt.state[param]` finds its moments directly — a plain-tensor lookup."""
    name_to_p = dict(module.named_parameters())
    p = name_to_p[fp_name]
    parts = [p.detach().to(torch.float64).cpu().numpy().tobytes()]
    st = opt.state.get(p, {})
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st:
            parts.append(st[k].detach().to(torch.float64).cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def proof4_checkpoint_resume(args, device, world, local_rank, ckpt_dir) -> Check:
    """Two-phase within one run: (1) train N steps, save (probe-gated), record loss@N +
    fingerprint (params + optimizer moments); (2) reconstruct a fresh DDP model + optimizer +
    scheduler, load, check the fingerprint is bit-identical and the step-N loss matches the
    pre-save trajectory. Data is a pure function of global step, so the resumed step feeds the
    identical batch. Returns PASS, BLOCKED (probe failed), or FAIL (a real checkpoint fault)."""
    import numpy  # noqa: F401 — required by .numpy() in _fingerprint; fail loud here if absent

    rank = dist.get_rank()
    n = args.ckpt_steps
    save_dir = os.path.join(ckpt_dir, "proof4")

    # Phase 1: train N steps on a fresh model+opt+sched.
    m = wrap_ddp(build_model(args, device), args, local_rank, world)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, n // 2), gamma=0.5)
    for step in range(n):
        tok = synth_batch(step, rank * args.batch, args.batch, args.seq, device)
        opt.zero_grad(set_to_none=True)
        batch_loss(m, tok).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        tok_n = synth_batch(n, rank * args.batch, args.batch, args.seq, device)
        loss_at_n_pre = batch_loss(m, tok_n).item()
    fp_name = "blocks.0.mlp.0.weight"  # a fixed param that always trains
    fp_pre = _fingerprint(_module(m), opt, fp_name)

    p4_name = "A checkpoint saves and resumes exactly where it left off"
    p4_thresh = "params + optimizer moments bit-identical AND resume loss within 1e-4"
    p4_what = (
        "Long jobs must survive interruption. On resume the model, optimizer moments and "
        "scheduler must come back bit-for-bit; otherwise a 'resumed' run silently restarts "
        "from worse weights and wastes the compute already spent."
    )

    if not save_checkpoint(m, opt, sched, n, save_dir, world, device):
        return Check(
            name=p4_name,
            status=BLOCKED,
            measured="checkpoint probe-write failed (UC-volume 403)",
            threshold=p4_thresh,
            what_why=p4_what,
            sufficient="Blocked by an external precondition (BR-2 UC-volume write permission), "
            "not a fault in checkpoint/resume itself.",
            likely_means="The checkpoint directory could not be written — a permissions/BR-2 "
            "block on the UC volume, not a training failure.",
        )

    # Phase 2: fresh model+opt+sched, load the replicated state on every rank, assert bit-identical
    # + trajectory continuity. Every rank reads the same single file (map to its own device).
    m2 = wrap_ddp(build_model(args, device), args, local_rank, world)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=args.lr)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=max(1, n // 2), gamma=0.5)
    state = torch.load(os.path.join(save_dir, f"step_{n}.pt"), map_location=device)
    _module(m2).load_state_dict(state["model"])
    opt2.load_state_dict(state["optim"])
    sched2.load_state_dict(state["sched"])

    fp_post = _fingerprint(_module(m2), opt2, fp_name)
    with torch.no_grad():
        loss_at_n_post = batch_loss(m2, tok_n).item()

    fp_match = fp_pre == fp_post
    loss_diff = abs(loss_at_n_post - loss_at_n_pre)
    loss_match = loss_diff < 1e-4
    ok = fp_match and loss_match
    if rank == 0 and ok:
        print(
            f"DDP_CKPT_RESUME_OK fingerprint_match={fp_match} "
            f"resumed_from=loss@{n}={loss_at_n_pre:.6f} (post={loss_at_n_post:.6f})",
            flush=True,
        )

    return Check(
        name=p4_name,
        status=PASS if ok else FAIL,
        measured=f"params/moments bit-identical={fp_match}; resume loss {loss_at_n_post:.6f} vs "
        f"pre-save {loss_at_n_pre:.6f} (diff {loss_diff:.2e})",
        threshold=p4_thresh,
        what_why=p4_what,
        sufficient="A matching fingerprint proves the weights AND Adam moments round-tripped "
        "exactly; the matching next-step loss proves the trajectory continues rather "
        "than restarting. A failure shows as a fingerprint mismatch or a loss spike.",
        likely_means="Checkpoint saved but resume did not reproduce the pre-save state — usually "
        "optimizer/scheduler state was dropped so the moments reset and the loss "
        "spikes. Capture this report and the save/load lines for support.",
    )


# ==========================================================================================
# Rung 4 (stretch) — DDP-OOM baseline for the open-q #17 counterfactual. This is the CONTROL that
# should CUDA-OOM first; the FSDP-fit arm lives in ../fsdp/train_fsdp.py (--memprobe --arm fsdp),
# which meta-inits + shards a model too big to replicate. DDP cannot dodge that OOM — that is the
# point — so this file carries the ddp arm only.
# ==========================================================================================
def memprobe(rank, world, args, device, local_rank):
    """Full replicated model + one optimizer step. STATE-dominated shape (deep+wide, short seq,
    small batch, no activation checkpointing) so the OOM is driven by the P+G+2P state DDP cannot
    shard. Reports peak CUDA GB or catches the CUDA OOM; host-OOM (exit 137, no traceback) vs
    CUDA-OOM must be labeled by the reader."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        model = wrap_ddp(build_model(args, device), args, local_rank, world)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        tok = synth_batch(0, rank * args.batch, args.batch, args.seq, device)
        opt.zero_grad(set_to_none=True)
        batch_loss(model, tok).backward()
        opt.step()
        peak = (
            (torch.cuda.max_memory_allocated() / 2**30)
            if device.type == "cuda"
            else 0.0
        )
        if rank == 0:
            print(
                f"DDP_MEMPROBE_OK arm=ddp peak_gb={peak:.2f} "
                f"shape=L{args.layers}-d{args.dim}-h{args.heads}-s{args.seq} "
                f"batch={args.batch} world={world} — DDP replicates full state on every rank; "
                f"this is the OOM baseline the FSDP arm (../fsdp) is measured against "
                f"(open-q #17)",
                flush=True,
            )
        return True
    except torch.cuda.OutOfMemoryError as e:  # CUDA OOM (NOT host-OOM/exit137)
        if rank == 0:
            print(
                f"DDP_MEMPROBE_CUDA_OOM arm=ddp — {type(e).__name__}: CUDA out of memory "
                f"(labeled CUDA-OOM, not host-OOM). Expected: DDP replicates the full model, so "
                f"a state-dominated model OOMs here where FSDP would fit at world={world}",
                flush=True,
            )
        return False


# ==========================================================================================
# MLflow — tracking endpoint only. Client API bound to MLFLOW_RUN_ID (never start_run — resuming
# the launcher-owned run fails silently on the job plane). Degrades to stdout when absent (local).
# ==========================================================================================
class _MlflowReceipt:
    def __init__(self, client, run_id: str):
        self._c = client
        self._run_id = run_id

    def log_metric(self, key, value, step=None):
        self._c.log_metric(
            self._run_id, key, value, step=step if step is not None else 0
        )

    def log_param(self, key, value):
        self._c.log_param(self._run_id, key, value)


def open_mlflow(rank: int):
    if rank != 0:
        return None
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        print("[rank0] MLFLOW_RUN_ID unset — logging to stdout only", flush=True)
        return None
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("[rank0] mlflow not installed — logging to stdout only", flush=True)
        return None
    try:
        client = MlflowClient()
        client.log_param(run_id, "mlflow_tracking_reachable", "yes")
        return _MlflowReceipt(client, run_id)
    except Exception as e:  # noqa: BLE001
        print(
            f"[rank0] mlflow tracking endpoint NOT reachable: {e} — stdout only",
            flush=True,
        )
        return None


def _maybe_resume(rank, model, opt, sched, ckpt_dir, args, device) -> int:
    """Find the newest surviving checkpoint and LOAD model+optim+scheduler into the live objects,
    returning the step to resume from (prints RESUMED_FROM_STEP=N) or 0 (COLD_START). Plain
    torch.load of the replicated single-writer file on every rank (contrast FSDP's DCP load)."""
    latest = -1
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.startswith("step_") and name.endswith(".pt"):
                try:
                    latest = max(latest, int(name[len("step_") : -len(".pt")]))
                except ValueError:
                    pass
    if latest < 0:
        if rank == 0:
            print("COLD_START", flush=True)
        return 0
    state = torch.load(os.path.join(ckpt_dir, f"step_{latest}.pt"), map_location=device)
    _module(model).load_state_dict(state["model"])
    opt.load_state_dict(state["optim"])
    if sched is not None and "sched" in state:
        sched.load_state_dict(state["sched"])
    if rank == 0:
        print(f"RESUMED_FROM_STEP={latest}", flush=True)
    return latest


# ==========================================================================================
# Worker — one rank. Runs Proofs 2, (1+3 via the loop), and 4; emits completion lines.
# ==========================================================================================
def worker(rank: int, world: int, args):
    backend = "gloo" if args.local else "nccl"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.local:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend, rank=rank, world_size=world)
    if not args.local:
        torch.cuda.set_device(local_rank)
    device = torch.device("cpu" if args.local else "cuda")

    nccl = None
    try:
        nccl = torch.cuda.nccl.version()
    except Exception:  # noqa: BLE001 — no CUDA locally
        pass
    runtime_str = (
        f"torch {torch.__version__}, nccl {nccl}, cuda {torch.version.cuda}, DDP"
    )
    if rank == 0:
        print(
            f"DDP_VERSIONS torch={torch.__version__} nccl={nccl} cuda={torch.version.cuda} "
            f"world={world} device={'cpu' if args.local else 'cuda'}",
            flush=True,
        )

    if args.memprobe:  # rung 4 — one arm, then exit
        memprobe(rank, world, args, device, local_rank)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    mlf = open_mlflow(rank)
    ckpt_dir = args.ckpt_dir

    # Proof 2 first — a fresh model, one backward, before training mutates anything.
    try:
        reduce_check = proof2_allreduce(args, device, world, local_rank)
    except Exception as e:  # noqa: BLE001
        reduce_check = _fail_from_exc(
            "Gradients are combined correctly across GPUs (all-reduce)",
            f"max gradient difference < {REDUCE_TOL:.1e}",
            "DDP must average per-rank gradients so every rank gets the correct combined gradient.",
            "The all-reduce proof raised before producing a number — check DDP_VERSIONS for a "
            "runtime/NCCL mismatch and send this report to support.",
            e,
        )

    # Proofs 1 + 3 — the training model + loop (Proof 1 recorded after the first step).
    model = wrap_ddp(build_model(args, device), args, local_rank, world)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, args.steps // 3), gamma=0.5
    )

    start_step = 0
    if args.auto_resume and ckpt_dir:
        start_step = _maybe_resume(rank, model, opt, sched, ckpt_dir, args, device)

    if rank == 0 and mlf:
        mlf.log_param("torch_version", torch.__version__)
        mlf.log_param("world", world)
        mlf.log_param(
            "shape",
            f"L{args.layers}-d{args.dim}-h{args.heads}-s{args.seq}-v{TASK_VOCAB}",
        )
        mlf.log_param("mixed_precision", args.mp)
    train_check, replication_check = train_loop(
        model, opt, sched, args, device, world, start_step, mlf, ckpt_dir
    )

    # Completion line (Proofs 1+2+3). Emitted ONLY when all three STRICTLY PASS at world>=2. At
    # world=1 replication + all-reduce are vacuous (N/A), so the receipt must NOT print.
    complete = (
        world >= 2
        and replication_check.status == PASS
        and reduce_check.status == PASS
        and train_check.status == PASS
    )
    if rank == 0 and complete:
        print(
            "DDP_TRAIN_COMPLETE proofs=1,2,3 (replication+allreduce+convergence)",
            flush=True,
        )

    checks = [replication_check, reduce_check, train_check]
    sentinels = ["DDP_TRAIN_COMPLETE" if complete else "DDP_TRAIN_INCOMPLETE"]

    # Proof 4 — may be BLOCKED (blocked-on-BR-2) without failing the training receipt.
    if args.proof4 and ckpt_dir:
        try:
            ckpt_check = proof4_checkpoint_resume(
                args, device, world, local_rank, ckpt_dir
            )
        except Exception as e:  # noqa: BLE001
            ckpt_check = _fail_from_exc(
                "A checkpoint saves and resumes exactly where it left off",
                "params + optimizer moments bit-identical AND resume loss within 1e-4",
                "On resume the model and optimizer state must come back bit-for-bit or a resumed "
                "job silently restarts from worse weights.",
                "Checkpoint save or load raised — capture this report and the save/load lines "
                "for support.",
                e,
            )
        checks.append(ckpt_check)
        if rank == 0:
            if ckpt_check.status == PASS and complete:
                print(
                    "DDP_SUITE_COMPLETE proofs=1,2,3,4 (adds checkpoint/resume)",
                    flush=True,
                )
                sentinels.append("DDP_SUITE_COMPLETE")
            elif ckpt_check.status == BLOCKED:
                print(
                    "DDP_SUITE_BLOCKED proof4=blocked-on-BR-2 (training receipt stands via "
                    "DDP_TRAIN_COMPLETE)",
                    flush=True,
                )
                sentinels.append("DDP_SUITE_BLOCKED")
    elif args.proof4:
        checks.append(
            Check(
                name="A checkpoint saves and resumes exactly where it left off",
                status=SKIPPED,
                measured="no --ckpt-dir provided",
                threshold="params + optimizer moments bit-identical AND resume loss within 1e-4",
                what_why="Checkpoint/resume lets a long job survive interruption without losing "
                "progress.",
                sufficient="Deliberately not run this invocation: --proof4 was set but no checkpoint "
                "directory was given, so there is nowhere to save.",
                likely_means="",
            )
        )

    dist.barrier()

    exit_code = 0
    if rank == 0:
        scope = "smoke" if world == 1 else "acceptance"
        shape = f"world={world}, {'cpu' if args.local else 'cuda'}"
        try:
            run_id = (
                os.environ.get("MLFLOW_RUN_ID")
                or os.environ.get("MLFLOW_RUN_NAME")
                or "local"
            )
            exit_code = render_report(
                checks,
                run_id=run_id,
                profile=("local-cpu" if args.local else "air"),
                shape=shape,
                scope=scope,
                runtime=runtime_str,
                sentinels=" ".join(sentinels),
                test_id="ddp",
            )  # utils/verification/results/registry.py
        except Exception:  # noqa: BLE001 — never lose the verdict
            _tb.print_exc()
            exit_code = 1

    dist.destroy_process_group()
    if rank == 0 and exit_code:
        sys.exit(exit_code)
    return exit_code


# ==========================================================================================
# Entry point.
# ==========================================================================================
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="single-host CPU/gloo pre-flight; spawns --local-world procs (real DDP replication)",
    )
    p.add_argument(
        "--local-world",
        type=int,
        default=2,
        help="world size for --local (≥2 to be non-vacuous)",
    )
    p.add_argument("--master-port", type=int, default=29521)
    # model / task shape
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    # training
    p.add_argument("--steps", type=int, default=300)
    p.add_argument(
        "--batch", type=int, default=16, help="per-rank batch (global = batch·world)"
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--reduce-batch", type=int, default=8, help="per-rank batch for Proof 2"
    )
    p.add_argument(
        "--mp",
        action="store_true",
        help="bf16 autocast forward (DDP has no MixedPrecisionPolicy)",
    )
    # checkpoint / resume
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help="checkpoint dir (UC volume on AIR; /tmp fallback)",
    )
    p.add_argument(
        "--proof4", action="store_true", help="run Proof 4 (checkpoint/resume)"
    )
    p.add_argument(
        "--ckpt-steps", type=int, default=20, help="steps before the Proof 4 save"
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="save a checkpoint every N training steps (0=off)",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from newest checkpoint (open-q #10)",
    )
    # rung 4 (stretch) — DDP-OOM baseline (open-q #17). Dial deep+wide, SHORT seq, small batch.
    p.add_argument(
        "--memprobe",
        action="store_true",
        help="rung 4: DDP-OOM baseline, one arm then exit",
    )
    args = p.parse_args()

    if args.local:
        if args.ckpt_dir is None and args.proof4:
            args.ckpt_dir = "/tmp/ddp_local_ckpt"
        import torch.multiprocessing as mp

        try:
            mp.spawn(
                worker,
                args=(args.local_world, args),
                nprocs=args.local_world,
                join=True,
            )
        except Exception:  # noqa: BLE001
            _tb.print_exc()
            return 1
        return 0
    else:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        return worker(rank, world, args) or 0


if __name__ == "__main__":
    sys.exit(main())
