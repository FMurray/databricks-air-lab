"""Real FSDP2 training loop on AIR (BR-4) — four assertion-gated proofs.

The fabric + numerics are already green at 160 GPUs, but only with *synthetic collectives*
(`nccl-allreduce`, `distributed_correctness_probe.py`). Neither trains a model, and neither
exercises the collectives FSDP actually uses. This trainer proves four distinct, unproven
properties, each behind its own sentinel that is unreachable unless its assertions passed
("Exited 0" is not evidence):

  1. FSDP_SHARDING_OK      — params/grads/optimizer state are sharded (≈1/world persistent
                             storage), the property that distinguishes FSDP from DDP.
  2. FSDP_REDUCE_OK        — FSDP2's gradient *reduce-scatter* is numerically correct vs a
                             single-process reference (a different collective on a different
                             memory layout than the probe's all-reduce parity check).
  3. FSDP_TRAIN_OK         — a real forward→backward→optimizer step converges (loss down).
  4. FSDP_CKPT_RESUME_OK   — sharded DCP checkpoint saves + resumes from where it left off.

Completion lines:
  FSDP_BR4_COMPLETE     — Proofs 1+2+3. THE BR-4 acceptance line.
  FSDP_SUITE8_COMPLETE  — all four (adds the checkpoint proof = suite #8).

Design:
  * Self-contained + egress-free: depends on preinstalled `torch` only (`dependencies: []`),
    builds its model in-code, and generates synthetic deterministic data on-device from a
    closed-form formula (no RNG in data, no downloads). Runnable now, without the pkg repo.
  * Determinism is split: DATA is closed-form (pure function of global step), MODEL INIT uses
    RNG. CPU and CUDA RNG streams differ, so init on CPU under a fixed seed then `.to(device)`
    for the training model and the Proof-2 reference (the rung-4 stress model is the sole
    exception — meta/sharded init, so a DDP-sized model can't host-OOM before it shards).
  * fp32 master params. `--mp` enables MixedPrecisionPolicy (bf16 compute, fp32 reduce/master)
    for the training-loop speedup only; Proof 2 is always fp32/AMP-off; persistent storage
    stays fp32 either way, which is what the open-q #17 envelope accounting assumes.

Pre-flight (single-host, CPU/gloo, real FSDP sharding via 2 spawned procs — answers the
rung-0 "does fully_shard run on CPU/gloo?" question):
    python3 train_fsdp.py --local
On AIR it is launched by torchrun (see workloads/fsdp-multinode.example.yaml).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import textwrap
import traceback as _tb
from dataclasses import dataclass, field
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

# ------------------------------------------------------------------------------------------
# Pre-registered constants — pinned in NOTES.md BEFORE submit. Data constants are bit-exact
# invariants (closed-form, no RNG); the loss ceiling / reduction tolerance are LOOSE bounds
# derived from the CPU pre-flight (they shift with seed/LR/K/precision — see NOTES.md).
# ------------------------------------------------------------------------------------------
INIT_SEED = 1234                     # model-init RNG seed (data uses NO RNG)

# Synthetic task — next token is a closed-form modular function of a window of prior tokens.
# Pure function of (global_step, row): resume regenerates the identical batch for step N with
# no sampler to checkpoint (a precondition for Proof 4's trajectory-continuity assertion).
TASK_VOCAB = 64                      # token id range [0, TASK_VOCAB)
TASK_WINDOW = 2                      # recurrence looks back this many tokens
TASK_COEFFS = (1, 1)                 # next = (Σ_j C_j · x[t-1-j] + BIAS) mod VOCAB  (Fibonacci-mod)
TASK_BIAS = 0
# seed tokens (first TASK_WINDOW positions of each row) — unpredictable per row, so no
# position→token shortcut; content varies with global step so the task can't be memorized by
# position. x[i] = (gid·SEED_A + i·SEED_B + SEED_C) mod VOCAB, gid = step·1_000_003 + row_start + row.
SEED_A, SEED_B, SEED_C = 131, 17, 5

# Loose bounds derived from the CPU pre-flight (2 procs, gloo, fp32). See NOTES.md for the
# exact --local numbers and their derivation. These are UPPER bounds / tolerances, not
# invariants; a broken optimizer step or reduction still leaves loss above the ceiling.
EXPECTED_LOSS_CEILING = 4.10         # < ln(VOCAB)=4.159 (uniform-init step-0); pinned from --local
LOSS_DROP_MARGIN = 0.05              # final-window mean must be below step-0 by at least this
REDUCE_TOL = 2e-4                    # Proof 2 fp32 gathered-grad tolerance (looser than probe's 1e-9 fp64)
SMOOTH_WINDOW = 20                   # steps in the final smoothing window for convergence


# ==========================================================================================
# Acceptance report — see the acceptance-report skill (.claude/skills/acceptance-report). One
# `Check` record per proof, a single workload-agnostic renderer, exit code derived LAST. Checks
# RECORD an outcome (status + measured value); they do not assert-and-raise, so the report always
# renders — including on failure. Machine sentinels (FSDP_*_OK, FSDP_BR4_COMPLETE, …) are kept
# as-is above/around the checks for grep/receipts; this report is a plain-English layer on top
# of them.
# ==========================================================================================
WORKLOAD = "FSDP2 TRAINING (BR-4)"

# Status enum — exactly these five (see format spec §"Status enum").
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
# Synthetic task — closed-form, no RNG, pure function of global step.
# ==========================================================================================
def synth_batch(step: int, row_start: int, row_count: int, seq: int, device) -> torch.Tensor:
    """Token grid of shape (row_count, seq), long. Rows are global ids [row_start, +row_count).

    Deterministic in (step, row): seed the first TASK_WINDOW columns from a closed form, then
    roll the fixed modular recurrence forward. No RNG anywhere, so the CPU-derived loss ceiling
    transfers and step N regenerates identically on resume.
    """
    gid = (step * 1_000_003 + row_start
           + torch.arange(row_count, device=device, dtype=torch.long)).unsqueeze(1)  # (R,1)
    x = torch.empty(row_count, seq, dtype=torch.long, device=device)
    for i in range(TASK_WINDOW):
        x[:, i] = ((gid.squeeze(1) * SEED_A + i * SEED_B + SEED_C) % TASK_VOCAB)
    for t in range(TASK_WINDOW, seq):
        acc = torch.full((row_count,), TASK_BIAS, dtype=torch.long, device=device)
        for j, c in enumerate(TASK_COEFFS):
            acc = acc + c * x[:, t - 1 - j]
        x[:, t] = acc % TASK_VOCAB
    return x


def batch_loss(model, tokens: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy, mean over all tokens (matches the reduction convention so a
    mean-of-per-rank-means equals the global mean when local batches are equal-sized)."""
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    logits = model(inp)                                    # (B, L-1, V)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))


# ==========================================================================================
# Model — a small in-code decoder transformer. fully_shard is applied per block + top-level.
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
    """CPU-init-under-fixed-seed then .to(device) — identical params on every rank without a
    cross-rank RNG-stream issue (init RNG differs CPU vs CUDA)."""
    torch.manual_seed(INIT_SEED)
    m = TinyTransformer(TASK_VOCAB, args.dim, args.heads, args.layers, args.seq)
    return m.to(device)


def wrap_fsdp(model: nn.Module, mesh, mp: bool) -> nn.Module:
    """Per-block fully_shard + one top-level call. Wrapping strategy is pinned in NOTES.md
    because it changes the exact local≈full/world ratio (padding of the last dim-0 shard)."""
    # mp_policy defaults to a sentinel inside fully_shard; pass it only when enabled (None breaks it).
    kw = {"mesh": mesh}
    if mp:
        kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.blocks:
        fully_shard(blk, **kw)
    fully_shard(model, **kw)
    return model


def build_stress_model_sharded(args, mesh):
    """Rung-4 ONLY: build a deep+wide model on META device, then fully_shard so each rank
    materializes only its 1/world slice. Materializing the full DDP-sized model in one process's
    HOST memory first would host-OOM before it ever shards — self-inflicting the exact
    host-OOM-≠-CUDA-OOM confusion the memory story must avoid. Params init lazily post-shard via
    `to_empty` + reset (init values are irrelevant to a memory-envelope measurement)."""
    with torch.device("meta"):
        m = TinyTransformer(TASK_VOCAB, args.dim, args.heads, args.layers, args.seq)
    kw = {"mesh": mesh}
    if args.mp:
        kw["mp_policy"] = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in m.blocks:
        fully_shard(blk, **kw)
    fully_shard(m, **kw)
    m.to_empty(device=mesh.device_type)
    for p in m.parameters():
        with torch.no_grad():
            p.normal_(0, 0.02)
    return m


# ==========================================================================================
# Proof 1 — sharding is real (FSDP_SHARDING_OK).
# ==========================================================================================
def proof1_sharding(model, world: int, device, opt) -> Check:
    """Per-rank PARAMETER STORAGE ≈ total/world (not peak memory — FSDP all-gathers full layers
    transiently, so peak ≫ full/world). Also logs the full training-state envelope (open-q #17)
    after moments exist: params + grads (post-reduce-scatter shard) + optimizer state, vs the
    full-model counterfactual (P+G+2P)·4B. Records a Check; the OK sentinel prints only on PASS."""
    local = sum(p.to_local().numel() for p in model.parameters())
    full = sum(p.full_tensor().numel() for p in model.parameters())   # collective — all ranks
    ratio = local / (full / world)

    # Padding tolerance: the last dim-0 shard is padded to divide evenly, so local can slightly
    # exceed full/world. With the pinned shape padding is <~1% of a shard (NOTES.md); allow 3%.
    tol = 0.03
    vacuous = world == 1
    ok = vacuous or abs(ratio - 1.0) <= tol

    # Full-state envelope (bytes), measured on the persistent sharded state after one opt step.
    esz = next(model.parameters()).to_local().element_size()
    local_param_b = local * esz
    local_grad_b = sum(p.grad.to_local().numel() for p in model.parameters()
                       if p.grad is not None) * esz
    local_optim_b = 0
    for st in opt.state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and hasattr(st[k], "to_local"):
                local_optim_b += st[k].to_local().numel() * st[k].element_size()
            elif k in st:
                local_optim_b += st[k].numel() * st[k].element_size()
    full_state_b = (full + full + 2 * full) * 4    # (P + G + 2·P optim) fp32
    local_state_b = local_param_b + local_grad_b + local_optim_b
    mem_gb = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0

    if dist.get_rank() == 0 and ok and not vacuous:
        print(f"FSDP_SHARDING_OK world={world} full={full} local={local} ratio={ratio:.4f} "
              f"mem={mem_gb:.2f}GB state_local={local_state_b/2**30:.4f}GB "
              f"state_full={full_state_b/2**30:.4f}GB "
              f"[param={local_param_b} grad={local_grad_b} optim={local_optim_b} bytes]",
              flush=True)

    status = NA if vacuous else (PASS if ok else FAIL)
    measured = (f"per-rank storage = {local_state_b/2**30:.4f} GB vs full-model "
                f"{full_state_b/2**30:.4f} GB; shard ratio {ratio:.4f} of full/world "
                f"(params {local}/{full})")
    return Check(
        name="Parameters, gradients and optimizer state are actually sharded",
        status=status,
        measured=measured,
        threshold=f"shard ratio within {tol:.0%} of 1.0 (each rank holds ≈1/{world} of the model)",
        what_why="This is the property that makes FSDP different from DDP: each GPU keeps only "
                 "its slice of the model, gradients and Adam moments. Without it, every GPU holds "
                 "the whole model and a large model that should fit will run out of memory.",
        sufficient=f"A ratio near 1.0 means storage really is split {world} ways; DDP or a broken "
                   f"wrap would show ratio ≈ {world} (the full model on every rank). At world=1 "
                   "there is nothing to split, so this is N/A, not a pass.",
        likely_means="The model was not sharded — most often fully_shard failed to apply or the "
                     "device mesh has the wrong size. Send the FSDP_VERSIONS line and this report.",
    )


# ==========================================================================================
# Proof 2 — gradient reduction (reduce-scatter) is correct (FSDP_REDUCE_OK).
# ==========================================================================================
def proof2_reduce(args, mesh, device, world: int) -> Check:
    """Build an FSDP model and a single-process reference from an IDENTICAL init (same seed,
    same process — no cross-rank RNG issue). One backward, fp32, AMP off, mean-reduction loss
    on a global batch split across ranks. Gather each reduce-scattered grad with full_tensor()
    (collective) and assert max|g_full − g_ref| < REDUCE_TOL vs the reference computed
    single-process over the same global batch. Compared after the FIRST backward, before any
    optimizer step — isolates reduction from the optimizer and avoids the FSDP+AMP
    step-compounding trap. The K-step endpoint comparison is never done."""
    rank = dist.get_rank()
    local_b = args.reduce_batch
    global_b = local_b * world

    # Reference: unwrapped, fp32, full global batch, single process (rank 0 result is the truth;
    # every rank builds it identically for a local comparison too).
    ref = build_model(args, device)
    ref_tokens = synth_batch(0, 0, global_b, args.seq, device)
    ref.zero_grad(set_to_none=True)
    batch_loss(ref, ref_tokens).backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in ref.named_parameters()}

    # FSDP model from the SAME seed → identical init; fp32, no mixed precision.
    fm = wrap_fsdp(build_model(args, device), mesh, mp=False)
    my_tokens = synth_batch(0, rank * local_b, local_b, args.seq, device)
    fm.zero_grad(set_to_none=True)
    batch_loss(fm, my_tokens).backward()

    max_diff = 0.0
    for n, p in fm.named_parameters():
        g_full = p.grad.full_tensor()                      # collective — call on all ranks
        max_diff = max(max_diff, (g_full - ref_grads[n]).abs().max().item())

    vacuous = world == 1
    ok = max_diff < REDUCE_TOL
    if rank == 0 and ok and not vacuous:
        print(f"FSDP_REDUCE_OK world={world} grad_diff={max_diff:.3e} tol={REDUCE_TOL:.1e} "
              f"global_batch={global_b}", flush=True)

    status = NA if vacuous else (PASS if ok else FAIL)
    return Check(
        name="Gradients are combined correctly across GPUs (reduce-scatter)",
        status=status,
        measured=f"largest gradient difference vs a single-process reference = {max_diff:.3e} "
                 f"(global batch {global_b})",
        threshold=f"max gradient difference < {REDUCE_TOL:.1e}",
        what_why="Each GPU computes gradients on its own slice of the batch; FSDP must sum-and-"
                 "split them so every rank ends up with the correct averaged gradient. If this is "
                 "even slightly wrong, training looks like it runs but silently learns the wrong "
                 "thing — the hardest kind of bug to notice.",
        sufficient=f"Matching a trusted single-process gradient to within {REDUCE_TOL:.1e} means "
                   "the collective is numerically correct, not just non-crashing. A failure shows "
                   "as a difference above the tolerance. At world=1 the reduce-scatter is a no-op, "
                   "so this is N/A, not a pass.",
        likely_means="The gradient reduce-scatter produced wrong values — possibly a runtime/NCCL "
                     "mismatch or a wrapping bug. Do NOT trust training numbers from this run; "
                     "send this report and the FSDP_VERSIONS line to support.",
    )


# ==========================================================================================
# Proof 3 — the loop trains (FSDP_TRAIN_OK). Scope: shows the loop runs and converges; does
# NOT re-prove reduction (a subtly-wrong reduction can still fall — that's Proof 2's job).
# ==========================================================================================
def train_loop(model, opt, sched, args, mesh, device, world, start_step, mlf, ckpt_dir):
    """Run steps [start_step, args.steps). Emits Proof 1 after the first optimizer step (moments
    allocated), logs the loss curve to MLflow, optionally checkpoints for the max_retries test,
    and (if FSDP_FAIL_AT_STEP is set) hard-exits to exercise platform resume (open-q #10).

    Returns (train_check, sharding_check) — the convergence Check and the Proof-1 record it
    captured mid-loop. A non-finite loss is recorded as a FAIL (no raise) so the report renders."""
    rank = dist.get_rank()
    losses = []
    sharding_check = None
    fail_at = int(os.environ.get("FSDP_FAIL_AT_STEP", "-1"))
    save_every = args.save_every

    train_name = "The training loop runs a real step and the loss goes down"
    train_thresh = (f"final-window mean below step-0 by ≥ {LOSS_DROP_MARGIN} AND below "
                    f"ceiling {EXPECTED_LOSS_CEILING}")
    train_what = ("A full forward → backward → optimizer step on the sharded model, repeated, "
                  "must actually reduce the loss. This is the end-to-end proof that a model can "
                  "be trained on this platform, not just that the pieces initialize.")
    train_likely = ("Loss did not fall as expected. If the reduce-scatter check passed, this is "
                    "almost always tuning/precision (learning rate, steps, mixed precision) — NOT "
                    "a platform fault; do not report 'FSDP doesn't work on AIR'. Re-triage K/LR "
                    "before escalating.")

    step0_loss = None
    for step in range(start_step, args.steps):
        tokens = synth_batch(step, rank * args.batch, args.batch, args.seq, device)
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(model, tokens)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        lv = loss.item()
        losses.append(lv)
        if step0_loss is None:
            step0_loss = lv

        # Proof 1 fires once, after the first step so Adam moments exist for the envelope.
        if sharding_check is None:
            sharding_check = proof1_sharding(model, world, device, opt)

        if lv != lv or abs(lv) == float("inf"):            # NaN/±inf guard — record, don't raise
            train_check = Check(
                name=train_name, status=FAIL,
                measured=f"non-finite loss at step {step}: {lv}", threshold=train_thresh,
                what_why=train_what,
                sufficient="A finite, falling loss is required; NaN/inf means the step diverged.",
                likely_means="The loss became NaN or inf — usually too high a learning rate or a "
                             "mixed-precision overflow. Lower --lr or drop --mp and retry.")
            return train_check, sharding_check

        if rank == 0:
            if mlf:
                mlf.log_metric("train_loss", lv, step=step)
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                print(f"[step {step}] loss={lv:.4f}", flush=True)

        if save_every and ckpt_dir and step > start_step and step % save_every == 0:
            collective_dcp_save(model, opt, sched, step, ckpt_dir, world, device)

        if fail_at >= 0 and step == fail_at and start_step == 0:
            # Whole-rank hard exit to let the PLATFORM's max_retries decide (torchrun
            # --max-restarts=0 in the YAML isolates this from torchrun's elastic restart).
            # Guarded on start_step == 0 so the fault fires ONCE, on the cold run: the
            # resumed attempt (start_step > 0 via --auto-resume) skips it and completes, so
            # the open-q #10 test TERMINATES in success (attempt 2 emits RESUMED_FROM_STEP +
            # FSDP_TRAIN_OK) instead of re-failing at the same step every retry.
            print(f"[rank{rank}] FSDP_FORCED_EXIT at step {step} (open-q #10 max_retries test)",
                  flush=True)
            sys.exit(137)

    # Convergence: smoothed final-window mean below step-0 by a margin AND below a loose ceiling.
    win = min(SMOOTH_WINDOW, len(losses))
    final = sum(losses[-win:]) / win
    drop = step0_loss - final
    ok = (final < step0_loss - LOSS_DROP_MARGIN) and (final < EXPECTED_LOSS_CEILING)
    if rank == 0:
        if ok:
            print(f"FSDP_TRAIN_OK step0={step0_loss:.4f} final={final:.4f} drop={drop:.4f} "
                  f"ceiling={EXPECTED_LOSS_CEILING} steps={args.steps}", flush=True)
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
    return train_check, sharding_check


# ==========================================================================================
# Proof 4 — checkpoint + resume is correct (FSDP_CKPT_RESUME_OK).
# ==========================================================================================
def _solo_probe_write(path: str, timeout_s: int = 10) -> bool:
    """Write+fsync+remove a tiny sentinel, guarded by a per-process alarm. A SOLO write IS
    safely abortable (unlike the collective DCP save). Proves permission/403 (BR-2's failure
    mode), not capacity/throughput for a real multi-GB shard."""
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
    except Exception as e:                                 # noqa: BLE001 — any failure ⇒ probe fail
        print(f"[rank{dist.get_rank()}] probe write FAILED: {type(e).__name__}: {e}", flush=True)
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def collective_dcp_save(model, opt, sched, step, ckpt_dir, world, device) -> bool:
    """Collective probe-write-first: every rank writes a solo-timeout-guarded sentinel to its
    shard path, all-reduce (MIN) the success flag, and BRANCH COLLECTIVELY — every rank enters
    DCP or every rank skips. That identical branch on the all-reduced flag is the desync guard;
    the DCP save is NEVER per-rank aborted (aborting one rank inside a collective desyncs the PG
    and the next NCCL collective hangs to TIMEDOUT). If any probe fails ⇒ skip, blocked-on-BR-2."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    rank = dist.get_rank()
    probe = _solo_probe_write(os.path.join(ckpt_dir, f".probe_rank{rank}"))
    flag = torch.tensor([1.0 if probe else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)            # any failure ⇒ min is 0 on every rank
    if flag.item() < 1.0:
        if rank == 0:
            print(f"FSDP_CKPT_PROBE_FAILED step={step} — blocked-on-BR-2 (UC-volume 403); "
                  f"skipping DCP save collectively", flush=True)
        return False
    msd, osd = get_state_dict(model, opt)                  # sharded DTensors (model + optim together)
    state = {"model": msd, "optim": osd, "step": step}
    if sched is not None:
        state["sched"] = sched.state_dict()
    dcp.save(state, checkpoint_id=os.path.join(ckpt_dir, f"step_{step}"))
    if rank == 0:
        print(f"[rank0] DCP save OK step={step} → {ckpt_dir}/step_{step}", flush=True)
    return True


def _fingerprint(model, opt, fp_name: str) -> str:
    """Bit-exact hash of one fixed param (gathered) + its Adam moments. Folding moments in means a
    dropped-moment regression trips the bit-exact check, not only the softer loss-continuity check.

    Uses the canonical `get_state_dict` mapping keyed by parameter NAME, not `opt.state[param]`:
    FSDP2 reshards params into fresh DTensor objects across forward/backward, so the live parameter
    is not identity-equal to the optimizer's state key and an object lookup silently misses the
    moments. The named state_dict is stable and returns the full (gathered) DTensors — this must be
    called collectively on every rank."""
    from torch.distributed.checkpoint.state_dict import get_state_dict

    msd, osd = get_state_dict(model, opt)
    def _gather(t):                                        # DTensor → full; plain tensor → itself
        return t.full_tensor() if hasattr(t, "full_tensor") else t
    parts = [_gather(msd[fp_name]).detach().to(torch.float64).cpu().numpy().tobytes()]
    st = osd.get("state", {}).get(fp_name, {})
    for k in ("exp_avg", "exp_avg_sq"):
        if k in st:
            parts.append(_gather(st[k]).detach().to(torch.float64).cpu().numpy().tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def proof4_checkpoint_resume(args, mesh, device, world, ckpt_dir) -> Check:
    """Two-phase within one run: (1) train N steps, save (probe-gated), record loss@N +
    fingerprint (params + optimizer moments); (2) reconstruct a fresh FSDP model + optimizer +
    scheduler, load, check the fingerprint is bit-identical and that the step-N loss matches the
    pre-save trajectory (next-step loss, not a cold-start value). Data is a pure function of
    global step, so the resumed step feeds the identical batch — the comparison is well-posed.

    Returns a Check: PASS ("ok"), BLOCKED (probe failed, blocked-on-BR-2), or FAIL (a real
    checkpoint fault — fingerprint/loss mismatch or a raised exception, captured by the caller)."""
    import numpy  # noqa: F401 — required by .numpy() in _fingerprint; fail loud here if absent
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    rank = dist.get_rank()
    n = args.ckpt_steps
    save_dir = os.path.join(ckpt_dir, "proof4")

    # Phase 1: train N steps on a fresh model+opt+sched.
    m = wrap_fsdp(build_model(args, device), mesh, mp=False)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, n // 2), gamma=0.5)
    for step in range(n):
        tok = synth_batch(step, rank * args.batch, args.batch, args.seq, device)
        opt.zero_grad(set_to_none=True)
        batch_loss(m, tok).backward()
        opt.step()
        sched.step()
    # loss@N: forward on the step-N batch WITHOUT stepping (the pre-save trajectory value).
    with torch.no_grad():
        tok_n = synth_batch(n, rank * args.batch, args.batch, args.seq, device)
        loss_at_n_pre = batch_loss(m, tok_n).item()
    fp_name = "blocks.0.mlp.0.weight"                      # a fixed sharded param that always trains
    fp_pre = _fingerprint(m, opt, fp_name)

    p4_name = "A checkpoint saves and resumes exactly where it left off"
    p4_thresh = "params + optimizer moments bit-identical AND resume loss within 1e-4"
    p4_what = ("Long jobs must survive interruption. On resume the model, gradients moments and "
               "scheduler must come back bit-for-bit; otherwise a 'resumed' run silently restarts "
               "from worse weights and wastes the compute already spent.")

    if not collective_dcp_save(m, opt, sched, n, save_dir, world, device):
        return Check(
            name=p4_name, status=BLOCKED,
            measured="checkpoint probe-write failed (UC-volume 403)",
            threshold=p4_thresh, what_why=p4_what,
            sufficient="Blocked by an external precondition (BR-2 UC-volume write permission), "
                       "not a fault in checkpoint/resume itself. The BR-4 receipt still stands.",
            likely_means="The checkpoint directory could not be written — a permissions/BR-2 "
                         "block on the UC volume, not a training failure.")

    # Phase 2: fresh model+opt+sched, load, assert bit-identical + trajectory continuity.
    m2 = wrap_fsdp(build_model(args, device), mesh, mp=False)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=args.lr)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=max(1, n // 2), gamma=0.5)
    # optimizer state must exist before set_state_dict can load into it — one dummy step, then
    # DCP load overwrites it with the saved (correct) state.
    tok0 = synth_batch(0, rank * args.batch, args.batch, args.seq, device)
    batch_loss(m2, tok0).backward()
    opt2.step()
    opt2.zero_grad(set_to_none=True)

    msd, osd = get_state_dict(m2, opt2)
    state = {"model": msd, "optim": osd, "step": 0, "sched": sched2.state_dict()}
    dcp.load(state, checkpoint_id=os.path.join(save_dir, f"step_{n}"))
    set_state_dict(m2, opt2, model_state_dict=msd, optim_state_dict=osd)
    sched2.load_state_dict(state["sched"])

    fp_post = _fingerprint(m2, opt2, fp_name)
    with torch.no_grad():
        loss_at_n_post = batch_loss(m2, tok_n).item()

    fp_match = fp_pre == fp_post
    loss_diff = abs(loss_at_n_post - loss_at_n_pre)
    loss_match = loss_diff < 1e-4
    ok = fp_match and loss_match
    if rank == 0 and ok:
        print(f"FSDP_CKPT_RESUME_OK fingerprint_match={fp_match} "
              f"resumed_from=loss@{n}={loss_at_n_pre:.6f} (post={loss_at_n_post:.6f})", flush=True)

    return Check(
        name=p4_name,
        status=PASS if ok else FAIL,
        measured=f"params/moments bit-identical={fp_match}; resume loss {loss_at_n_post:.6f} vs "
                 f"pre-save {loss_at_n_pre:.6f} (diff {loss_diff:.2e})",
        threshold=p4_thresh,
        what_why=p4_what,
        sufficient="A matching fingerprint proves the weights AND Adam moments round-tripped "
                   "exactly; the matching next-step loss proves the trajectory continues rather "
                   "than restarting. A failure shows as a fingerprint mismatch or a loss spike "
                   "on resume.",
        likely_means="Checkpoint saved but resume did not reproduce the pre-save state — usually "
                     "optimizer/scheduler state was dropped so the moments reset and the loss "
                     "spikes. Capture this report and the save/load lines for support.",
    )


# ==========================================================================================
# MLflow — tracking endpoint only (log_metric/log_param). This is a DIFFERENT egress path from
# the root-storage artifact-upload path that caused the `07` TIMEDOUT saga; confirm reachability
# on the target before relying on it. Degrades to stdout when mlflow is absent (local).
# ==========================================================================================
class _MlflowReceipt:
    """Thin adapter over MlflowClient bound to the AIR-injected run. Same call sites as the fluent
    API (`log_metric`/`log_param`) so the training loop is unchanged. The client API — NOT
    `mlflow.start_run(run_id=…)` — is deliberate: resuming the launcher-owned run transitions its
    status and failed SILENTLY on the job plane (see burn.py / nccl_allreduce_ctypes.py, the repo's
    receipt pattern). It also avoids the fluent API spinning up a stray local run when there is no
    tracking run to attach to."""
    def __init__(self, client, run_id: str):
        self._c = client
        self._run_id = run_id

    def log_metric(self, key, value, step=None):
        self._c.log_metric(self._run_id, key, value, step=step if step is not None else 0)

    def log_param(self, key, value):
        self._c.log_param(self._run_id, key, value)


def open_mlflow(rank: int):
    if rank != 0:
        return None
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        # No AIR-injected run to attach to (e.g. --local). Do NOT let the fluent API start a stray
        # local run just to log into — skip and fall back to stdout.
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
    except Exception as e:                                 # noqa: BLE001
        print(f"[rank0] mlflow tracking endpoint NOT reachable: {e} — stdout only", flush=True)
        return None


# ==========================================================================================
# Rung 4 (stretch) — DDP-OOM / FSDP-fit counterfactual. Half-closes open-q #17: proves sharding
# defers OOM for a STATE-DOMINATED model of size X; does NOT validate the customer FM (egress-gated).
# ==========================================================================================
def memprobe(rank, world, args, mesh, device):
    """One arm per invocation (`--arm fsdp|ddp`). The stress model must be STATE-dominated
    (deep+wide, short seq, small batch, activation-checkpointing OFF) so the OOM is driven by the
    param/grad/optim state FSDP actually shards — if it OOMs on activations (identical under FSDP
    and DDP) the control proves nothing. Reports peak CUDA GB or catches the CUDA OOM; host-OOM
    (exit 137, no traceback) vs CUDA-OOM must be labeled by the reader."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tag = args.arm
    try:
        if args.arm == "fsdp":
            model = build_stress_model_sharded(args, mesh)
        else:  # ddp baseline — full replicated model, the control that should OOM first
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = build_model(args, device)
            model = DDP(model) if world > 1 else model
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        tok = synth_batch(0, rank * args.batch, args.batch, args.seq, device)
        opt.zero_grad(set_to_none=True)
        batch_loss(model, tok).backward()
        opt.step()
        peak = (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0
        if rank == 0:
            print(f"FSDP_MEMPROBE_OK arm={tag} peak_gb={peak:.2f} "
                  f"shape=L{args.layers}-d{args.dim}-h{args.heads}-s{args.seq} "
                  f"batch={args.batch} world={world} — sharding defers OOM for a state-dominated "
                  f"model of size X; B300 decision still needs the customer FM's real "
                  f"param/optim-state profile (open-q #17 half-closed)", flush=True)
        return True
    except torch.cuda.OutOfMemoryError as e:                # CUDA OOM (NOT host-OOM/exit137)
        if rank == 0:
            print(f"FSDP_MEMPROBE_CUDA_OOM arm={tag} — {type(e).__name__}: CUDA out of memory "
                  f"(labeled CUDA-OOM, not host-OOM). Expected for arm=ddp; if arm=fsdp also OOMs, "
                  f"the model exceeds what sharding can defer at world={world}", flush=True)
        return False


# ==========================================================================================
# Worker — one rank. Runs Proofs 2, (1+3 via the loop), and 4; emits completion lines.
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
    runtime_str = (f"torch {torch.__version__}, nccl {nccl}, cuda {torch.version.cuda}, "
                   f"fully_shard={callable(fully_shard)}")
    if rank == 0:
        # Version-scoped: the sharding assertion + DCP API are pinned to the runtime build.
        print(f"FSDP_VERSIONS torch={torch.__version__} nccl={nccl} cuda={torch.version.cuda} "
              f"fully_shard={callable(fully_shard)} world={world} device={mesh_dev}", flush=True)

    if args.memprobe:                                      # rung 4 — one arm, then exit
        memprobe(rank, world, args, mesh, device)
        dist.barrier()
        dist.destroy_process_group()
        return 0

    mlf = open_mlflow(rank)
    ckpt_dir = args.ckpt_dir

    # Each proof RECORDS a Check (template principle 1); exceptions become FAIL records so the
    # report always renders. Collectives run on every rank; only rank 0 renders the report.
    # Proof 2 first — a fresh model, one backward, before training mutates anything.
    try:
        reduce_check = proof2_reduce(args, mesh, device, world)
    except Exception as e:                                 # noqa: BLE001
        reduce_check = _fail_from_exc(
            "Gradients are combined correctly across GPUs (reduce-scatter)",
            f"max gradient difference < {REDUCE_TOL:.1e}",
            "FSDP must sum-and-split per-rank gradients so every rank gets the correct average.",
            "The reduce-scatter proof raised before producing a number — check FSDP_VERSIONS "
            "for a runtime/NCCL mismatch and send this report to support.", e)

    # Proofs 1 + 3 — the training model + loop (Proof 1 recorded after the first step).
    model = wrap_fsdp(build_model(args, device), mesh, mp=args.mp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.steps // 3), gamma=0.5)

    # Auto-resume (open-q #10 / #8): if a checkpoint survives, LOAD it (not just report the step) so
    # a resumed run continues the trajectory instead of silently retraining from cold weights.
    start_step = 0
    if args.auto_resume and ckpt_dir:
        start_step = _maybe_resume(rank, model, opt, sched, ckpt_dir, args, device)

    if rank == 0 and mlf:
        mlf.log_param("torch_version", torch.__version__)
        mlf.log_param("world", world)
        mlf.log_param("shape", f"L{args.layers}-d{args.dim}-h{args.heads}-s{args.seq}-v{TASK_VOCAB}")
        mlf.log_param("mixed_precision", args.mp)
    train_check, sharding_check = train_loop(
        model, opt, sched, args, mesh, device, world, start_step, mlf, ckpt_dir)

    # Completion line for BR-4 (Proofs 1+2+3 only). THE acceptance receipt — emitted ONLY when all
    # three STRICTLY PASS at world>=2. At world=1 sharding + reduce-scatter are vacuous (N/A), so the
    # receipt must NOT print: anyone grepping the sentinel would read a false acceptance for proofs
    # the run never performed (format-spec: "a run can't claim a proof it didn't perform"). The
    # world=1 API-gate run still renders ACCEPTED WITH CAVEATS in the report; it just isn't BR-4.
    br4 = (world >= 2 and sharding_check.status == PASS
           and reduce_check.status == PASS and train_check.status == PASS)
    if rank == 0 and br4:
        print("FSDP_BR4_COMPLETE proofs=1,2,3 (sharding+reduce+convergence)", flush=True)

    checks = [sharding_check, reduce_check, train_check]
    sentinels = ["FSDP_BR4_COMPLETE" if br4 else "FSDP_BR4_INCOMPLETE"]

    # Proof 4 (suite #8) — may be BLOCKED (blocked-on-BR-2) without failing BR-4.
    if args.proof4 and ckpt_dir:
        try:
            ckpt_check = proof4_checkpoint_resume(args, mesh, device, world, ckpt_dir)
        except Exception as e:                             # noqa: BLE001
            ckpt_check = _fail_from_exc(
                "A checkpoint saves and resumes exactly where it left off",
                "params + optimizer moments bit-identical AND resume loss within 1e-4",
                "On resume the model and optimizer state must come back bit-for-bit or a resumed "
                "job silently restarts from worse weights.",
                "Checkpoint save or load raised — capture this report and the DCP save/load lines "
                "for support.", e)
        checks.append(ckpt_check)
        if rank == 0:
            if ckpt_check.status == PASS and br4:
                print("FSDP_SUITE8_COMPLETE proofs=1,2,3,4 (adds checkpoint/resume)", flush=True)
                sentinels.append("FSDP_SUITE8_COMPLETE")
            elif ckpt_check.status == BLOCKED:
                print("FSDP_SUITE8_BLOCKED proof4=blocked-on-BR-2 (BR-4 receipt stands via "
                      "FSDP_BR4_COMPLETE)", flush=True)
                sentinels.append("FSDP_SUITE8_BLOCKED")
    elif args.proof4:
        checks.append(Check(
            name="A checkpoint saves and resumes exactly where it left off",
            status=SKIPPED, measured="no --ckpt-dir provided",
            threshold="params + optimizer moments bit-identical AND resume loss within 1e-4",
            what_why="Checkpoint/resume lets a long job survive interruption without losing "
                     "progress.",
            sufficient="Deliberately not run this invocation: --proof4 was set but no checkpoint "
                       "directory was given, so there is nowhere to save.",
            likely_means=""))

    dist.barrier()

    # Render the plain-English acceptance report from the records (rank 0 only) and derive the
    # exit code last. Guard so the renderer itself can never swallow the report.
    exit_code = 0
    if rank == 0:
        scope = "smoke" if world == 1 else "acceptance"
        shape = f"world={world}, {mesh_dev}"
        try:
            # Prefer MLFLOW_RUN_ID (AIR-injected; the join key a confirmer uses to find this run
            # in Jobs/MLflow) over the display name; fall back to the name, then "local".
            run_id = (os.environ.get("MLFLOW_RUN_ID")
                      or os.environ.get("MLFLOW_RUN_NAME") or "local")
            exit_code = render_report(
                checks, run_id=run_id,
                profile=("local-cpu" if args.local else "air"),
                shape=shape, scope=scope, runtime=runtime_str,
                sentinels=" ".join(sentinels),
                test_id="fsdp")  # utils/verification/results/registry.py
        except Exception:                                  # noqa: BLE001 — never lose the verdict
            _tb.print_exc()
            exit_code = 1

    dist.destroy_process_group()
    # Exit code is derived from the rendered verdict (rank 0 only). sys.exit so it propagates in
    # BOTH launch paths: torchrun reads the process exit; mp.spawn (--local) re-raises a non-zero
    # child exit. Non-rank-0 procs exit 0 (only rank 0 rendered the verdict).
    if rank == 0 and exit_code:
        sys.exit(exit_code)
    return exit_code


def _maybe_resume(rank, model, opt, sched, ckpt_dir, args, device) -> int:
    """Find the newest surviving checkpoint and LOAD model+optim+scheduler into the live objects,
    returning the step to resume from (prints RESUMED_FROM_STEP=N) or 0 (COLD_START).

    This is what makes the open-q #10 max_retries test meaningful: a resumed *new container* must
    continue the trajectory, not silently retrain from cold weights. Reading it back in a fresh
    container is mechanism (b) — the platform's whole-job resubmit — and hard-depends on a
    container-durable path (a /tmp checkpoint would not survive)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    latest = -1
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.startswith("step_"):
                try:
                    latest = max(latest, int(name.split("_", 1)[1]))
                except ValueError:
                    pass
    if latest < 0:
        if rank == 0:
            print("COLD_START", flush=True)
        return 0

    # Prime optimizer state (one dummy step) so set_state_dict has slots to load into, then load.
    tok0 = synth_batch(0, rank * args.batch, args.batch, args.seq, device)
    batch_loss(model, tok0).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    msd, osd = get_state_dict(model, opt)
    state = {"model": msd, "optim": osd, "step": latest, "sched": sched.state_dict()}
    dcp.load(state, checkpoint_id=os.path.join(ckpt_dir, f"step_{latest}"))
    set_state_dict(model, opt, model_state_dict=msd, optim_state_dict=osd)
    sched.load_state_dict(state["sched"])
    if rank == 0:
        print(f"RESUMED_FROM_STEP={latest}", flush=True)
    return latest


# ==========================================================================================
# Entry point.
# ==========================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true",
                   help="single-host CPU/gloo pre-flight; spawns --local-world procs (real FSDP sharding)")
    p.add_argument("--local-world", type=int, default=2, help="world size for --local (≥2 to be non-vacuous)")
    p.add_argument("--master-port", type=int, default=29520)
    # model / task shape
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    # training
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=16, help="per-rank batch (global = batch·world)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--reduce-batch", type=int, default=8, help="per-rank batch for Proof 2")
    p.add_argument("--mp", action="store_true", help="MixedPrecisionPolicy (bf16 compute / fp32 master) for the loop")
    # checkpoint / resume
    p.add_argument("--ckpt-dir", default=None, help="checkpoint dir (UC volume on AIR; /tmp fallback)")
    p.add_argument("--proof4", action="store_true", help="run Proof 4 (checkpoint/resume, suite #8)")
    p.add_argument("--ckpt-steps", type=int, default=20, help="steps before the Proof 4 save")
    p.add_argument("--save-every", type=int, default=0, help="save a checkpoint every N training steps (0=off)")
    p.add_argument("--auto-resume", action="store_true", help="resume from newest checkpoint (open-q #10)")
    # rung 4 (stretch) — DDP-OOM/FSDP-fit counterfactual (open-q #17). Dial the model deep+wide,
    # SHORT seq, small batch so state (not activations) dominates; disable activation checkpointing.
    p.add_argument("--memprobe", action="store_true", help="rung 4: memory-envelope counterfactual, one arm then exit")
    p.add_argument("--arm", choices=["fsdp", "ddp"], default="fsdp", help="memprobe arm (ddp is the control that should OOM first)")
    args = p.parse_args()

    if args.local:
        # Real FSDP sharding on CPU/gloo across spawned procs — answers the rung-0 question.
        if args.ckpt_dir is None and args.proof4:
            args.ckpt_dir = "/tmp/fsdp_local_ckpt"
        import torch.multiprocessing as mp
        # mp.spawn re-raises a non-zero child exit as an exception; rank 0's exit code (from the
        # rendered verdict) surfaces via that. Treat a clean join as success.
        try:
            mp.spawn(worker, args=(args.local_world, args), nprocs=args.local_world, join=True)
        except Exception:                                  # noqa: BLE001
            _tb.print_exc()
            return 1
        return 0
    else:
        # torchrun-launched: RANK/WORLD_SIZE/LOCAL_RANK from env.
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        return worker(rank, world, args) or 0


if __name__ == "__main__":
    sys.exit(main())
