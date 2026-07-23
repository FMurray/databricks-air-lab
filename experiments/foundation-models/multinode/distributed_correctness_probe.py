"""Provably-correct distributed computation probe (2 proofs, both assertion-gated).

Proof 1 — exact sharded matmul, pre-registered checksum:
  Rank r builds integer-valued fp64 shards A_r (N x Ks), B_r (Ks x M) from a closed-form
  formula (no RNG -> platform/version independent). C = sum_r A_r @ B_r via all_reduce.
  All values stay far below 2^53, so every intermediate is EXACT: C must equal the
  reference bit-for-bit and sum(C) must equal the checksum pre-computed off-cluster
  (colsum(A_r) . rowsum(B_r) identity — a different computation path than the matmul).
  A missing/duplicated/wrong rank changes the checksum: passing requires all WORLD
  distinct shards computed and reduced correctly.

Proof 2 — data-parallel gradient parity:
  Identical fp64 MLP on every rank (formula-init). Rank r backprops its own data shard;
  grads averaged via all_reduce. Rank 0 recomputes the gradient over the FULL global
  batch single-process and asserts max|distributed - reference| < 1e-9.

Local pre-verification (no GPU, no distributed): python3 distributed_correctness_probe.py --local
Emit checksum only:                              python3 distributed_correctness_probe.py --emit-expected
"""

import argparse
import os
import sys
import time

import torch

# --- fixed problem shape (checksum below is pre-registered for exactly these values)
WORLD = 16
N, KS, M = 4096, 1024, 4096   # per-rank shards: A_r (N x KS), B_r (KS x M)
BATCH_PER_RANK, D_IN, D_H, D_OUT = 512, 64, 128, 8

# Pre-registered on 2026-07-22 via `--emit-expected` on a MacBook (CPU, torch 2.13.0,
# no RNG involved anywhere). The run's all-reduced matmul must reproduce this exactly.
EXPECTED_CHECKSUM = 3093.0


def shard(rank: int, rows: int, cols: int, c1: int, c2: int, c3: int, device) -> torch.Tensor:
    """Integer-valued fp64 matrix in [-8, 8] from a closed-form formula (no RNG)."""
    i = torch.arange(rows, dtype=torch.float64, device=device).unsqueeze(1)
    j = torch.arange(cols, dtype=torch.float64, device=device).unsqueeze(0)
    return torch.remainder(i * c1 + j * c2 + rank * c3, 17.0) - 8.0


def a_shard(rank, device):
    return shard(rank, N, KS, 1103, 367, 97, device)


def b_shard(rank, device):
    return shard(rank, KS, M, 613, 241, 53, device)


def checksum_analytic(device="cpu") -> float:
    """sum(sum_r A_r @ B_r) via the identity sum(A@B) = colsum(A) . rowsum(B).

    Never materializes any product matrix — an independent computation path from
    the distributed matmul it validates.
    """
    total = 0.0
    for r in range(WORLD):
        total += torch.dot(a_shard(r, device).sum(dim=0), b_shard(r, device).sum(dim=1)).item()
    return total


def build_model(device) -> torch.nn.Sequential:
    model = torch.nn.Sequential(
        torch.nn.Linear(D_IN, D_H), torch.nn.Tanh(), torch.nn.Linear(D_H, D_OUT)
    ).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        for p_idx, p in enumerate(model.parameters()):
            flat = torch.arange(p.numel(), dtype=torch.float64, device=device)
            p.copy_(((torch.remainder(flat * 31 + p_idx * 7, 23.0) - 11.0) / 23.0).reshape(p.shape))
    return model


def data_shard(rank, device):
    x = shard(rank, BATCH_PER_RANK, D_IN, 811, 149, 41, device) / 8.0
    y = shard(rank, BATCH_PER_RANK, D_OUT, 419, 233, 29, device) / 8.0
    return x, y


def grads_for(model, x, y):
    model.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x), y).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def run_local(check_expected: bool) -> float:
    """Single-process CPU verification of both proofs (pre-flight before GPU spend)."""
    c = None
    for r in range(WORLD):
        part = a_shard(r, "cpu") @ b_shard(r, "cpu")
        c = part if c is None else c + part
    analytic = checksum_analytic()
    assert c.sum().item() == analytic, "matmul checksum != analytic identity"
    if check_expected and EXPECTED_CHECKSUM is not None:
        assert analytic == EXPECTED_CHECKSUM, f"{analytic} != pre-registered {EXPECTED_CHECKSUM}"

    model = build_model("cpu")
    per_rank = [grads_for(model, *data_shard(r, "cpu")) for r in range(WORLD)]
    avg = [torch.stack(gs).mean(dim=0) for gs in zip(*per_rank)]
    x_all = torch.cat([data_shard(r, "cpu")[0] for r in range(WORLD)])
    y_all = torch.cat([data_shard(r, "cpu")[1] for r in range(WORLD)])
    ref = grads_for(model, x_all, y_all)
    diff = max((a - b).abs().max().item() for a, b in zip(avg, ref))
    assert diff < 1e-9, f"gradient parity diff {diff}"
    print(f"LOCAL_VERIFY_OK checksum={analytic:.0f} grad_diff={diff:.3e}")
    return analytic


def run_distributed():
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == WORLD, f"probe pre-registered for WORLD={WORLD}, got {world}"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dev = torch.device("cuda")

    # --- Proof 1: exact sharded matmul vs pre-registered checksum
    t0 = time.time()
    c = a_shard(rank, dev) @ b_shard(rank, dev)          # this rank's REAL work
    dist.all_reduce(c)                                    # sum of 16 distinct partials
    torch.cuda.synchronize()
    if rank == 0:
        got = c.sum().item()
        assert EXPECTED_CHECKSUM is not None, "run --emit-expected and pin EXPECTED_CHECKSUM first"
        assert got == EXPECTED_CHECKSUM, f"checksum {got} != pre-registered {EXPECTED_CHECKSUM}"
        ref = None                                        # bit-exact full-matrix reference
        for r in range(WORLD):
            part = a_shard(r, dev) @ b_shard(r, dev)
            ref = part if ref is None else ref + part
        assert torch.equal(c, ref), "distributed C differs from single-GPU reference"
        print(f"PROOF1_MATMUL_EXACT_OK checksum={got:.0f} (pre-registered match, "
              f"bit-exact vs reference, {time.time()-t0:.1f}s)", flush=True)

    # --- Proof 2: data-parallel gradient parity
    model = build_model(dev)
    grads = grads_for(model, *data_shard(rank, dev))
    for g in grads:
        dist.all_reduce(g)
        g /= world
    if rank == 0:
        x_all = torch.cat([data_shard(r, dev)[0] for r in range(WORLD)])
        y_all = torch.cat([data_shard(r, dev)[1] for r in range(WORLD)])
        ref = grads_for(model, x_all, y_all)
        diff = max((a - b).abs().max().item() for a, b in zip(grads, ref))
        assert diff < 1e-9, f"gradient parity diff {diff} exceeds 1e-9"
        print(f"PROOF2_GRAD_PARITY_OK max_abs_diff={diff:.3e} over "
              f"{sum(g.numel() for g in ref)} params, global batch {WORLD*BATCH_PER_RANK}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("DISTRIBUTED_CORRECTNESS_OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true", help="single-process CPU verification")
    p.add_argument("--emit-expected", action="store_true", help="print analytic checksum and exit")
    args = p.parse_args()
    if args.emit_expected:
        print(f"{checksum_analytic():.0f}")
        sys.exit(0)
    if args.local:
        run_local(check_expected=True)
    else:
        run_distributed()
