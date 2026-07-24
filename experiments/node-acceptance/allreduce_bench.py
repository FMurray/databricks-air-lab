"""All-reduce bandwidth bench (UAT A2) — pip-only stand-in for nccl-tests.

Launch under torchrun (see workloads/nccl-allreduce.example.yaml). Sweeps message sizes,
reports algorithm + bus bandwidth per size (busbw = algbw * 2(n-1)/n, matching nccl-tests
so numbers are comparable to published NVLink/fabric line rates).

Env knobs: MIN_MB (8), MAX_MB (1024), ITERS (20), MIN_BUSBW_GBPS (optional pass bar —
unset means record-only, no bar; set it once expected fabric numbers are known).
"""
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Correctness first: all-reduce of ones must equal world size everywhere.
    t = torch.ones(1024, device="cuda")
    dist.all_reduce(t)
    assert torch.all(t == world), f"rank {rank}: all-reduce sum wrong (got {t[0].item()}, want {world})"

    min_mb = int(os.environ.get("MIN_MB", "8"))
    max_mb = int(os.environ.get("MAX_MB", "1024"))
    iters = int(os.environ.get("ITERS", "20"))
    results = []
    size_mb = min_mb
    while size_mb <= max_mb:
        n_elem = size_mb * 1024 * 1024 // 4
        buf = torch.randn(n_elem, device="cuda")
        for _ in range(5):  # warmup
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        per_op = (time.time() - t0) / iters
        algbw = size_mb / 1024 / per_op                    # GB/s
        busbw = algbw * 2 * (world - 1) / world
        results.append((size_mb, per_op, algbw, busbw))
        if rank == 0:
            print(f"RESULT size_mb={size_mb} time_ms={per_op*1e3:.2f} "
                  f"algbw_gbps={algbw:.1f} busbw_gbps={busbw:.1f}")
        size_mb *= 4

    if rank == 0:
        peak = max(r[3] for r in results)
        nodes = int(os.environ.get("NUM_NODES", "1"))
        print(f"RESULT allreduce=DONE world={world} nodes={nodes} peak_busbw_gbps={peak:.1f}")
        bar = os.environ.get("MIN_BUSBW_GBPS")
        assert not bar or peak >= float(bar), f"peak busbw {peak:.1f} < required {bar} GB/s"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
