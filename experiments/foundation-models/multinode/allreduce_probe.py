"""Multi-node NCCL probe: init, per-rank identity, all-reduce correctness + bandwidth.

Answers open-q #2 (does multi-node schedule; latency) and #3 (env plumbing on the
snapshot path) — launched via torchrun from probe_multinode.sh.
"""

import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.cuda.current_device()
    print(f"[rank {rank}/{world}] host={socket.gethostname()} local_rank={local_rank} "
          f"gpu={torch.cuda.get_device_name(dev)} nccl={torch.cuda.nccl.version()}", flush=True)

    # correctness: sum of ones == world size everywhere
    t = torch.ones(1, device=dev)
    dist.all_reduce(t)
    assert t.item() == world, f"all_reduce wrong: {t.item()} != {world}"

    # bandwidth: 256MB fp32 all-reduce, 10 iters (algbw; ring busbw = algbw * 2(n-1)/n)
    n = 64 * 1024 * 1024
    big = torch.ones(n, device=dev)
    for _ in range(3):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    algbw = n * 4 / dt / 1e9
    if rank == 0:
        print(f"all_reduce 256MB x{iters}: {dt*1000:.1f} ms/iter, algbw {algbw:.1f} GB/s, "
              f"busbw ~{algbw * 2 * (world - 1) / world:.1f} GB/s", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("MULTINODE_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
