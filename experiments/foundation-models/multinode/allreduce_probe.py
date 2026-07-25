"""Multi-node NCCL probe: init, per-rank identity, all-reduce correctness + bandwidth.

Answers open-q #2 (does multi-node schedule; latency) and #3 (env plumbing on the
snapshot path) — launched via torchrun from probe_multinode.sh.
"""

import os
import signal
import socket
import time

import torch
import torch.distributed as dist


def log_receipt(params: dict, metrics: dict):
    """Rank-0 receipt via MLflow params/metrics — the durable channel on workspaces
    where run logs never land (docs/06-uat-suite.md). alarm() so a blocked tracking
    call fails the receipt, not the run."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        print("no MLFLOW_RUN_ID; skipping receipt", flush=True)
        return
    signal.alarm(120)
    try:
        # client API, not start_run(): resuming transitions the launcher-owned run's status
        # and a second resume failed silently on the job plane (gpu-burn run 1021313559310836)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        for k, v in params.items():
            client.log_param(run_id, k, v)
        for k, v in metrics.items():
            client.log_metric(run_id, k, v)
        print(f"receipt logged to MLflow run {run_id}", flush=True)
    except Exception as e:
        print(f"receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


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
    busbw = algbw * 2 * (world - 1) / world
    if rank == 0:
        print(f"all_reduce 256MB x{iters}: {dt*1000:.1f} ms/iter, algbw {algbw:.1f} GB/s, "
              f"busbw ~{busbw:.1f} GB/s", flush=True)

    # every rank checks in so the receipt proves all ranks ran, not just rank 0
    ranks = [None] * world
    dist.all_gather_object(ranks, {
        "rank": rank, "node_rank": os.environ.get("NODE_RANK", "0"),
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(dev),
    })

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        nodes = sorted({r["node_rank"] for r in ranks})
        log_receipt(
            params={
                "probe_sentinel": "MULTINODE_PROBE_OK",  # unreachable unless all asserts passed
                "world_size": world,
                "nodes_seen": ",".join(nodes),
                "gpu_name": ranks[0]["gpu"],
                "torch_version": torch.__version__,
                "nccl_version": ".".join(map(str, torch.cuda.nccl.version())),
            },
            metrics={"allreduce_256mb_ms": dt * 1000, "algbw_gbps": algbw, "busbw_gbps": busbw},
        )
        print("MULTINODE_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
