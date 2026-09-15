"""Minimal two-node DDP training proof for the native ai_runtime_task path."""

from __future__ import annotations

import os
import signal
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


SENTINEL = "AI_RUNTIME_MULTINODE_TRAINING_OK"


def log_receipt(run_id: str, params: dict[str, object], metrics: dict[str, float]) -> None:
    """Persist the rank-0 receipt in the launcher-owned MLflow run."""
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        for key, value in params.items():
            client.log_param(run_id, key, value)
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
    finally:
        signal.alarm(0)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    node_rank = int(os.environ["NODE_RANK"])
    expected_nodes = int(os.environ["NUM_NODES"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260914)

    model = DistributedDataParallel(
        torch.nn.Linear(16, 1, bias=False).to(device),
        device_ids=[local_rank],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Distinct data per rank proves DDP must synchronize gradients for parameters
    # to remain identical after the optimizer step.
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    features = torch.randn(32, 16, generator=generator, device=device)
    targets = features.sum(dim=1, keepdim=True) * 0.25
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features), targets)
    loss.backward()
    optimizer.step()

    checksum = next(model.parameters()).detach().double().sum()
    checksum_min = checksum.clone()
    checksum_max = checksum.clone()
    dist.all_reduce(checksum_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(checksum_max, op=dist.ReduceOp.MAX)
    assert checksum_min.item() == checksum_max.item(), (
        f"DDP parameters diverged: min={checksum_min.item()} max={checksum_max.item()}"
    )

    rank_receipts: list[dict[str, object] | None] = [None] * world_size
    dist.all_gather_object(
        rank_receipts,
        {
            "rank": rank,
            "node_rank": node_rank,
            "host": socket.gethostname(),
            "local_rank": local_rank,
            "loss": loss.item(),
        },
    )
    receipts = [receipt for receipt in rank_receipts if receipt is not None]
    nodes_seen = sorted({int(receipt["node_rank"]) for receipt in receipts})
    assert expected_nodes >= 2, f"not multi-node: NUM_NODES={expected_nodes}"
    assert len(nodes_seen) == expected_nodes, (
        f"expected {expected_nodes} node ranks, saw {nodes_seen}"
    )
    assert world_size == int(os.environ["WORLD_SIZE"])

    print(
        "RANK_RECEIPT "
        f"rank={rank} node_rank={node_rank} local_rank={local_rank} "
        f"world_size={world_size} loss={loss.item():.8f}",
        flush=True,
    )

    if rank == 0:
        run_id = os.environ["MLFLOW_RUN_ID"]
        log_receipt(
            run_id,
            {
                "probe_sentinel": SENTINEL,
                "task_type_under_test": "ai_runtime_task",
                "world_size": world_size,
                "num_nodes": expected_nodes,
                "nodes_seen": ",".join(map(str, nodes_seen)),
                "accelerator_type": "GPU_1xA10",
                "torch_version": torch.__version__,
            },
            {
                "rank0_loss": loss.item(),
                "parameter_checksum": checksum.item(),
            },
        )
        print(
            f"{SENTINEL} world_size={world_size} nodes_seen={nodes_seen} "
            f"parameter_checksum={checksum.item():.12f}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
