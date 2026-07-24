# Run multi-node training

Goal: train across multiple 8×H100 nodes with torchrun — no custom Docker, no cluster setup.

!!! note "Status: Public Preview (recently)"
    Multi-node reached Public Preview on 2026-07-17 — recent enough that some published docs may
    still say Private Preview. Shapes: **multiples of `GPU_8xH100` only**. Field guide reports
    max 16 nodes / 128 GPUs, sweet spot 3–8 nodes, AWS-only (reported, not lab-verified).

✅ **Verified end-to-end 2026-07-22** (e2-demo-field-eng): 2 nodes / 16×H100 scheduled on-demand in
~40s (run 505819227973807), and a pre-registered distributed-correctness proof passed bit-exact
(run 723000000990125). Receipts: `experiments/foundation-models/NOTES.md`.

## The recipe

`num_accelerators: 16` on `accelerator_type: GPU_8xH100` = 2 nodes. That's the entire multi-node API:

```yaml
experiment_name: my-distributed-training
environment:
  version: "4"
  dependencies: []          # required with version; torch is preinstalled
compute:
  num_accelerators: 16      # 2 × 8xH100 nodes
  accelerator_type: GPU_8xH100
code_source:
  type: snapshot
  snapshot: {root_path: .., include_paths: [src/]}
command: >
  torchrun --nnodes=$NUM_NODES --nproc-per-node=$LOCAL_WORLD_SIZE
  --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT
  $CODE_SOURCE_PATH/src/train.py
max_retries: 0
timeout_minutes: 60
```

Working template: `workloads/multinode-probe.example.yaml`.

## What the platform injects (so torchrun just works)

✅ Verified on the snapshot path, run 505819227973807:

| Variable | Meaning |
|---|---|
| `NUM_NODES`, `NODE_RANK` / `POD_RANK` | topology + this node's rank |
| `LOCAL_WORLD_SIZE`, `WORLD_SIZE` | GPUs per node, total GPUs |
| `MASTER_ADDR`, `MASTER_PORT` | rendezvous |
| `NCCL_DEBUG=INFO`, `NCCL_IB_TIMEOUT=22`, `NCCL_CUMEM_ENABLE=0`, `AWS_OFI_NCCL_VERSION=v1.15.0` | pre-tuned NCCL/EFA |

## What the fabric delivers

Measured on 2×8×H100 (256 MB all_reduce, 16 ranks): 1.4 ms/iter, algbw 191 GB/s, busbw ~359 GB/s —
near the p5's 400 GB/s EFA line rate, over GPUDirect RDMA (`efa-direct`, 32 NICs/node) inter-node
and NVLink/NVLS intra-node.

!!! note "Smoke-grade number"
    One message size × 10 iterations is a health check, not a benchmark. Before quoting bandwidth
    in a deck, run nccl-tests across sizes.

## Sharp edges

- **The CLI streams node 0 only.** Other nodes: `air logs <run-id> --node N`. All nodes report the
  same hostname — distinguish nodes by `NODE_RANK`, never by hostname.
- **MLflow: all nodes share one run — log metrics from rank 0 only**
  ([Track metrics](track-metrics-with-mlflow.md)).
- Notebook `@distributed(gpus=8)` is **single-node only**; the CLI is the only multi-node path.

## Prove your setup before training on it

Correctness by construction beats spot checks. `workloads/multinode-correctness.example.yaml` runs
two assertion-gated proofs (exact 16-shard matmul against a pre-registered checksum; distributed
gradient parity vs. a single-process reference at fp64) and prints sentinels only if they pass —
55 seconds of cluster time to know your all-reduce is actually correct.
