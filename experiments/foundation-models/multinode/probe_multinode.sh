#!/usr/bin/env bash
# Runs on EACH node. Dumps the injected env (open-q #3), then torchrun-launches the
# NCCL all-reduce probe across nodes using the documented Docker-path env vars —
# this run verifies the snapshot path injects the same ones.
set -euo pipefail

echo "=== dist-related env on $(hostname):"
env | sort | grep -E 'RANK|WORLD|MASTER|NODE|LOCAL_|NUM_NODES|NCCL|CUDA_VISIBLE' || true
nvidia-smi -L

NPROC="${LOCAL_WORLD_SIZE:-$(nvidia-smi -L | wc -l)}"
torchrun \
  --nnodes="${NUM_NODES:-1}" \
  --node_rank="${NODE_RANK:-${POD_RANK:-0}}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  --nproc_per_node="$NPROC" \
  "$CODE_SOURCE_PATH/experiments/foundation-models/multinode/allreduce_probe.py"
