#!/usr/bin/env bash
# Runs on EACH node. Dumps the injected env (open-q #3), then torchrun-launches the
# NCCL all-reduce — this *is* the UAT allreduce probe (plumbing + headline allreduce-fabric).
# Uses the documented Docker-path env vars; this run verifies the snapshot path injects
# the same ones.
set -euo pipefail

echo "=== dist-related env on $(hostname):"
env | sort | grep -E 'RANK|WORLD|MASTER|NODE|LOCAL_|NUM_NODES|NCCL|CUDA_VISIBLE' || true
nvidia-smi -L

# torch/torchrun live in the baked-in AI env, NOT on the bare v5 interpreter's PATH
# (docs/06-uat-suite.md; rdma-stress/NOTES.md — survey 96419244890099). Launch via the
# AI-env python + `-m torch.distributed.run` (== torchrun) so we don't depend on PATH.
AI_PY="${AI_ENV_PYTHON:-/opt/databricks-environments/databricks-ai/bin/python}"
echo "=== launching torch.distributed.run via: $AI_PY"

NPROC="${LOCAL_WORLD_SIZE:-$(nvidia-smi -L | wc -l)}"
"$AI_PY" -m torch.distributed.run \
  --nnodes="${NUM_NODES:-1}" \
  --node_rank="${NODE_RANK:-${POD_RANK:-0}}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  --nproc_per_node="$NPROC" \
  "$CODE_SOURCE_PATH/experiments/foundation-models/multinode/allreduce_probe.py"
