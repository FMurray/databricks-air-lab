#!/usr/bin/env bash
set -euo pipefail

echo "TOPOLOGY NUM_NODES=${NUM_NODES:-unset} NODE_RANK=${NODE_RANK:-unset} POD_RANK=${POD_RANK:-unset} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-unset} WORLD_SIZE=${WORLD_SIZE:-unset}"

AI_PY="${AI_ENV_PYTHON:-/opt/databricks-environments/databricks-ai/bin/python}"
NPROC="${LOCAL_WORLD_SIZE:-$(nvidia-smi -L | wc -l)}"

"$AI_PY" -m torch.distributed.run \
  --nnodes="${NUM_NODES}" \
  --node_rank="${NODE_RANK:-${POD_RANK}}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="$NPROC" \
  "$CODE_SOURCE_PATH/train_probe.py"
