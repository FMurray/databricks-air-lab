#!/usr/bin/env bash
set -euo pipefail

: "${CODE_SOURCE_PATH:?AI Runtime did not set CODE_SOURCE_PATH}"
export STRESS_SECONDS=600
export BUF_MB=1024
export FABRIC_ONLY=1
export P2P_RING=1

exec python "$CODE_SOURCE_PATH/node-acceptance/nccl_allreduce_ctypes.py"
