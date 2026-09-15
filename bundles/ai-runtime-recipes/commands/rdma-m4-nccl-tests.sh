#!/usr/bin/env bash
set -euo pipefail

: "${CODE_SOURCE_PATH:?AI Runtime did not set CODE_SOURCE_PATH}"

exec bash "$CODE_SOURCE_PATH/rdma-stress/build_and_run_nccl_tests.sh"
