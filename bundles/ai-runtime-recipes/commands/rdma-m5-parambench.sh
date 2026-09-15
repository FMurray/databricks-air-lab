#!/usr/bin/env bash
set -euo pipefail

: "${CODE_SOURCE_PATH:?AI Runtime did not set CODE_SOURCE_PATH}"
AI_PY=/opt/databricks-environments/databricks-ai/bin/python

exec "$AI_PY" "$CODE_SOURCE_PATH/experiments/rdma-stress/parambench_probe.py"
