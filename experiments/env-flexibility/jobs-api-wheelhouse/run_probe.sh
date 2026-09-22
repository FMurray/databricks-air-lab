#!/usr/bin/env bash
set -euo pipefail

: "${CODE_SOURCE_PATH:?AI Runtime did not set CODE_SOURCE_PATH}"

exec python "$CODE_SOURCE_PATH/verify_environment.py"
