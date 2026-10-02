#!/usr/bin/env bash
set -euo pipefail

: "${CODE_SOURCE_PATH:?AI Runtime did not set CODE_SOURCE_PATH}"

# The launcher installs environments[].spec dependencies into `python3` (install_deps.py targets
# its own interpreter), so check from that interpreter rather than whatever `python` resolves to.
exec python3 "$CODE_SOURCE_PATH/verify_environment.py"
