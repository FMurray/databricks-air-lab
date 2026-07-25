#!/usr/bin/env bash
# Vendor the UAT test dependencies for linux/amd64 AIR nodes (run on any host; uv cross-targets).
# Unpacked --target install → nodes need only PYTHONPATH, no pip. See README.md.
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION="${PY_VERSION:-3.12}"   # match the AIR env's python (v5 → verify via probe if unsure)
PACKAGES=(emoji xxhash)            # pure-python + one compiled ext (platform-tag proof)

rm -rf vendor
uv pip install --target vendor \
  --python-platform x86_64-unknown-linux-gnu \
  --python-version "$PY_VERSION" \
  --only-binary :all: \
  "${PACKAGES[@]}"

du -sh vendor
echo "vendored for linux/amd64 py$PY_VERSION: ${PACKAGES[*]}"
