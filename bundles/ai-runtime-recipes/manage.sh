#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=forrest_serverless

usage() {
  cat <<'EOF'
Usage:
  ./manage.sh list
  ./manage.sh validate
  ./manage.sh deploy
  ./manage.sh run <recipe> --yes

Recipes:
  rdma-m1-smoke           2 nodes x 8 H100, 60-second all-reduce smoke
  rdma-m1-soak            2 nodes x 8 H100, 10-minute all-reduce soak
  rdma-m2a-fabric-only    4 nodes x 1 active H100, 10-minute all-reduce soak
  rdma-m2b-p2p-ring       4 nodes x 1 active H100, 10-minute P2P ring soak
  rdma-m4-nccl-tests      1 node  x 8 H100, nccl-tests build + benchmark
  rdma-m5-parambench      1 node  x 1 A10, parambench probe
EOF
}

require_cli() {
  local version major minor
  version="$(databricks version | sed -E 's/.*v([0-9]+\.[0-9]+).*/\1/')"
  major="${version%%.*}"
  minor="${version#*.}"
  if [[ ! "$major" =~ ^[0-9]+$ || ! "$minor" =~ ^[0-9]+$ ]] ||
     (( major < 1 || (major == 1 && minor < 6) )); then
    echo "Databricks CLI >= 1.6.0 is required for bundle ai_runtime_task support; found: $(databricks version)" >&2
    exit 1
  fi
}

resource_for_recipe() {
  case "$1" in
    rdma-m1-smoke) echo rdma_m1_smoke ;;
    rdma-m1-soak) echo rdma_m1_soak ;;
    rdma-m2a-fabric-only) echo rdma_m2a_fabric_only ;;
    rdma-m2b-p2p-ring) echo rdma_m2b_p2p_ring ;;
    rdma-m4-nccl-tests) echo rdma_m4_nccl_tests ;;
    rdma-m5-parambench) echo rdma_m5_parambench ;;
    *) echo "Unknown recipe: $1" >&2; usage >&2; exit 2 ;;
  esac
}

action="${1:-}"
case "$action" in
  list)
    usage
    ;;
  validate)
    require_cli
    cd "$BUNDLE_DIR"
    exec databricks bundle validate --strict --target "$TARGET"
    ;;
  deploy)
    require_cli
    cd "$BUNDLE_DIR"
    databricks bundle validate --strict --target "$TARGET"
    exec databricks bundle deploy --target "$TARGET"
    ;;
  run)
    require_cli
    recipe="${2:-}"
    if [[ -z "$recipe" || "${3:-}" != "--yes" ]]; then
      echo "A recipe and --yes are required because this starts billable GPU compute." >&2
      usage >&2
      exit 2
    fi
    resource="$(resource_for_recipe "$recipe")"
    cd "$BUNDLE_DIR"
    exec databricks bundle run "$resource" --target "$TARGET"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
