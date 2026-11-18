#!/usr/bin/env bash
# Deploy the AIR GPU usage dashboard to /Workspace/Users/<you>/ and publish it.
#
#   bundles/air-usage-dashboard/manage.sh deploy -p <profile> --var warehouse_id=<id>    # create / update
#   bundles/air-usage-dashboard/manage.sh validate -p <profile> --var warehouse_id=<id>  # show what would deploy
#   bundles/air-usage-dashboard/manage.sh remove -p <profile> --var warehouse_id=<id>    # delete it and the state
#
# Needs Databricks CLI >= 1.20. Without -p the CLI uses DATABRICKS_CONFIG_PROFILE, then the
# `default_profile` in ~/.databrickscfg, then [DEFAULT]: pass -p to be sure which workspace you hit.
# BUNDLE_VAR_warehouse_id=<id> in the environment works instead of --var.
set -euo pipefail
DBX="${DATABRICKS_CLI:-databricks}"  # which CLI binary; tests point it at a specific version
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

usage() { sed -n '2,10p' "$HERE/manage.sh" | sed 's/^# \{0,1\}//'; }

url() {  # the deployed dashboard's URL, from the bundle summary (blank if the CLI doesn't report one)
  "$DBX" bundle summary -o json "$@" 2>/dev/null |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["resources"]["dashboards"]["air_gpu_usage"].get("url", ""))' ||
    true
}

action="${1:-}"
[ $# -gt 0 ] && shift
case "$action" in
  validate)
    exec "$DBX" bundle validate "$@"
    ;;
  deploy)
    "$DBX" bundle deploy "$@"
    echo "Dashboard deployed: $(url "$@")"
    ;;
  remove)
    yes=0
    args=()
    for a in "$@"; do
      if [ "$a" = "--yes" ]; then yes=1; else args+=("$a"); fi
    done
    if [ "$yes" != 1 ]; then
      printf 'Delete the AIR GPU usage dashboard and its deploy state? [y/N] '
      read -r answer
      [ "$answer" = y ] || [ "$answer" = Y ] || { echo "Nothing deleted."; exit 1; }
    fi
    # The dashboard is a bundle resource, so destroy removes it along with the state.
    "$DBX" bundle destroy --auto-approve "${args[@]+"${args[@]}"}"
    echo "Deleted the dashboard"
    ;;
  ""|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
