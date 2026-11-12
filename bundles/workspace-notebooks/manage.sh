#!/usr/bin/env bash
# Copy this repo's workspace notebooks into /Workspace/Users/<you>/databricks-air-lab/.
#
#   bundles/workspace-notebooks/manage.sh deploy -p <profile>    # copy / update (run from anywhere)
#   bundles/workspace-notebooks/manage.sh validate -p <profile>  # show what would go where
#   bundles/workspace-notebooks/manage.sh remove -p <profile>    # delete the folder and deploy state
#
# Needs Databricks CLI >= 1.20. Without -p the CLI uses DATABRICKS_CONFIG_PROFILE, then the
# `default_profile` in ~/.databrickscfg, then [DEFAULT]: pass -p to be sure which workspace you hit.
set -euo pipefail
DBX="${DATABRICKS_CLI:-databricks}"  # which CLI binary; tests point it at a specific version
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

usage() { sed -n '2,9p' "$HERE/manage.sh" | sed 's/^# \{0,1\}//'; }

folder() {  # the deployed folder, from the resolved bundle config
  "$DBX" bundle validate -o json "$@" 2>/dev/null |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["workspace"]["file_path"])'
}

action="${1:-}"
[ $# -gt 0 ] && shift
case "$action" in
  validate)
    exec "$DBX" bundle validate "$@"
    ;;
  deploy)
    "$DBX" bundle deploy "$@"
    echo "Notebooks are in $(folder "$@")"
    ;;
  remove)
    yes=0
    args=()
    for a in "$@"; do
      if [ "$a" = "--yes" ]; then yes=1; else args+=("$a"); fi
    done
    dir="$(folder "${args[@]+"${args[@]}"}")"
    if [ "$yes" != 1 ]; then
      printf 'Delete %s and everything in it, plus the deploy state? [y/N] ' "$dir"
      read -r answer
      [ "$answer" = y ] || [ "$answer" = Y ] || { echo "Nothing deleted."; exit 1; }
    fi
    # `bundle destroy` removes only the deploy state (~/.bundle/...): the notebooks live outside it.
    "$DBX" bundle destroy --auto-approve "${args[@]+"${args[@]}"}"
    "$DBX" workspace delete --recursive "$dir" "${args[@]+"${args[@]}"}"
    echo "Deleted $dir"
    ;;
  ""|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
