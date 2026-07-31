# Vendored dependencies UAT — no-PyPI workspaces (W-vendor)

> **Customer-facing "for dummies" recipe: [docs/cookbook/install-packages-from-a-volume.md](../../../docs/cookbook/install-packages-from-a-volume.md)**
> — 4 copy-paste steps, troubleshooting table. This README is the UAT engineering detail behind it.

The hardened/customer workspaces block PyPI **by design** (no `environment.dependencies`).

**Check the documented path FIRST — workspace default package repositories (GA):** an admin
points the workspace at an internal mirror (Artifactory etc.) via secret scope
`databricks-package-management` (keys `pip-index-url`, `pip-extra-index-urls`, `pip-cert`;
auth embedded in URL; UI: Settings → Compute → Default Package Repositories). Serverless
notebooks and jobs inherit it automatically — `environment.dependencies` then just works.
docs: /aws/en/admin/workspace-settings/default-python-packages. ⚠️ Unverified whether the
AIR Gen-AI env build inherits it (needs a workspace admin to set it + rerun
`workloads/probes/pip-probe.yaml`; if the error changes from PyPI DNS to the custom index,
it's inherited). For the customer: this is the production answer.

This family proves the two **zero-admin fallback** paths (also the fully air-gapped story),
GA surface only (no Docker):

| Variant | Delivery | Workload | Status gate |
|---|---|---|---|
| A | **code snapshot** (workspace filesystem): vendor dir rides `include_paths`, `PYTHONPATH` at run | `workloads/vendored-wheels-snapshot.example.yaml` | none — works today; small deps only |
| B | **UC Volume**: vendor dir staged once in a volume, `PYTHONPATH=/Volumes/...` | `workloads/vendored-wheels-ucvolume.example.yaml` | catalog bucket fix (docs/06 ask #1) |
| C | **env-build from workspace wheels**: `environment.dependencies` accepts direct paths (`- /Workspace/Shared/.../pkg.whl`), `-r` files, and inline `--index-url` (per `air -h config.environment`) | untested — build after A/B verdicts | workspace-file size cap (~500MB/file) rules out torch-scale wheels |

## How it works

Dependencies are vendored **unpacked** (not wheels) with uv's cross-targeting, so the AIR
node needs no pip at all — just `PYTHONPATH`:

```bash
# host-side (Apple Silicon OK — uv cross-targets linux/amd64):
./experiments/env-flexibility/vendored-wheels/vendor_deps.sh
# variant A: the dir ships automatically via the workload's include_paths
# variant B: stage it once:  databricks fs cp -r experiments/env-flexibility/vendored-wheels/vendor \
#              dbfs:/Volumes/<catalog>/<schema>/<volume>/vendor -p <profile>
```

`verify_deps.py` imports each vendored package (incl. one compiled ext, `xxhash` — proves
platform/py-version tags are right), prints `RESULT` lines, and logs MLflow params
(receipt channel) with `mlflow.end_run()`.

## Sharp edges

- `vendor/` is **committed** (5.3MB, small test deps): the CLI's snapshot tar RESPECTS
  .gitignore (verified run 619243740765768 — an ignored vendor/ silently vanishes from the
  snapshot). Committed = handoff-ready. Node python is 3.12 (verified); re-run
  `vendor_deps.sh` only to change packages or python version.
- Watch snapshot size with heavy deps (torch-adjacent = hundreds of MB → use variant B).
- Customer production analog: internal mirror (Artifactory) replaces the vendor step;
  the PYTHONPATH/volume mechanics are identical.
