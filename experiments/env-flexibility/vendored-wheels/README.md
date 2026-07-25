# Vendored dependencies UAT — no-PyPI workspaces (W-vendor)

The hardened/customer workspaces block PyPI **by design** (no `environment.dependencies`).
This family proves the two supported dependency-delivery paths, GA surface only (no Docker):

| Variant | Delivery | Workload | Status gate |
|---|---|---|---|
| A | **code snapshot** (workspace filesystem): vendor dir rides `include_paths`, `PYTHONPATH` at run | `workloads/vendored-wheels-snapshot.example.yaml` | none — works today |
| B | **UC Volume**: vendor dir staged once in a volume, `PYTHONPATH=/Volumes/...` | `workloads/vendored-wheels-ucvolume.example.yaml` | catalog bucket fix (docs/06 ask #1) |

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
