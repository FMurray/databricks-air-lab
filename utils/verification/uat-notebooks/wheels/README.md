# Vendored wheels: air CLI for the notebook check

Complete offline wheel set for `databricks-air` (CLI) so the `checks/air-cli-from-notebook`
UAT check can `%pip install` it on serverless where PyPI is unreachable by design:

```
%pip install --no-index --find-links /Workspace/Shared/databricks-air-lab/uat/wheels databricks-air
```

The wheels are **not committed** (`*.whl` is gitignored — ~12 MB of binaries bloated the
repo and every air-CLI snapshot tar). The live set lives only in the workspace at
`/Shared/databricks-air-lab/uat/wheels` (the suite's workspace home — this source dir
`utils/verification/uat-notebooks/` is not itself mirrored). To rebuild it, re-vendor
below into this dir, then upload:

```
databricks workspace import-dir . /Shared/databricks-air-lab/uat/wheels --overwrite -p <profile>
```

Target: **linux x86_64, CPython 3.12** (serverless env v5). ~12 MB, 27 wheels,
`databricks-air 1.0.0` (NB: 1.0.0 released ~2026-07-30; earlier repo findings are scoped
to v0.1.x — see docs/06).

Re-vendor (from a machine with access to the internal proxy):

```
cd utils/verification/uat-notebooks/wheels
# antlr4-python3-runtime ships source-only; build its universal wheel first
python3 -m pip wheel "antlr4-python3-runtime==4.9.3" --no-deps -w . \
    --index-url https://pypi-proxy.cloud.databricks.com/simple
python3 -m pip download databricks-air --dest . --find-links . \
    --index-url https://pypi-proxy.cloud.databricks.com/simple \
    --platform manylinux2014_x86_64 --python-version 312 --only-binary=:all:
# completeness check (offline resolve for the target platform):
python3 -m pip install databricks-air --no-index --find-links . --dry-run \
    --target /tmp/air-check --platform manylinux2014_x86_64 --python-version 312 --only-binary=:all:
```
