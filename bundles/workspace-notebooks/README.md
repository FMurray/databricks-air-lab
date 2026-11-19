# Workspace notebooks + AIR GPU usage dashboard

One bundle, one command. It copies the repo's workspace-side notebooks (today: `utils/billing/`, including the
reserved GPU attribution notebooks and the AIR sessions probe) into your own workspace folder, and creates and
publishes the **AIR GPU usage** dashboard ([DASHBOARD.md](DASHBOARD.md)):

```bash
uv run bundles/workspace-notebooks/deploy.py deploy -p <profile> --var warehouse_id=<sql-warehouse-id>
```

`deploy.py` uses the Databricks SDK: no Databricks CLI needed. `uv run` installs its two dependencies
(`databricks-sdk`, `pyyaml`, declared in the script) on first use, from any directory. Inside the repo's environment
(`uv sync`), `python bundles/workspace-notebooks/deploy.py ...` works too.

Notebooks land in `/Workspace/Users/<you>/databricks-air-lab/`, under the same paths as in the repo; the dashboard
at `/Workspace/Users/<you>/AIR GPU usage.lvdash.json`:

```text
/Workspace/Users/<you>/
  AIR GPU usage.lvdash.json          dashboard (published)
  databricks-air-lab/utils/billing/
    reserved_gpu_attribution         SQL notebook
    reserved_gpu_attribution_pyspark Python notebook
    air_sessions_probe               Python notebook
    queries.py, air_gpu_usage.sql, ...  workspace files
```

Find a warehouse ID with `databricks warehouses list -p <profile>` (or SQL Warehouses > the warehouse > its URL); a
serverless SQL warehouse is the best fit. Without `warehouse_id`, deploy copies the notebooks and skips the dashboard.

| Command | Does |
|---|---|
| `deploy.py deploy -p <profile> --var warehouse_id=<id>` | copies or updates the notebooks; creates or updates the dashboard and publishes it |
| `deploy.py validate -p <profile> --var warehouse_id=<id>` | shows what deploy would do (reads the workspace, writes nothing) |
| `deploy.py remove -p <profile>` | deletes the notebooks folder, trashes the dashboard, and clears old `databricks bundle` deploy state (asks first; `--yes` skips the prompt) |

`BUNDLE_VAR_warehouse_id=<id>` in the environment works instead of `--var`; `destroy` is the same as `remove`.

Open a notebook, attach a SQL warehouse or serverless compute, and Run all. The attribution notebooks and the
dashboard need SELECT on `system.access.audit`, `system.billing.usage` and `system.lakeflow.*`.

## Same config as `databricks bundle`

`deploy.py` reads `databricks.yml` in this folder: the notebooks folder (`workspace.file_path`), what to copy
(`sync.paths`), and the dashboard resource. So with a Databricks CLI (1.20 or newer), plain
`databricks bundle deploy -p <profile> --var warehouse_id=<id>` from this directory deploys the same thing; the CLI
requires the warehouse ID. A dashboard either way deployed is found by its path and updated in place by the other.

The script supports the parts of the bundle format this file uses: `${workspace.current_user.userName}` and
`${var.*}` references, `sync.paths`, and `resources.dashboards`. It keeps no deploy state: it finds the dashboard by
path and works out stale files by listing the synced folders.

## Things to know

- **Pass `-p`.** Without it the SDK uses `DATABRICKS_*` environment variables, then `[DEFAULT]` in
  `~/.databrickscfg`, which may not be the workspace you mean.
- **Deploys overwrite.** Sync is one way, from your checkout to the workspace. Edits you make to the deployed copies
  (such as filling in `v_params`) are replaced on the next deploy, and files you delete from `utils/billing/` are
  deleted remotely. The same goes for the dashboard. Clone a notebook or the dashboard in the workspace (⋮ > Clone)
  before customising it.
- **Private files stay private.** Only files git tracks, or would track (`.gitignore` applies), are copied, so
  `docs/private/` (including the pre-filled attribution notebook) is never uploaded. Import that one by hand if you
  need it: `databricks workspace import /Workspace/Users/<you>/reserved_gpu_attribution_filled --file
  docs/private/reserved_gpu_attribution_filled.sql --format SOURCE --language SQL -p <profile>`.
- **Viewers of the dashboard need system-table access**: it's published without embedded credentials (see
  [DASHBOARD.md](DASHBOARD.md)).
- **`bundle destroy` alone leaves the notebooks.** It deletes only the CLI's deploy state under
  `~/.bundle/air-lab-notebooks/`, because the notebooks live outside it. `deploy.py remove` deletes all of it.

## Adding notebooks

Add a folder to `sync.paths` in `databricks.yml` (paths are relative to this directory, for example
`../../experiments/classic-ml`). Source-format notebooks (first line `# Databricks notebook source` or
`-- Databricks notebook source`) become notebooks, and everything else becomes a workspace file.

`utils/verification/uat-notebooks/` isn't included yet: several of those notebooks hard-code
`/Workspace/Shared/databricks-air-lab`, which doesn't exist in every workspace.

## Changing the dashboard

Edit `utils/billing/air_gpu_usage.sql` (the dataset) or the layout in `build_dashboard.py`, then rebuild and deploy:

```bash
python3 bundles/workspace-notebooks/build_dashboard.py          # rewrites src/air_gpu_usage.lvdash.json
uv run bundles/workspace-notebooks/deploy.py deploy -p <profile> --var warehouse_id=<id>
```

Commit the SQL, the script and the JSON together; the offline test fails if the JSON is stale.

## Tested offline

`tests/test_deploy_offline.py` runs `deploy.py` (validate, deploy twice, remove) and, if a CLI is available, a real
`databricks bundle deploy`, against an in-memory fake of the workspace and Lakeview APIs, then checks what landed
where. It also checks the dashboard JSON is fresh, its parameters match the SQL, and every widget field is a dataset
column. It never reads your `~/.databrickscfg`.

```bash
uv run --with databricks-sdk --with pyyaml python bundles/workspace-notebooks/tests/test_deploy_offline.py
DATABRICKS_CLI=/path/to/databricks-1.20 uv run --with databricks-sdk --with pyyaml python bundles/workspace-notebooks/tests/test_deploy_offline.py
```

It passes with databricks-sdk 0.169 on Python 3.10, and CLI 1.26. `deploy.py` hasn't been run against a real
workspace yet.
