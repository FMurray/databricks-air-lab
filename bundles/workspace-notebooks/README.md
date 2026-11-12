# Workspace notebooks bundle

Copies the repo's workspace-side notebooks (today: `utils/billing/`, including the reserved GPU
attribution SQL and PySpark notebooks) into your own workspace folder, with one command:

```bash
bundles/workspace-notebooks/manage.sh deploy -p <profile>
```

They land in `/Workspace/Users/<you>/databricks-air-lab/`, under the same paths as in the repo:

```text
/Workspace/Users/<you>/databricks-air-lab/utils/billing/
  reserved_gpu_attribution           SQL notebook
  reserved_gpu_attribution_pyspark   Python notebook
  queries.py, air_usage.sql, ...     workspace files
```

Open a notebook, attach a SQL warehouse or serverless compute, and Run all. Running the
attribution notebooks needs SELECT on `system.access.audit`, `system.billing.usage` and
`system.lakeflow.*`.

Works without `/Workspace/Shared` and without Git folders. Needs Databricks CLI 1.20 or newer
(it uses the direct deployment engine, so nothing is downloaded at deploy time).

| Command | Does |
|---|---|
| `manage.sh deploy -p <profile>` | copies or updates the notebooks |
| `manage.sh validate -p <profile>` | shows the resolved paths without changing anything |
| `manage.sh remove -p <profile>` | deletes the folder and the deploy state (asks first; `--yes` skips the prompt) |

Plain `databricks bundle deploy -p <profile>` from this directory does the same as `manage.sh deploy`.

## Things to know

- **Pass `-p`.** Without it the CLI picks a profile from `DATABRICKS_CONFIG_PROFILE`, then
  `default_profile` in `~/.databrickscfg`, then `[DEFAULT]`, which may not be the workspace you
  mean.
- **Deploys overwrite.** Sync is one way, from your checkout to the workspace. Edits you make to
  the deployed copies (such as filling in `v_params`) are replaced on the next deploy, and files you
  delete from `utils/billing/` are deleted remotely. Clone a notebook in the workspace (⋮ > Clone)
  before customising it.
- **Private files stay private.** Sync honours `.gitignore`, so `docs/private/` (including the
  pre-filled attribution notebook) is never uploaded. Import that one by hand if you need it:
  `databricks workspace import /Workspace/Users/<you>/reserved_gpu_attribution_filled --file docs/private/reserved_gpu_attribution_filled.sql --format SOURCE --language SQL -p <profile>`.
- **`bundle destroy` alone leaves the notebooks.** It deletes only the deploy state under
  `~/.bundle/air-lab-notebooks/`, because the notebooks live outside it. `manage.sh remove` deletes both.

## Adding notebooks

Add a folder to `sync.paths` in `databricks.yml` (paths are relative to this directory, for
example `../../experiments/classic-ml`). Source-format notebooks (first line
`# Databricks notebook source` or `-- Databricks notebook source`) become notebooks, and
everything else becomes a workspace file.

`utils/verification/uat-notebooks/` isn't included yet: several of those notebooks hard-code
`/Workspace/Shared/databricks-air-lab`, which doesn't exist in every workspace.

## Tested offline

`tests/test_deploy_offline.py` runs a real `databricks bundle deploy`, plus `manage.sh deploy` and
`remove`, against an in-memory fake of the workspace API, then checks what landed where. It never
reads your `~/.databrickscfg`.

```bash
python3 bundles/workspace-notebooks/tests/test_deploy_offline.py                     # CLI on PATH
DATABRICKS_CLI=/path/to/databricks-1.20 python3 bundles/workspace-notebooks/tests/test_deploy_offline.py
```

It passes with CLI 1.20.0 and 1.26. It hasn't been deployed to a real workspace yet.
