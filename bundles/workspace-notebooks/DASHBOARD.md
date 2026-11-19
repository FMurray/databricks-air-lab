# AIR GPU usage dashboard

A Lakeview dashboard of AIR GPU-hours over time, by **capacity** (reserved / on-demand), **accelerator**,
**workload shape** (nodes x GPUs per node) and **usage policy ID**. It reads the same system tables as the
attribution notebooks in `utils/billing/`, with no intermediate tables to build. It's deployed and published with
the notebooks by this bundle: see [README.md](README.md) (`deploy.py deploy -p <profile> --var warehouse_id=<id>`).

## What it shows

**Usage** page: total, reserved and on-demand GPU-hours. Then GPU-hours per day by capacity, by accelerator, by
shape, by workload (jobs vs notebooks) and by policy. Then GPU-hours per policy, split by capacity, and a pivot of
policy x accelerator x shape x workload against capacity. Filters at the top: since (a parameter, default 90 days
back), reserved accelerators (a parameter), and capacity, accelerator, shape, workload, policy and workspace.

**Attribution quality** page: how each GPU-hour's policy, capacity and GPU count were found (below).

## How the numbers are made

The dataset is `utils/billing/air_gpu_usage.sql`, a single-statement version of `reserved_gpu_attribution.sql`
(Parts A and N), widened from reserved-only to all AIR usage. Its header has the details. In short:

- **GPU-hours = GPU-hold time x GPUs**, not billed DBUs / rate. Reserved billing rows are per node, with no job,
  notebook or policy, so the reserved bill can't be split by workload. No DBU rate is needed.
- **Jobs**: accelerator type and `accelerator_count` come from the `ai_runtime_task` in jobs audit events, matched to
  each task run as of its start. Hold time is the run's execution time (the `hold_basis` parameter).
- **Notebooks**: AIR sessions from notebook audit events, cut into holds at idle gaps longer than the timeout.
  **The accelerator isn't logged for notebooks**: reserved holds count on the first reserved accelerator, and
  on-demand holds as `unknown (notebook)`. Both assume `nb_gpus` (8) GPUs, so on-demand notebooks on smaller
  accelerators (e.g. A10) are overstated.
- **Capacity**: on-demand if the workload has on-demand AIR billing rows, or runs on an accelerator not in
  `reserved_accelerators`. Otherwise reserved, by elimination. Workloads that ended in the last 24 hours show as
  `pending billing` until billing lands.
- **Policy**: from the on-demand billing rows when there are any. Otherwise from the latest policy-setting jobs audit
  event (jobs), or from the notebook's billed usage on other compute (notebooks). `(none found)` = neither.
- A workload's GPU-hours are spread over the days it ran, by wall-clock overlap. Days are in the warehouse's time
  zone (UTC by default).

## Things to know

- **Viewers need system-table access.** The dashboard is published without embedded credentials, so each viewer
  queries as themselves and needs SELECT on `system.access.audit`, `system.billing.usage` and
  `system.lakeflow.job_task_run_timeline`. Embedding the publisher's credentials would show account-wide billing
  and audit data to anyone it's shared with. To change that, set `embed_credentials` in `databricks.yml`.
- **Every refresh scans the system tables** from `since` minus a lookback (180 days of jobs audit for task specs).
  Keep `since` modest. If refreshes get slow, the next step is to materialize the dataset on a schedule.
- **Other parameters** (`hold_basis`, `nb_idle_timeout_min`, `nb_gpus`, `spec_lookback_days`) aren't on the
  dashboard. Change their defaults in the dataset (draft view > Data), or in `PARAMS` in `build_dashboard.py`
  followed by a redeploy.
- **Deploys overwrite.** Edits made to the dashboard in the workspace are replaced on the next deploy. Clone it in
  the workspace before customising.
- **No workspace filter.** System tables are account-wide (audit: the account's workspaces in the region). Use
  the Workspace filter.

To change the dashboard, or for how it's tested, see [README.md](README.md).
