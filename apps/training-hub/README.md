# Training Hub

A web app (Streamlit → HTML on `:8501`) for teams sharing a reserved serverless-GPU (AIR) pool.
Users land on a tab matching who they are and act from there:

| Who you are | What you want | Go to |
|---|---|---|
| Platform lead / org admin | See utilization, quotas, spend by team, active runs | **Fleet** tab |
| Team member | Try a training run: build a valid AIR workload YAML and submit it | **Submit a workload** tab |
| Anyone new to AIR | Learn what a workload looks like before running one | **Submit** tab — pick a template, read the generated YAML |

This README follows [Diátaxis](https://diataxis.fr): jump to the section matching your intent —
[serve it now](#serve-the-app-agents-start-here) · [tutorial](#tutorial-first-run) ·
[how-to guides](#how-to-guides) · [reference](#reference) · [explanation](#explanation).

## Serve the app (agents: start here)

One command, copy-paste runnable with **zero configuration** — the app falls back to
`config/teams.example.yaml` and renders demo data when no real config exists (verified 2026-07-22):

```bash
cd apps/training-hub
uv run --with-requirements requirements.txt -- streamlit run app.py --server.headless true
```

No `uv`? Any Python ≥3.10 env works: `pip install -r requirements.txt && streamlit run app.py --server.headless true`.

**Done when:** `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501` prints `200`,
and the page shows "Training Hub" with two tabs, **Fleet** and **Submit a workload**
(the browser tab title is set client-side, so check the rendered page, not curl's `<title>`).
Without workspace credentials the Fleet tab's
"GPU spend by team" and "Active runs" panels show in-app errors instead of data — that is
expected, not a failure. To light them up, see
[Point the Fleet tab at live data](#point-the-fleet-tab-at-live-data).

## Tutorial: first run

Goal: serve the app and generate your first AIR workload YAML. No workspace access needed.

1. Serve the app as above and open http://localhost:8501.
2. Open the **Fleet** tab: the four metrics at the top (reserved nodes, GPUs, allocated,
   unallocated) come from `config/teams.example.yaml` — this is the admin's view of the pool.
3. Open **Submit a workload**: pick the `single-node-finetune` template. The right pane shows
   the complete AIR workload YAML updating live as you edit fields.
4. Change **Num accelerators** and watch `compute.num_accelerators` change in the YAML —
   this schema is what `air run -f <file>.yaml` accepts.
5. Click **Download YAML**. You now have a valid workload file; submitting it is a
   [how-to](#submit-a-workload-to-air) once you have workspace access.

## How-to guides

### Configure real teams and quotas

```bash
cp config/teams.example.yaml config/teams.yaml   # gitignored; fill in real teams/emails/quotas
```

Or point `HUB_TEAMS_CONFIG` at a config elsewhere. Members are matched to billing principals
case-insensitively; anyone absent from the config shows up as `unmapped` spend.

### Point the Fleet tab at live data

```bash
export DATABRICKS_CONFIG_PROFILE=<profile-for-the-target-workspace>
export HUB_WAREHOUSE_HTTP_PATH="/sql/1.0/warehouses/<id>"
streamlit run app.py
```

Auth comes from the ambient SDK config. **The warehouse ID must belong to the workspace the
profile points to** — a DEFAULT profile aimed elsewhere yields a misleading PERMISSION_DENIED
from the wrong workspace. The Fleet tab prints which host it is actually querying; check it first
when debugging.

### Submit a workload to AIR

From the Submit tab: **Download YAML**, then `air run -f <file>.yaml`, or click
**Submit via air CLI** if the `air` binary is on PATH where the app runs (it won't be in a
deployed Databricks App — download-and-submit is the path there).

### Deploy as a Databricks App

```bash
cp -r ../../utils .   # vendor the shared query module (utils/billing/queries.py)
databricks apps create training-hub
databricks sync . /Workspace/Users/<you>/training-hub
databricks apps deploy training-hub --source-code-path /Workspace/Users/<you>/training-hub
```

Grant the app's service principal: `SELECT` on `system.billing.usage`, CAN_USE on the
warehouse, and Jobs API visibility on the workloads it should list.

## Reference

### Layout

- `app.py` — Streamlit entrypoint (two tabs)
- `hub/config.py` — teams/quota/reservation config loader (falls back to the example file)
- `hub/usage.py` — billing queries against `system.billing.usage` (needs a SQL warehouse)
- `hub/jobs.py` — active runs via the Jobs API
- `hub/templates.py` — AIR workload YAML generation + `air` CLI submit helper
- `app.yaml` — Databricks Apps runtime config
- `OTEL-INTEGRATION.md` — the hub's telemetry contract for workloads — **required data source
  for the Fleet tab's utilization/health views** (see the `air-otel-telemetry` skill)

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `HUB_TEAMS_CONFIG` | Path to teams/reservation YAML | `config/teams.yaml`, else the example file |
| `HUB_WAREHOUSE_HTTP_PATH` | SQL warehouse for billing queries | unset → usage panel shows an error |
| `DATABRICKS_CONFIG_PROFILE` | Workspace auth for SDK + SQL | ambient SDK resolution |

### Workload templates

Defined in `hub/templates.py`; schema mirrored from `workloads/*.example.yaml`, verified against
`air` v0.1.0 (`env_variables` is rejected — set env inline in the command).

- `single-node-finetune` — one node, snapshot code source; the default starting point
- `multi-node-distributed` — multi-node via CLI (the only multi-node path); torchrun in command

Accelerators offered: `GPU_1xA10`, `GPU_8xA10`, `GPU_1xH100`, `GPU_8xH100`
(full catalog vs. current CLI: unverified, see TODO in `templates.py`).

## Explanation

### Why this exists

At a customer with a multi-team reserved GPU pool, allocation is coordinated **manually by a
platform lead**: teams ask them for capacity, they arbitrate team-by-team, and org leaders have
no view of utilization vs. the reservation. Cadence is driven by model release schedules. The
platform product does not yet provide per-team access control or per-workload chargeback inside
a reserved pool, so the hub fills the gap operationally (config-declared quotas + visibility)
rather than by enforcement.

### Data-source requirement (2026-08-06)

Fleet utilization, availability, and run-health views **must be built on the hub's OTEL
telemetry tables** (`OTEL-INTEGRATION.md`). `system.billing.usage` is joined for spend $ only;
the Jobs API supplies run inventory only. Rationale: reserved pools bill as aggregate records
and per-workload attribution inside a pool is not yet a product capability — the OTEL pipeline
is the per-run, per-team utilization source the hub controls end-to-end, and the Submit tab
injects the emission contract into every generated workload so the fleet view populates itself.

### Why quotas are declared, not enforced

The product has no per-team entitlement today; the hub makes over/under-quota visible so the
platform lead stops being the middleman. Enforcement waits on the product.

### Known gaps (tracked as repo open questions)

- **Attribution inside a reserved pool is coarse** (open-q #5/#16): pools bill aggregate
  records; per-principal rows may be incomplete until per-workload tagging ships. The hub shows
  what the system tables expose and labels unattributed spend as such.
- **Identifying AIR runs among job runs** — filter heuristic TBD (see `hub/jobs.py`); the
  Fleet tab currently lists all active job runs.
- **In-app submission** uses the `air` CLI when present; the underlying REST surface is not yet
  documented — direct API submission is a follow-up.

### Roadmap sketch

1. Utilization vs. reservation over time — from OTEL `gpu.utilization.percent` per team
   (required source, see above); DBU→node-hour rate per SKU only for the $ overlay
2. Capacity request/approval flow (replace ad-hoc pings to the platform lead)
3. Release-calendar overlay (teams' cadence is release-driven)
4. Per-run cost once per-workload tagging lands
