# Training Hub

A Databricks App prototype for teams sharing a reserved serverless-GPU (AIR) pool.

## The problem

At a customer with a multi-team reserved GPU pool, allocation is coordinated **manually by a platform
lead**: teams ask him for capacity, he arbitrates team-by-team, and org leaders have no view of
utilization vs. the reservation. Cadence is driven by model release schedules. The platform product
does not yet provide per-team access control or per-workload chargeback inside a reserved pool, so the
hub fills the gap operationally (config-declared quotas + visibility) rather than by enforcement.

## MVP (thin slice of both personas)

- **Fleet view (admin):** reservation summary, declared team quotas, GPU-SKU spend by team from
  `system.billing.usage`, and currently-active runs.
- **Submit view (team member):** generate a valid AIR workload YAML from a template
  (schema mirrored from `workloads/*.example.yaml`), download it or submit via the `air` CLI.

## Layout

- `app.py` — Streamlit entrypoint (two tabs)
- `hub/config.py` — teams/quota/reservation config (`config/teams.yaml`, gitignored; see
  `config/teams.example.yaml`)
- `hub/usage.py` — billing queries (needs a SQL warehouse: set `HUB_WAREHOUSE_HTTP_PATH`)
- `hub/jobs.py` — active runs via the Jobs API
- `hub/templates.py` — AIR workload YAML generation + submit helper
- `app.yaml` — Databricks Apps runtime config

## Run locally

```bash
cd apps/training-hub
pip install -r requirements.txt
cp config/teams.example.yaml config/teams.yaml   # fill in real teams/emails/quotas
export HUB_WAREHOUSE_HTTP_PATH="/sql/1.0/warehouses/<id>"
export DATABRICKS_CONFIG_PROFILE=<profile-for-the-target-workspace>
streamlit run app.py
```

Auth comes from the ambient SDK config. **The warehouse ID must belong to the workspace that
profile points to** — a DEFAULT profile aimed elsewhere yields a misleading PERMISSION_DENIED
from the wrong workspace. The Fleet tab shows which host it's actually querying.

## Deploy as a Databricks App

```bash
cp -r ../../utils .   # vendor the shared query module (utils/billing/queries.py)
databricks apps create training-hub
databricks sync . /Workspace/Users/<you>/training-hub
databricks apps deploy training-hub --source-code-path /Workspace/Users/<you>/training-hub
```
Grant the app's service principal: `SELECT` on `system.billing.usage`, CAN_USE on the warehouse,
and Jobs API visibility on the workloads it should list.

## Known gaps (tracked as repo open questions)

- **Attribution inside a reserved pool is coarse** (open-q #5/#16): pools bill aggregate records;
  per-principal rows may be incomplete until per-workload tagging ships. The hub shows what the
  system tables expose and labels unattributed spend as such.
- **Identifying AIR runs among job runs** — filter heuristic TBD (see `hub/jobs.py`).
- **In-app submission** uses the `air` CLI when present; the underlying REST surface is not yet
  documented — direct API submission is a follow-up.
- Quotas are **declared, not enforced** — the product has no per-team entitlement today; the hub
  makes over/under-quota visible so the platform lead stops being the middleman.

## Roadmap sketch

1. Utilization vs. reservation over time (needs DBU→node-hour rate per SKU)
2. Capacity request/approval flow (replace ad-hoc pings to the platform lead)
3. Release-calendar overlay (teams' cadence is release-driven)
4. Per-run cost once per-workload tagging lands
