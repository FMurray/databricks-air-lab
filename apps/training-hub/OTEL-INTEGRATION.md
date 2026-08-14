# OTEL (Zerobus) integration for Training Hub

Status: **REQUIREMENT** (decided 2026-08-06; was scoping) — the hub's OTEL telemetry tables are
the mandated data source for the Fleet tab's utilization, availability, and run-health views.
`system.billing.usage` joins in for spend $ only; the Jobs API for run inventory only. Not yet
built — the sizing table below is now the committed order of work.

Why a requirement, not an option: reserved pools bill as one aggregate record, per-workload
tagging inside pools has no committed design yet, and per-pool usage dashboards are a product
roadmap concept — this pipeline is the only per-run/per-team utilization source under our
control today (and its components are PCI-listed, which matters for the target workspace).

Foundation it builds on: the proven OTLP→Zerobus pipeline in `experiments/docker-otel-zerobus/`
(SP-minted audience-scoped tokens, GPU gauges, requester identity via resource attrs + baggage;
NOTES.md there has the sharp edges).

## Why the hub is the right integration point

The hub sits on both sides of the telemetry loop:
- **Consumer**: the Fleet tab needs what billing can't give — *live* utilization, run health, and
  per-workload attribution inside a reserved pool (billing bills pools as one aggregate record).
- **Enabler**: the Submit tab *generates the workload YAML*. If telemetry must be hand-wired per
  workload it will never happen org-wide; if the hub injects the contract into every generated
  YAML, the fleet view populates itself.

## The contract (decide first, everything else follows)

1. **Well-known tables** per workspace: `<catalog>.<schema>.air_otel_logs|metrics` — add a
   `telemetry:` section to `teams.yaml` (catalog, schema, region, workspace_id). One pair of
   tables per pool/workspace; teams separated by attributes, not tables (revisit if isolation
   demands per-team tables — downscoped tokens make either work).
2. **Attribute conventions** (already emitted by the experiment code):
   - `service.name` = experiment name
   - `air.requester` (derived from HYPERPARAMETERS_PATH — requires a `parameters:` block!)
   - `air.team` (NEW — hub knows the team at submit time; today attribution relies on
     requester→team mapping in teams.yaml)
   - `air.mlflow_run_id` (correlation key to Jobs/MLflow/system tables)
   - `air.node_rank`, `air.world_size`; metrics additionally `gpu` index per series
3. **Identity/auth model**: per-team SPs (team → SP appId in teams.yaml), tokens minted in-workload
   with per-table `authorization_details` (enforcement verified: wrong scope → PERMISSION_DENIED).
   Interim: one shared SP + self-reported `air.team`. Authenticated-attribution gap goes to the
   Zerobus PM (server-side principal stamping feature ask).

## Requirement A — Fleet tab consumes telemetry (the mandated source)

New `hub/telemetry.py` + queries in `utils/visibility/queries.py` (mirror the billing pattern:
one importable query module shared by app/CLI/notebooks). Panels, in value order:

1. **Pool utilization vs reservation** — avg/p95 `gpu.utilization.percent` and memory headroom by
   team over the lookback, next to declared quotas. This is the "are we using what we reserved"
   view the platform lead arbitrates blind today.
2. **Run health board** — join Jobs API active runs ↔ latest telemetry per `air.mlflow_run_id`:
   last loss, current GPU util, minutes since last event. A stale heartbeat on an active run =
   hung workload (this exact view would have caught the XGBoost-on-H100 hang in minutes).
3. **Failure feed** — last N `severity_number >= WARN` log rows with requester/team.
4. **Right-sizing signals** — telemetry util × billing DBUs: "team X ran 40 H100-hours at 3%
   utilization" (utils backlog #3's data source).

Mechanics: same SQL warehouse the hub already uses; `st.cache_data(ttl=30)` — Zerobus
time-to-table is P50 ≤5s, so a 30s refresh reads as "live" without hammering the warehouse.

## Requirement B — Submit tab injects the contract (default-on)

Telemetry injection is mandatory in generated YAMLs, not opt-in — a fleet view built on OTEL
only works if every workload emits. `telemetry=False` remains as an escape hatch and renders a
visible "this run will be invisible to the Fleet tab" warning in the UI.

1. `templates.build_workload(telemetry=True)` (the default) adds to every generated YAML:
   - env: WORKSPACE_ID/URL, ZEROBUS_REGION, table names, AIR_TEAM
     ⚠️ schema conflict to re-verify first: templates.py says env_variables is rejected by
     air v0.1.0, but the fevm runs (2026-07-17) used TOP-LEVEL env_variables successfully.
     Likely `environment.env_variables` (rejected) vs top-level (works). If truly rejected,
     fall back to an export preamble in `command`.
   - secrets: team SP client id/secret refs
   - a `parameters:` block (without it HYPERPARAMETERS_PATH is absent and requester silently
     disappears — the sharpest edge we hit)
2. **Package the emission helper** — ✅ the module exists: `utils/telemetry/airtel.py`
   (`airtel.init(service_name=...)` → log bridge + GPU gauges + identity, no-ops off-GPU;
   see the `air-otel-telemetry` skill — never hand-write telemetry wiring). Remaining scope
   is delivery only: pip-installable from a UC volume wheel, or `dependencies:` entry, or
   vendored via snapshot include_paths. Stretch: an `airtel-run -- python train.py` wrapper
   for zero-code-change workloads (sets up the logging bridge, execs the real command).
3. **Hub audit trail (self-telemetry)** — the app emits its own OTEL log events (who generated /
   submitted which template, team, accelerators) to the same logs table with
   `service.name=training-hub`. Databricks Apps run as an app SP → grant it MODIFY on the tables;
   it mints its own tokens. This gives the platform lead a submission audit log for free.

## Sizing & order

| # | Item | Size | Depends on |
|---|---|---|---|
| 1 | `telemetry:` config + `hub/telemetry.py` + utilization & failure panels | S | tables exist (done in fevm) |
| 2 | Re-verify env_variables schema; template injection incl. `parameters:` block | S | — |
| 3 | `airtel` packaging + wheel-on-volume delivery | M | 2 |
| 4 | Run-health join panel | S | 1 |
| 5 | Hub audit events (app SP grants) | S | 1 |
| 6 | Right-sizing (telemetry × billing join) | M | 1 + billing queries |
| 7 | Per-team SP provisioning story | M (org, not code) | admin |

## Open questions / risks

- **env_variables schema** — reconcile templates.py's "rejected" note vs the working fevm YAMLs (both claim v0.1.0 verification; probably `environment.env_variables` vs top-level).
- **Cost/retention** of OTEL tables: Zerobus ingestion pricing, OPTIMIZE/retention policy (tables are liquid-clustered; keep predictive optimization on per docs).
- **Multi-node fan-in**: 16-GPU runs emit from every node — verify per-node exporters don't need coordination (they shouldn't; rank attr separates them). Volume estimate: ~5 gauge series × nodes × 6/min — trivial vs the 15k rec/s stream limit.
- **PCI workspaces**: pipeline components are PCI-listed (see experiment NOTES); the hub app itself would also need to run in the compliance-profile workspace for the customer case.
- **App deployment**: deployed Apps can't shell out to the `air` CLI (existing limitation in templates.submit) — submission via Jobs API from the app SP is the real fix, separate scope.
- **Don't rebuild MLflow**: loss curves and per-run deep-dives stay in MLflow; the hub's telemetry views are fleet-level only. Keep that boundary or scope creeps to a monitoring product.
