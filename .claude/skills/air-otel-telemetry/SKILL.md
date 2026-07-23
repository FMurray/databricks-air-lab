---
name: air-otel-telemetry
description: >-
  Wire OTEL telemetry (structured logs, training metrics, GPU utilization, requester identity)
  from Databricks AIR / serverless GPU workloads into Unity Catalog Delta tables via the Zerobus
  OTLP endpoint. Use this whenever a task involves adding telemetry, logging, metrics, GPU
  monitoring, observability, or event export to an AIR workload, training job, or serverless GPU
  container — including "ship logs to Delta", "track GPU utilization across runs", "fleet
  observability", "chargeback telemetry", or wiring the Training Hub's telemetry contract into a
  workload. Also use it when debugging why OTLP exports to Zerobus silently produce no rows.
---

# OTEL telemetry for AIR workloads (via Zerobus)

Ships structured logs + metrics (incl. GPU) from training containers into UC Delta tables,
SQL-queryable and joinable with `system.billing.usage`. Complements MLflow (per-run UI), does not
replace it: this pipe is for fleet-level views across runs/teams.

**The one fact that saves hours:** the Zerobus edge reports auth failures as `grpc-status: 0`
(success). A misconfigured token means every OTLP client reports successful exports while rows
silently never appear. Never trust "export OK" — verify rows with SQL, and probe auth first
(`scripts/probe.py`).

## Prerequisites (one-time per workspace)

1. **Delta tables** with the predefined OTEL v2 schemas — run `references/tables.sql`
   (replace `<catalog>.<schema>`). Tables must pre-exist; Zerobus never creates them.
2. **Service principal** + OAuth secret. Workspace admin:
   `databricks service-principals create --display-name <name>` then
   `databricks service-principal-secrets-proxy create <numeric-sp-id>`.
3. **Grants** (explicit — `ALL PRIVILEGES` is NOT sufficient): `USE CATALOG`, `USE SCHEMA`,
   and `MODIFY, SELECT` per table, to the SP's application id.
4. **Secret scope** holding the SP creds:
   `databricks secrets create-scope <scope>` +
   `put-secret <scope> zerobus_sp_client_id` / `zerobus_sp_client_secret`.

## Auth (the part everyone gets wrong)

A plain `all-apis` token is rejected ("Invalid token audience" — hidden behind grpc-status 0).
Tokens must be minted with **both**:
- `resource=api://databricks/workspaces/<WORKSPACE_ID>/zerobusDirectWriteApi`
- `authorization_details` = UC-privileges JSON for the ONE target table

One token per signal/table. Only SP client_credentials supports this (user tokens can't carry
authorization_details). Full recipe + curl diagnostics: `references/auth-and-tokens.md`.

## Wiring a workload

1. Copy `assets/airtel.py` next to the training code. It provides
   `airtel.init(service_name=...)` → configures the log bridge, loss/step meters, per-GPU NVML
   gauges (graceful no-op off-GPU), identity resource attrs, and baggage propagation. Training
   code then just uses stdlib `logging` and the returned meter.
2. Dependencies (pip/uv or Dockerfile or vendored wheels):
   `opentelemetry-sdk>=1.27  opentelemetry-exporter-otlp-proto-grpc>=1.27  requests  nvidia-ml-py`
   (gRPC only — Zerobus has **no OTLP/HTTP endpoint**.)
3. Workload YAML must carry (see `references/workload-contract.md` for a complete example):
   - `env_variables` (TOP-LEVEL, not under `environment:` — nesting it is rejected):
     `WORKSPACE_ID`, `WORKSPACE_URL`, `ZEROBUS_REGION`, `OTEL_LOGS_TABLE`, `OTEL_METRICS_TABLE`
   - `secrets`: `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` → the SP scope keys
   - **a `parameters:` block — even a dummy one.** Without it AIR doesn't inject
     `HYPERPARAMETERS_PATH`, and requester-identity derivation silently returns nothing.
   - Never `env | sort` (or any raw env dump) in `command:` — workload secrets are env vars
     and land in job logs. Filter if you must: `env | grep -vE 'TOKEN|SECRET|KEY'`.

## Verify (always, in this order)

1. **Probe before training**: `scripts/probe.py` sends one synchronous log record and prints the
   real gRPC outcome; its `--raw` mode shows the edge's `x-databricks-reason-phrase` header that
   the OTLP client hides. Run it from the target network context when possible.
2. **Count rows**: time-to-table is P50 ≤5s. If the probe said OK and rows aren't visible within
   ~30s, treat it as auth/config failure, not latency — go to `references/auth-and-tokens.md`
   diagnostics.
3. After the real run: check both tables, and confirm identity attrs are present
   (`resource.attributes["air.requester"]` non-null — if null, the `parameters:` block is missing).

## Identity semantics (be honest about them)

`air.requester` is derived in-container from `HYPERPARAMETERS_PATH` and stamped as a resource
attribute + per-log-record attribute (OTEL baggage). It is **self-reported** — fine for dashboards,
not for chargeback. For authoritative attribution, join `air.mlflow_run_id` against
`system.lakeflow.job_run_timeline` (platform-recorded creator). Authenticated attribution =
per-team SPs whose tokens can only write their tables.

## Compliance note

Zerobus Ingest (incl. OTLP) and AI Runtime are both on the PCI DSS v4.0 support matrix
(docs.databricks.com/aws/en/security/privacy/pci); certification applies when the *workspace* runs
the compliance security profile.
