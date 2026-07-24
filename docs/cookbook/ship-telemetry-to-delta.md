# Ship telemetry to Delta

Goal: structured logs, training metrics, and GPU gauges from every workload, landing in Unity
Catalog Delta tables — SQL-queryable across your whole fleet and joinable with
`system.billing.usage`.

This complements MLflow (per-run UI), it doesn't replace it. The pipe: OTEL SDK in the workload →
Zerobus OTLP endpoint → your Delta tables. ✅ Proven end-to-end 2026-07-17 (fevm workspace, runs
774891161163006 / 382445774856272), including GPU gauges and requester identity.

!!! danger "Verify delivery — 'export OK' does not mean rows landed"
    The Zerobus edge currently returns **`grpc-status: 0` (success) on auth failures**, so a
    misconfigured token makes every OTLP client report successful exports while no rows appear —
    verified with invalid tokens, nonexistent tables, and wrong workspace IDs, all returning OK.
    **Always verify rows with SQL**; the recipe below builds that check in.

## The procedure lives in the skill

This recipe is intentionally a summary. The full procedure — table DDL, token recipe, probe
script, workload contract — is encoded in the **`air-otel-telemetry` skill**
(`.claude/skills/air-otel-telemetry/SKILL.md`, harness-neutral). Agents: invoke it rather than
re-deriving. The shape:

1. **One-time per workspace**: create the OTEL-schema Delta tables (Zerobus never creates them);
   create a service principal + OAuth secret; grant it `USE CATALOG`, `USE SCHEMA`, and per-table
   `MODIFY, SELECT` (explicit grants — `ALL PRIVILEGES` is not sufficient); park the creds in a
   secret scope.
2. **Auth: the critical details**: tokens must be minted via SP client_credentials with
   `resource=api://databricks/workspaces/<WS_ID>/zerobusDirectWriteApi` **and**
   `authorization_details` (UC-privileges JSON) for the one target table. A normal `all-apis`
   token is rejected — behind grpc-status 0.
3. **In the workload**: ship the canonical `utils/telemetry/airtel.py` module via snapshot
   `include_paths` (never hand-write the wiring); one `airtel.init(service_name=...)` call wires
   the log bridge, meters, and NVML GPU gauges. gRPC exporter only — Zerobus has no OTLP/HTTP.
4. **In the YAML**: `env_variables` + `secrets` (both top-level), and **a `parameters:` block,
   even a dummy one** — without it `HYPERPARAMETERS_PATH` isn't injected and requester identity
   silently comes back empty.

## Verify, in this order

1. Probe auth with one synchronous record (`scripts/probe.py` in the skill; `--raw` shows the
   `x-databricks-reason-phrase` header the OTLP client hides).
2. Count rows: time-to-table is P50 ≤ 5s. No rows within ~30s = auth/config failure, not latency.
3. After the real run: confirm `resource.attributes["air.requester"]` is non-null.

## Identity: be honest about it

`air.requester` is **self-reported** (derived in-container) — fine for dashboards, not for
chargeback. Authoritative attribution: join `air.mlflow_run_id` against
`system.lakeflow.job_run_timeline` (`utils/visibility/telemetry_identity.sql`). Authenticated
attribution = per-team SPs whose tokens can only write their team's tables — per-table
downscoping is enforced for real (✅ logs-only token got PERMISSION_DENIED on metrics export).

More on the manager-side view: [Attribute usage](../fleet-ops/attribute-usage.md).
