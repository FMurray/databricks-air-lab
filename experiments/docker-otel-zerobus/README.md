# Docker on AIR + OTEL event logs → Zerobus → UC Delta

Two questions in one experiment:

1. **Docker path mechanics** — build/register/run a custom image on AIR (DCS, Beta).
2. **Can event logs ship as OTEL via Zerobus?** — **Yes, natively.** Zerobus Ingest exposes an
   OTLP/**gRPC** endpoint (Beta): logs, traces, and metrics land in pre-created UC Delta tables using
   standard OTEL SDKs. No custom bridge/exporter needed.

## Why this combo matters

- The training container emits structured events (loss, epoch timing, OOM warnings) straight into a
  governed Delta table — queryable next to `system.billing.usage`. Feeds the admin/visibility
  deliverable (`utils/visibility/`).
- **OTLP is language-neutral**: the same pipe works from the Java OTEL SDK. This is the observability
  story for non-Python AIR workloads (the customer's Java question) where auto-MLflow doesn't exist.

## Zerobus OTLP facts (docs.databricks.com/aws/en/ingestion/opentelemetry/)

- Endpoint: `https://<workspace-id>.zerobus.<region>.cloud.databricks.com:443` — **gRPC only, no OTLP/HTTP**.
- Per-request gRPC metadata: `authorization: Bearer <token>` + `x-databricks-zerobus-table-name: <catalog>.<schema>.<table>`.
  One table per request ⇒ **separate exporter per signal** (logs / spans / metrics tables).
- Tables must pre-exist with the predefined v2 OTEL schemas (`setup/create_tables.sql`); attributes/body land as `VARIANT`
  (DBR 15.3+ to query). Rows are denormalized — no joins needed; `service_name` is a top-level column.
- Auth: service principal OAuth (M2M). Needs explicit `MODIFY, SELECT` per table (`ALL PRIVILEGES` is NOT sufficient)
  + `USE CATALOG`/`USE SCHEMA`. Static bearer tokens expire in **1 hour** → long workloads should run an
  OTEL Collector with `oauth2clientauthextension` token refresh; our smoke test mints its own token from
  SP client id/secret at startup instead.

## Docker-on-AIR facts (docs .../ai-runtime/cli/docker-images)

- Base images: `databricksruntime/air:dcs-base-aws-{runtime,devel}` (CUDA/NCCL/EFA preconfigured;
  `devel` only if compiling CUDA extensions). Python venv at `/opt/venv`, managed by `uv`.
- Register per user per tag: `air register image <url> -p <profile>` (2–6 min, pulls & caches). Private
  images: Docker Hub PAT via `docker login` / `--interactive-authenticate` / `--scope/--key` secret.
- Docker Hub only; <20 GB; `WORKDIR` ignored (absolute paths, e.g. `/app/...`); no `environment.dependencies`/`version` alongside.
- **Injected env vars:** `NUM_NODES`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, `POD_RANK`/`NODE_RANK`,
  and multi-node only: `LOCAL_ADDR`, `MASTER_ADDR`, `MASTER_PORT`.

## Runbook

```bash
# 0. One-time: create OTEL tables + SP grants (Databricks SQL)
#    setup/create_tables.sql   (replace <catalog>.<schema>, <sp-uuid>)

# 1. Build & push (Docker Hub only!)
docker build -t <dockerhub-org>/air-otel-smoke:0.1 experiments/docker-otel-zerobus/
docker push <dockerhub-org>/air-otel-smoke:0.1

# 2. Register with AIR
air register image <dockerhub-org>/air-otel-smoke:0.1 -p <profile>

# 3. Put SP creds in a secret scope referenced by the workload YAML
databricks secrets put-secret air_lab zerobus_sp_client_id
databricks secrets put-secret air_lab zerobus_sp_client_secret

# 4. Run (fill in env_variables in the YAML copy first)
cp workloads/docker-otel-zerobus.example.yaml workloads/docker-otel-zerobus.yaml
air run --file workloads/docker-otel-zerobus.yaml -p <profile>

# 5. Verify rows landed
#    SELECT time, service_name, severity_text, body:msg::string, attributes
#    FROM <catalog>.<schema>.air_otel_logs ORDER BY time DESC LIMIT 50;
```

## What to record in NOTES.md

- [ ] Registration time; whether re-register needed after new tag push
- [ ] Egress: can the AIR container even reach `*.zerobus.*.cloud.databricks.com:443`? (serverless network policy!)
- [ ] Latency: emit → row visible in Delta
- [ ] Batching behavior under `BatchLogRecordProcessor` defaults; any throttling (check Zerobus limits page)
- [ ] Token expiry behavior on a >1h run (expect export failures after 60 min → measure, then test collector-sidecar-in-container refresh)
- [ ] Injected env vars observed vs documented (also closes open-q #3 for the non-Docker path if same)
