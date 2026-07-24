# Run notes — 2026-07-16 (e2-demo-field-eng, ws 1444828305810485, us-west-2)

## What worked end-to-end ✅

- **Docker path on AIR is fully functional.** podman build (no Docker Desktop license needed) →
  push to personal Docker Hub (`forrestm/air-otel-smoke:0.1`) → `air register image` (~2 min) →
  `air run` → **Job SUCCESS** on A10G, 14s user code. Run: jobs/runs/37776040541298.
- Injected env vars confirmed on a real node: `NUM_NODES=1, LOCAL_WORLD_SIZE, WORLD_SIZE=1,
  POD_RANK=0/NODE_RANK=0` + `MOSAICML_PLATFORM=true, MOSAICML_LOG_DIR=/databricks/customer-logs`
  (MosaicML lineage), `PWD=/mnt/work`, venv on PATH. GPU: A10G, driver 580.126.16, CUDA 13.0.
- **AIR → Zerobus egress is OPEN** (open-q #7b): the container reached
  `<ws>.zerobus.us-west-2...:443` and got app-layer responses. Serverless egress policy did not block.

## Cross-build gotchas (Apple Silicon → linux/amd64)

- `uv` binary segfaults under qemu-user; base image venv has no pip/ensurepip.
  Fix: vendor wheels on host (`uv pip install --target vendor --python-platform
  x86_64-unknown-linux-gnu --python-version 3.12 --only-binary :all:`) + COPY + PYTHONPATH.
- podman machine (libkrun) has no Rosetta here; emulation is fine for COPY-only builds.

## air CLI v0.1.0 schema deltas vs docs

- `env_variables` and `secrets` are TOP-LEVEL, not under `environment`.

## ⚠️ Secret-handling lesson

- `env | sort` in `command:` dumped the injected ZEROBUS_TOKEN secret into job logs
  (bounded: 1h-expiry user token). Never dump raw env in workloads; filter TOKEN/SECRET/KEY.

## 🔴 Zerobus OTLP: root cause of "export OK, zero rows"

Symptom: OTLP Export RPC returns OK from laptop AND from AIR node; tables stay empty;
no stream ever appears in `system.lakeflow.zerobus_stream` for our tables.

Root cause (proven with raw HTTP/2 curl):
```
HTTP/2 200
x-databricks-reason-phrase: Invalid token audience   ← real error
grpc-status: 0                                       ← reported as SUCCESS
grpc-message: {"X-Databricks-Reason-Phrase":"Invalid token audience"}
```
1. The bearer token must be minted with `resource=api://databricks/workspaces/<WS_ID>/zerobusDirectWriteApi`
   AND `authorization_details` (UC privileges JSON) — docs only show this via **service principal
   client_credentials**. A normal `all-apis` user/CLI token → "Invalid token audience".
2. **Edge bug/footgun: auth failures are returned with `grpc-status: 0` (OK)** — every OTLP
   client treats the export as successful. Silent data loss. Verified: garbage token, nonexistent
   table, and bogus workspace ID ALL return OK. → report to Zerobus team (on-call playbook:
   Confluence UN/5268832571). Huge customer-relevance: silent-drop observability pipe.

Dead ends tried: RFC 8693 token-exchange of user token for the zerobus resource — endpoint exists
but requires `authorization_details` already in the subject token's claims → SP-only in practice.
Also ruled out: row tracking, variant shredding, catalog commits, default storage, region mismatch.
Note `system.lakeflow.zerobus_stream` is ACCOUNT-WIDE per region — other workspaces' streams
(trustly_demo, demos_prod) show OTLP working in us-west-2 generally.

## Blocked on

- An SP (client id + secret) with USE CATALOG/USE SCHEMA on `users`/`users.forrest_murray` and
  SELECT+MODIFY on the two `air_otel_*` tables — SP creation is admin-only on e2-demo-field-eng
  (forrest is not admin). Then: mint token per docs recipe, store in `air_lab` scope, re-run
  job 37776040541298's config — everything else is proven.

## Cleanup / debt

- Tables: `users.forrest_murray.air_otel_{logs,metrics,logs_nort}` (logs_nort was a row-tracking
  hypothesis test — droppable).
- Token in job-run logs (expired ~18:26 UTC). Secret scope `air_lab` holds `zerobus_token` (stale).
- Docker Desktop quit but still installed; podman is the sanctioned-adjacent path (license doc:
  Confluence UN/3570664348 recommends Arca; podman avoids the license question entirely).

## Run notes — 2026-07-17 (fevm-forrest-aws-stable, ws 7474645949300216, us-east-1)

### ✅ FULL PIPELINE PROVEN (v0.3, runs 774891161163006 / 382445774856272)

- SP `air-lab-zerobus` (appId 292da548-…) + workspace-level secret via `databricks
  service-principal-secrets-proxy create <sp-id>` (admin required).
- Working token recipe (the whole unlock): client_credentials + `resource=api://databricks/
  workspaces/<WS_ID>/zerobusDirectWriteApi` + `authorization_details` (UC privileges JSON).
  Minted per-table in-container (train.py). Correct-audience probe → row visible in Delta in seconds.
- **GPU metrics** land via NVML observable gauges (util/mem/power/temp per GPU) — verified from A10.
- **Requester identity**: `air.requester` derived from HYPERPARAMETERS_PATH → resource attr on all
  signals AND per-record log attribute via OTEL baggage + logging.Filter. ⚠️ HYPERPARAMETERS_PATH only
  exists when the YAML has a `parameters:` block — without one, requester silently absent.
- Per-table downscoping enforced for real: logs-only token → PERMISSION_DENIED on metrics export
  (and post-auth errors DO surface to the client; the grpc-status:0 silent-OK is edge-auth-only).
- Authoritative identity: join `air.mlflow_run_id` → system.lakeflow.job_run_timeline
  (utils/visibility/telemetry_identity.sql). Feature ask for Zerobus PM: server-side principal
  stamping; per-team SPs are the interim authenticated-attribution model.

### PCI compliance (checked 2026-07-17)

- Public PCI DSS v4.0 matrix lists **Lakeflow Connect - Zerobus Ingest ✓** and **AI Runtime
  Interactive ✓** (docs/aws/en/security/privacy/pci). AIR docs: all standards except FedRAMP High /
  DoD IL5; AIR CLI + custom Docker (DCS for air) inherit AIR's compliance posture.
- Zerobus also listed under HIPAA + HITRUST; being default-enabled for compliance-security-profile
  workspaces mid-July 2026. (Older internal FAQ saying "No PCI" is stale.)
- Operative requirement: the *workspace* must have the compliance security profile enabled — the
  components are certified, the deployment context does the qualifying. Customer deltas to flag:
  Docker-Hub-only image hosting (their security won't love public/external registry), and
  cross-region GPU fallback vs their Private-Link-only posture.

## Deploy notes — 2026-07-24 (fe-sandbox-mkazia-lw2, ws 7474656734648830, us-east-1)

Goal: replicate the proven fevm pipeline in the shared test workspace (catalog
`mkazia_lw2_catalog_7474656734648830`, per owner's direction). PrivateLink workspace
(host aliases to nvirginia.privatelink); serverless warehouse egresses via proxy to
**S3 FIPS endpoints** → looks like a hardened/compliance-profile sandbox.

### Done (my access: `users` group only, broad grants on the target catalog)

- Profile `mkazia-lw2` (OAuth U2M). Warehouse: Serverless Starter `e7e6ecf78c767db6`.
- Schema `mkazia_lw2_catalog_7474656734648830.airlab` created (metadata op — succeeded).
- Secret scope `air_lab` created (empty until an SP exists).
- Live YAML `workloads/docker-otel-zerobus-mkazia.yaml` (gitignored).
- Image `docker.io/forrestm/air-otel-smoke:0.3` registered via `air register image -p mkazia-lw2`.
- **Zerobus edge live for this ws**: raw probe (bogus token) → HTTP/2 200, `grpc-status: 16`,
  `x-databricks-reason-phrase: Malformed token` (request 0fb234dd, 2026-07-24). NB: malformed
  token → 16 at edge; it's *plausible-but-wrong-audience* tokens that get the silent status-0 drop.

### Blocked — needs workspace admin (mkazia)

1. **Catalog storage is broken from serverless**: ALL table reads/writes in
   `mkazia_lw2_catalog_7474656734648830` fail `[UNAUTHORIZED_ACCESS]` — S3 HEAD 403 on
   `s3://mkazia-lw2-catalog-7474656734648830/...` with UC-vended session creds
   (S3 request 29SYE4VY3CM208DE via s3-fips + proxy 192.168.200.20). Isolation: pre-existing
   `default.t1` also unreadable; `system.billing.usage` + `samples` read fine from the same
   warehouse ⇒ bucket policy / storage-credential IAM role for THIS bucket, not my grants and
   not serverless egress generally. Likely aws:SourceVpce-style bucket policy that doesn't
   allowlist serverless (NCC) endpoints. OTEL tables can't be created until fixed.
2. **SP creation admin-only** (SCIM 403). fevm SP is invalid_client here (different account or
   not workspace-assigned) — can't reuse. Need an SP created+assigned here, or admin for Forrest.

Open risk once unblocked: Zerobus commits server-side into the same bucket — if the bucket
policy is VPCe-scoped it may 403 Zerobus's writer too. Test immediately after the policy fix.
