# UAT suite — deploying the full lab to a target workspace

The repo doubles as a UAT suite for AIR: each experiment family exercises one product surface
the customer cares about. "Deploying" the suite to a workspace = staging its assets there and
getting each family to a verified green run. This doc tracks the current target deployment;
the procedure generalizes to any workspace.

Current target: `fe-sandbox-mkazia-lw2` (ws 7474656734648830, us-east-1, PrivateLink +
FIPS/proxy egress — a hardened sandbox, which is the point: it approximates the customer's
network posture better than our open sandboxes). Profile: `mkazia-lw2`. Catalog:
`mkazia_lw2_catalog_7474656734648830` (schema `airlab`). Warehouse: `e7e6ecf78c767db6`.

## Suite matrix (status 2026-07-24)

| # | Family / asset | Workload(s) | Compute | Status on target |
|---|---|---|---|---|
| 1 | Runtime contract probe | `exec-probe.yaml` | 1×A10 | 🔄 run 781012703677665 in flight |
| 2 | OTEL→Zerobus telemetry (Docker) | `docker-otel-zerobus-mkazia.yaml` | 1×A10 | ⛔ blocked: catalog bucket 403 + no SP (image ✅ registered, scope ✅, schema ✅, Zerobus edge ✅ live) |
| 3 | Multi-language / JVM (DJL) | `djl-train.yaml` | 1×A10 | ⏸ gated on #1 (egress + exec findings) |
| 4 | Classic ML (TabICL) | `tabicl-bench.yaml` → `tabicl-memprobe` | 1×A10 → 1×H100 | ⏸ gated on #1 |
| 5 | Multinode probe (cheap) | `multinode-probe.yaml` + A10 override | 2×(1×A10) | ⏸ gated on #1 |
| 6 | Multinode at scale | `multinode-correctness`, `fsdp-multinode` | 2×(8×H100) | 📋 staged; submit deliberately only (cost) |
| 7 | Training Hub app | `apps/training-hub/` | Apps | ⛔ blocked: Apps disabled for the org |
| 8 | Billing/visibility SQL | `utils/billing/`, `utils/visibility/` | warehouse | ✅ runnable (system tables readable); telemetry joins wait on #2 |
| 9 | Node acceptance: burn + health | `gpu-burn.example.yaml` | 1×A10 dry → 8×H100/node | 🆕 built 2026-07-24; dry-run gated on #1 |
| 10 | Node acceptance: all-reduce bench | `nccl-allreduce.example.yaml` | 2×A10 dry → 8/16×H100 | 🆕 built 2026-07-24; dry-run gated on #1 |
| 11 | Env flexibility: vLLM in std env | `vllm-smoke.example.yaml` | 1×A10 dry → 1×H100 | 🆕 built 2026-07-24; needs HF egress (probe reports) |
| 12 | Classic ML: XGBoost GPU (hang repro) | `xgboost-gpu.example.yaml` | 1×A10 control → 1×H100 | 🆕 built 2026-07-24 |
| 13 | FM: LoRA fine-tune | `lora-finetune.example.yaml` | 1×A10 dry → 8×H100 | 🆕 built 2026-07-24; needs HF egress |

## Admin prerequisites on the target (owner asks)

1. **Catalog bucket policy**: all table I/O in the target catalog 403s from serverless with
   UC-vended creds (pre-existing tables too; `system`/`samples` fine from the same warehouse).
   Likely a VPC-endpoint-scoped bucket policy missing the serverless (NCC) endpoints. Blocks #2
   tables and any Delta output; risk it also blocks Zerobus's server-side writer — retest after fix.
2. **Service principal**: SCIM create is admin-only; SPs from other workspaces are
   `invalid_client` here. Blocks #2 auth. Secret scope `air_lab` is ready to receive creds.
3. **Enable Databricks Apps** for the org. Blocks #7.

## Deploy procedure (per workspace)

1. `databricks auth login --host <ws-url> --profile <name>` (U2M OAuth).
2. Permission recon: groups, catalog grants, warehouse id, `databricks apps list`,
   SP-creation probe. Record blockers before burning GPU time.
3. Region → Zerobus endpoint via workspace DNS (`host <ws-host>` → region alias); liveness:
   skill probe `--raw` with a bogus token (expect `Malformed token`, grpc-status 16).
4. Stage: schema + OTEL tables (skill `references/tables.sql`), secret scope, SP creds,
   grants (MODIFY+SELECT explicit per table), `air register image` for Docker workloads.
5. Live YAMLs: `code_source` workloads are workspace-portable (profile picks the target at
   submit); only Docker/telemetry configs need a per-workspace copy (env: workspace id/url,
   region, tables).
6. Validate cheap-first: exec-probe (1×A10) → per-family A10 smokes → single-H100 → multinode
   A10 override → 8×H100 only deliberately.
7. Record every run in the family's `experiments/*/NOTES.md` with run id + workspace + date
   (see `.claude/skills/experiment-verification/SKILL.md`).

## Verified facts about this target so far

- AIR enabled; Docker image registration works (`forrestm/air-otel-smoke:0.3`,
  sha256:7f48a7fc8…, 2026-07-24) — DCS replication functional.
- Zerobus edge answers for the workspace id (reason-phrase `Malformed token` on bogus token).
- Serverless SQL works against `system` + `samples`; the target catalog's own bucket is the
  only storage failing (isolation receipts in `experiments/docker-otel-zerobus/NOTES.md`).
- Prior AIR usage exists (another user's `gpu-1xa10 ssh-tunnel` secret scopes) — GPU capacity
  has been exercised here before.
