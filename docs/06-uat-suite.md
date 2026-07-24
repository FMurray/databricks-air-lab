# UAT suite — deploying the full lab to a target workspace

The repo doubles as a UAT suite for AIR: each experiment family exercises one product surface
the customer cares about. "Deploying" the suite to a workspace = staging its assets there and
getting each family to a verified green run. This doc tracks the current target deployment;
the procedure generalizes to any workspace.

Current target: `fe-sandbox-mkazia-lw2` (ws 7474656734648830, us-east-1, PrivateLink +
FIPS/proxy egress — a hardened sandbox, which is the point: it approximates the customer's
network posture better than our open sandboxes). Profile: `mkazia-lw2`. Catalog:
`mkazia_lw2_catalog_7474656734648830` (schema `airlab`). Warehouse: `e7e6ecf78c767db6`.

## Colleague access (how testers run this)

- Repo synced to `/Workspace/Shared/databricks-air-lab` (committed content + this workspace's
  live YAML). **Start at `RUNNING-UAT.md` there** — setup + per-workload commands + ground rules.
- Submission is via the `air` CLI only. Persistent Jobs wrapping `gen_ai_compute_task`
  (jobs/create + run-now) do NOT work: 2/2 runs INTERNAL_ERROR (job 90660381038910, deleted);
  snapshot resolution appears tied to the CLI's SUBMIT_RUN path. `air run --dry-run` DOES
  upload the launch dir + snapshot without spending GPU — useful for staging.

## Suite matrix (status 2026-07-24)

| # | Family / asset | Workload(s) | Compute | Status on target |
|---|---|---|---|---|
| 1 | Runtime contract probe | `exec-probe.yaml` | 1×A10 | ✅ run 903444851308928 (attempt 1, 781012703677665, INTERNAL_ERROR — transient) |
| 2 | OTEL→Zerobus telemetry (Docker) | `docker-otel-zerobus-mkazia.yaml` | 1×A10 | ⛔ blocked: catalog bucket 403 + no SP (image ✅ registered, scope ✅, schema ✅, Zerobus edge ✅ live) |
| 3 | Multi-language / JVM (DJL) | `djl-train.yaml` | 1×A10 | ⏸ gated on #1 (egress + exec findings) |
| 4 | Classic ML (TabICL) | `tabicl-bench.yaml` → `tabicl-memprobe` | 1×A10 → 1×H100 | ⏸ gated on #1 |
| 5 | Multinode probe (cheap) | `multinode-probe.yaml` + A10 override | 2×(1×A10) | ✅ run 128835177125736 — 2-node coordination verified |
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
4. **ROOT CAUSE (unified) — serverless compute cannot reach the workspace's S3 / PyPI.**
   Direct evidence, run 906184622669132 (2026-07-24): in-workload `mlflow.log_artifact` fails
   `HTTPSConnectionPool(host='mkazia-lw2-workspace-root-storage.s3-fips.us-east-1.amazonaws.com')
   Max retries exceeded` — connection-level, from a GPU node, captured via MLflow params (the
   tracking API works). One network misconfiguration produces all of these symptoms:
   - **No logs on any AIR run** (`air logs` empty, zero MLflow log artifacts): the launcher
     ships log chunks to that same root-storage bucket.
   - **Runs TIMEDOUT even after user code completes**: launcher log-shipping hangs in cleanup
     (probe logged `probe_done=yes`, then the run sat until timeout).
   - **pip `dependencies` fail** (run 219665188914633 with just `[emoji]`): PyPI egress blocked
     from env build. Blocks gpu-burn(+NVML), xgboost, tabicl, vllm, lora.
   - **Catalog bucket 403 from serverless SQL** (ask #1) — same hardening theme.
   Fix is network-side (bucket policies / egress rules must allowlist the serverless data
   plane); nothing is user-fixable. Until then: use **MLflow params/metrics as the receipt
   channel** (tracking API is healthy) and wrap any artifact/storage call in a
   `signal.alarm` timeout so runs fail fast instead of hanging to TIMEDOUT.
   **Plane differential (driver run 791366682924044, identical code CPU vs GPU_1xA10):**
   - CPU serverless: `mlflow.log_artifact` → **OK in 0.4s**; root-storage TCP 443 ✅.
   - GPU node: root-storage **TCP 443 connects ✅ but the artifact upload times out (>60s)**
     — the block is not a plain connection deny; it behaves like stateful egress/proxy
     filtering that stalls data transfer from the GPU plane. Same pattern explains AIR
     launcher log-shipping hanging (log blackout + post-success TIMEDOUTs).
   - `pypi.org`: DNS `Temporary failure in name resolution` on BOTH planes → the pip blocker
     is an egress/DNS allowlist.
   Point the fix at: GPU-plane egress path to workspace root storage (proxy/firewall rules,
   not just bucket policy), plus DNS/egress allowlist for PyPI on both planes.
   **Self-contained repro: `/Workspace/Shared/databricks-air-lab/REPRO-GPU-EGRESS`** — attach
   Serverless GPU (A10, AI v4), Run-all, ~2 min. Verified verdicts: A10 →
   `upload=REPRODUCED after 60s` (run 845924716536114); plain serverless →
   `upload=OK in 10.3s` (run 786560643819370).

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

## Dry-run log (2026-07-24, all GPU_1xA10 unless noted)

| Run id | Workload | Outcome |
|---|---|---|
| 781012703677665 | exec-probe | ❌ INTERNAL_ERROR (generic, no logs) |
| 903444851308928 | exec-probe (resubmit) | ✅ SUCCESS |
| 663513851651434, 307947498581134 | exec-probe via persistent job | ❌ ×2 → wrapper approach dead, job deleted |
| 180684605548384 | serverless CPU notebook | ✅ SUCCESS |
| 30365680925847 | gpu-burn (deps: nvidia-ml-py) | ❌ INTERNAL_ERROR ~5 min |
| 966598736601417 | xgboost-gpu (deps: xgboost, sklearn) | ❌ INTERNAL_ERROR ~5 min |
| 164122580926717 | tabicl-bench (deps: tabicl) | ❌ INTERNAL_ERROR ~5 min |
| 128835177125736 | multinode-probe, 2×A10 | ✅ SUCCESS — 2-node coordination verified |
| 1073399507177055 | gpu-burn variant, deps [] | ❌ INTERNAL_ERROR ~3.5 min — kills the "pip deps alone" hypothesis |

Differential probes (all 1×A10, 2026-07-24):

| Run id | Probe | Outcome → meaning |
|---|---|---|
| 442589496243388 | `exit 1` deliberately | ❌ same generic INTERNAL_ERROR → **user-code failure and platform failure are indistinguishable** in run state |
| 983297781679761 | `python -c` torch CUDA check, no deps | ✅ → python entrypoints fine |
| 219665188914633 | deps `[emoji]` + import | ❌ → **pip deps broken** (ask #4) |

Reinterpretation: the "failed" workloads were (at least partly) real crashes hidden by the
log blackout — pip-deps workloads die on dependency install; gpu-burn-nodeps still under
diagnosis via a self-logging wrapper (stdout → MLflow artifact via MLFLOW_RUN_ID, `set +e`
because the launcher aborts the command block on first non-zero line).

## Verified facts about this target so far

- AIR enabled; Docker image registration works (`forrestm/air-otel-smoke:0.3`,
  sha256:7f48a7fc8…, 2026-07-24) — DCS replication functional.
- Zerobus edge answers for the workspace id (reason-phrase `Malformed token` on bogus token).
- Serverless SQL works against `system` + `samples`; the target catalog's own bucket is the
  only storage failing (isolation receipts in `experiments/docker-otel-zerobus/NOTES.md`).
- Prior AIR usage exists (another user's `gpu-1xa10 ssh-tunnel` secret scopes) — GPU capacity
  has been exercised here before.
