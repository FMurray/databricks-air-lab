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
| 6 | Multinode at scale | `multinode-probe` ✅, `multinode-correctness`, `fsdp-multinode` | 2×(8×H100) | ✅ probe on reserved pool: run 968264353316767 (2026-07-24) — 16 ranks/2 nodes, busbw ~332 GB/s, submit→SUCCESS 2 min; correctness/FSDP next |
| 7 | Training Hub app | `apps/training-hub/` | Apps | ⛔ blocked: Apps disabled for the org |
| 8 | Billing/visibility SQL | `utils/billing/`, `utils/visibility/` | warehouse | ✅ runnable (system tables readable); telemetry joins wait on #2 |
| 9 | Node acceptance: burn + health | `gpu-burn.example.yaml` | 1×A10 dry → 8×H100/node | ✅ **A1 COMPLETE 2026-07-25: 20/20 pool nodes PASS** — 160 distinct GPU UUIDs, 0 ECC / 0 throttle, 641–774 TFLOPS/GPU; peak 19 concurrent nodes; receipts + allocation map in `experiments/node-acceptance/NOTES.md` |
| 9b | RDMA / fabric stress (5 methods) | `rdma-m1-soak`…`rdma-m5-parambench` + `experiments/rdma-stress/` | 2–16×(8×H100) | 🧪 staged 2026-07-25; M3 counter exposure on H100 + 16-node soak pending (coordinate before firing) |
| 10 | Node acceptance: all-reduce bench | `nccl-allreduce.example.yaml` | 2×A10 dry → 8/16×H100 | 🆕 built 2026-07-24; dry-run gated on #1 |
| 11 | Env flexibility: vLLM in std env | `vllm-smoke.example.yaml` | 1×A10 dry → 1×H100 | 🆕 built 2026-07-24; needs HF egress (probe reports) |
| 12 | Classic ML: XGBoost GPU (hang repro) | `xgboost-gpu.example.yaml` | 1×A10 control → 1×H100 | 🆕 built 2026-07-24 |
| 13 | FM: LoRA fine-tune | `lora-finetune.example.yaml` | 1×A10 dry → 8×H100 | 🆕 built 2026-07-24; needs HF egress |
| 14 | Deps: vendored (snapshot) | `vendored-wheels-snapshot.example.yaml` | 1×A10 | ✅ run 846540776169482 — emoji+xxhash (compiled) via committed vendor/ + PYTHONPATH, verdict PASS |
| 15 | Deps: vendored (UC volume) | `vendored-wheels-ucvolume.example.yaml` | 1×A10 | ⛔ staged; gated on catalog bucket fix (#1) |
| 16 | Deps: workspace default package repo (documented GA path) | admin setting, `probes/pip-probe.yaml` re-test | — | 📋 needs workspace admin: scope `databricks-package-management` → internal index; verify AIR env build inherits |
| 17 | air CLI from a notebook (zero local setup) | `uat/checks/air-cli-from-notebook` + vendored CLI wheels | CPU (submits 1×A10) | ✅ 2026-07-30 — all probes pass via DRIVER (check run 11760699227540); notebook-submitted envvar-probe → SUCCESS (AIR run 677147480932865) |

### #17 success criteria (written before first run, 2026-07-30)

Question: can a tester drive the AIR CLI entirely from a serverless notebook (no local machine)?
Check = `uat/checks/air-cli-from-notebook` via the DRIVER (CPU shape, `air_mode=submit`). The
CLI installs from `utils/verification/uat-notebooks/wheels/` (vendored, **databricks-air 1.0.0**
— NB: 1.0.0 released ~7/30, all prior findings are v0.1.x-scoped); auth = notebook context token
via `DATABRICKS_HOST`/`DATABRICKS_TOKEN` (local pre-flight of that auth mode: ✅ both 0.1.0 and
1.0.0 dry-run, 2026-07-30). PASS requires all four probes ✅ in the exit JSON:

- `cli_install` — `air --version` exits 0 from the %pip-installed console script
- `context_auth_env` — context token exported (trivially ✅; real auth proof is the submit)
- `air_dry_run` — exit 0 + dry-run sentinel, run from the FUSE-mirrored repo
  (NB 1.0.0 dry-run skips upload AND submission — it proves config plumbing only, hence:)
- `air_submit` + `air_submitted_run` — real `air run` of `probes/envvar-probe.yaml` from
  `/Workspace/Shared/databricks-air-lab` (snapshot upload from FUSE = the genuinely novel leg),
  run_id parsed, polled to `SUCCESS` within 15 min

Expected failure modes worth distinguishing: %pip resolve failure (wheel set incomplete →
vendor gap), `air_submit` auth error (context token not honored by CLI → needs PAT/secret
path), snapshot upload error from FUSE (CLI can't tar workspace files → CLI-from-notebook dead
without a git clone step).

**✅ VERIFIED 2026-07-30, fe-sandbox-mkazia-lw2 — all criteria met on first DRIVER run**
(driver job 505966573941984 → check run 11760699227540, `air_mode=submit`):

```
cli_install        ✅ .../pythonEnv-.../bin/air -> v1.0.0
context_auth_env   ✅ host=https://fe-sandbox-mkazia-lw2.cloud.databricks.com
air_dry_run        ✅ exit=0
air_submit         ✅ exit=0; run_id=677147480932865
air_submitted_run  ✅ run 677147480932865 -> SUCCESS
```

The notebook-submitted workload (`probes/envvar-probe.yaml`, 1×A10, snapshot uploaded from the
FUSE mirror) ran to SUCCESS; archived locally as MLflow run `f6563955aead41379ccbb08f7435c231`
(source `fd9313843f5b4055b1548e0713296188`). Follow-up run with a pre-%pip probe (job
685034087490453) answered the bundling question: **the CLI is NOT preinstalled in the
serverless runtime** — `cli_not_preinstalled ✅ pre-%pip: which(air)=None module=False
py3.12.3` — vendored wheels (or an admin-set default package repo) are required. The check now
asserts this every run, so it flags if a future runtime starts bundling the CLI.

⚠️ Anomaly from the same DRIVER run, needs follow-up: `gpu-smoke@GPU_1xA10` (job run
975022265079249) reported `gpu_available=⏭️ skipped_no_gpu` — the same Jobs API
`hardware_accelerator` path returned a healthy A10 on 2026-07-24 (run 832802492734599). Not
re-tested; could be env v5 notebook-job GPU attach, pool contention, or check logic — do not
conclude GPU notebook jobs are broken from one skip, but re-run before the window.

## Admin prerequisites on the target (owner asks)

1. **Catalog bucket policy**: all table I/O in the target catalog 403s from serverless with
   UC-vended creds (pre-existing tables too; `system`/`samples` fine from the same warehouse).
   Likely a VPC-endpoint-scoped bucket policy missing the serverless (NCC) endpoints. Blocks #2
   tables and any Delta output; risk it also blocks Zerobus's server-side writer — retest after fix.
2. **Service principal**: SCIM create is admin-only; SPs from other workspaces are
   `invalid_client` here. Blocks #2 auth. Secret scope `air_lab` is ready to receive creds.
3. **Enable Databricks Apps** for the org. Blocks #7.
4. **RESOLVED (mitigation) / CAVEAT — AIR env v4 breaks job-submitted GPU egress; pin v5.**
   Final diagnosis 2026-07-25 after full A/B isolation: GPU runs submitted as **jobs**
   (notebook jobs AND AIR CLI Gen-AI tasks) cannot upload to the workspace root-storage
   bucket when the environment is **version 4**. **Version 5 works.** Interactive GPU
   notebooks are unaffected either way. Trigger method (run-now vs runs/submit) is NOT a
   factor (isolated: run-now + v4 fails, runs/submit + v5 works).

   | GPU run | env v4 | env v5 |
   |---|---|---|
   | Interactive notebook | ✅ 0.8s | ✅ |
   | Notebook job (`hardware_accelerator`) | ❌ 60s stall ×5 runs | ✅ 11.5s (run 733701251559072) |
   | AIR CLI Gen-AI task (multinode's path) | ❌ stall (run 683173786603437) | ✅ 0.4s + **`air logs` streams content — first time on this ws** (run 20867331866373) |

   **Mitigation applied:** every repo workload YAML + notebook env spec pins
   `version: "5"` / `environment_version: "5"`. A/B repro:
   `/Workspace/Shared/databricks-air-lab/REPRO-GPU-EGRESS` cell 6 (submits v4+v5 children,
   prints both verdicts) or CLI: `air run --file workloads/probes/cli-egress-probe.example.yaml
   -p <profile>` (healthy v5 default; `--override environment.version=4` reproduces).

   **Still report to eng/oncall**: (a) v4's job-plane egress is broken here while v5's works
   — v4 is a currently-valid env version and other tenants will hit this blind (no logs =
   undebuggable); (b) **timeout enforcement inconsistency**: one hung v4 run sailed ~6h past
   `timeout_minutes: 12` before manual cancel (run 683173786603437, billing hazard), others
   enforced correctly. Note: most "post-success hang" observations were OUR probes' missing
   `mlflow.end_run()` (metrics-monitor thread keeps python alive) — fixed in
   `workloads/probes/`; the 6h timeout-defeat stands regardless.

   **Related constraints (unchanged):**
   - PyPI unreachable from all planes (DNS) — **by design** (customer-realistic no-PyPI
     posture). Vendor wheels via `code_source` snapshot or UC volume (UAT track for both
     variants pending); `dependencies: []` until then. Affects gpu-burn(+NVML), xgboost,
     tabicl, vllm, lora.
   - Receipt discipline: MLflow params/metrics are the durable channel;
     `signal.alarm` around storage calls; `mlflow.end_run()` always.

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
