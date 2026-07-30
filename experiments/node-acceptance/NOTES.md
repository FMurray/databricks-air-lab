# Node acceptance — lab notebook

## A1 GPU-burn sweep, 20-node reserved pool — pre-registered success criteria (written BEFORE submit, 2026-07-24)

Context: UAT plan phase 1 A1 (docs/private/uat-plan-2026-07.md) against the 20 dedicated
H100 nodes on fe-sandbox-mkazia-lw2. Prior state: `burn.py` rewritten single-process after
the mp deadlock (run 603656373172623); earlier A10 attempts today failed with unknown cause
(log blackout — runs 30365680925847, 1073399507177055 predate the receipt instrumentation).

Changes for this workspace (2026-07-24): `burn.py` now writes MLflow receipts (start:
`gpus_visible`, `gpu_uuids`, `nvml`, torch version; end: `burn=PASS|FAIL` + per-GPU
tflops/temp/power/throttle/ECC-delta metrics; FAIL logs `burn_failures` before raising).
`nvidia-ml-py` vendored into `experiments/node-acceptance/vendor/` (PyPI unavailable by
design here); `gpu-burn-nodeps.yaml` is the submission vehicle (`dependencies: []`).
GPU UUIDs are the node-distinctness proof — hostnames are identical across AIR nodes.

Plan: (1) A10 pre-flight, 60 s burn, validates code + receipt path; (2) 20 parallel submits
`--override compute.accelerator_type=GPU_8xH100 env_variables.EXPECT_GPUS=8
env_variables.BURN_SECONDS=900`, timeout 30 min, max_retries 0.

Success = ALL of, per the pre-registered A1 bar (pass: no ECC errors / no HW throttle; all
160 GPUs enumerate):

1. Pre-flight: run SUCCESS + receipt `burn=PASS`, `gpus_visible=1`, `nvml=available`
   (vendored import worked), 1 GPU UUID present.
2. Sweep: 20/20 runs SUCCESS, each receipt `burn=PASS`, `gpus_visible=8`, `nvml=available`.
3. **160 distinct GPU UUIDs** across the 20 receipts (proves 20 distinct physical nodes —
   receipts with overlapping UUIDs mean node reuse, count only distinct nodes as accepted).
4. Every GPU: `ecc_uncorrected_delta` ∈ {0, -1=unsupported} and `hw_throttle_samples=0`
   (these are also assert-gated in burn.py — `burn=PASS` is unreachable otherwise).
5. Per-GPU fp16 matmul TFLOPS recorded (smoke-grade utilization signal, NOT a benchmark;
   H100 fp16 dense peak ~989 TFLOPS w/ sparsity caveats — record, don't gate).

Failure modes: INTERNAL_ERROR + no receipt = crash before enumeration; receipt with
`burn=FAIL` + `burn_failures` = health-check failure on named GPUs (that node fails
acceptance); SUCCESS + `nvml=missing` = vendoring broke (rerun after fix — ECC/throttle
unverified without NVML).

### Sweep attempt 1 — 2026-07-24 (2026-07-25T02:57Z): quota caps concurrency at 4

Pre-flights: run 1021313559310836 SUCCESS but end receipt missing → receipt helper bug
(`mlflow.start_run` resume fails silently on the job plane; switched to `MlflowClient`
client API). Run 609493256255695 ✅ full receipt: `burn=PASS`, `nvml=available` (vendored),
63.2 TFLOPS fp16, max 69°C / 300W, 0 throttle samples, ECC delta 0 (A10G).

20 submits (~7s apart, run ids in sweep log): **first 4 admitted, 16 FAILED immediately**,
verbatim state_message: "System error occurred during execution of task Gen AI Compute
Task: Workspace has exceeded its GPU quota for GPU_8xH100. Please reach out to your
databricks contact to increase the quota."

- **New admin ask: raise workspace GPU_8xH100 quota to match the 20-node reservation**
  (currently admits 4 concurrent nodes = 32 GPUs, measured by admission, not a stated limit).
- **Quota mechanism (internal receipts, 2026-07-24):** per-workspace, per-`gpuType` buckets
  counted in NODES of that shape — server log format from ES-1872498:
  `Workspace quota exceeded: workspaceId=… gpuType=GPU_8xH100 current=4 requested=2 max=4
  totalAfterRequest=6`. Same default `max=4` hit by another customer. A10 runs do NOT
  consume the 8xH100 bucket (separate `gpuType=GPU_1xA10` bucket exists — Confluence triage
  log). Increase path: ES ticket via account team (AIR Field Guide FAQ). Inferred label:
  "max=4 is the stock default" is consistent with two workspaces showing max=4; the AIR
  team owns the actual per-workspace values.
- **P2 (over-capacity behavior) answered early**: fast, explicit quota FAIL — no queueing,
  no silent cross-region fallback. Compliance-relevant good news; but parallel submitters
  must throttle client-side.
- Full 20-node coverage is impossible until the quota is raised; waves of ≤4 may revisit
  the same physical nodes (UUID dedup in receipts will show actual coverage).

### Wave results + the env v4/v5 differential (2026-07-24, 2026-07-25T03:0x–03:2xZ)

**Wave 1 (env v4) ✅ 4/4 PASS** — runs from sweep attempt 1 (MLflow 28a26128, 9d1e5735,
aa7508d3, c4ff6775): every GPU 641–646 fp16 TFLOPS (measured, MATMUL_N=8192 loop — smoke-grade
utilization, not a benchmark), max temp 75–81°C, 0 throttle samples, 0 ECC errors,
**32/32 distinct GPU UUIDs = 4 distinct physical nodes**.

**Waves 2–3 (env v5) ❌ 8/8 FAILED — env v5 has NO torch on the Gen-AI task path.**
`ModuleNotFoundError: No module named 'torch'` (run 366126592157969 log, verbatim; wave 3
runs 1055481184280694/760243512292882/653030219115367/855357070628493 same). The YAMLs had
been switched to v5 to chase the egress fix; reverted to v4.

**By-product finding (single-run evidence each way, needs a dedicated probe):**
- **env v5 Gen-AI GPU task SHIPS LOGS** — full launcher + traceback retrieved via
  `air logs 366126592157969` — and fails with result_state FAILED + task message, not the
  generic INTERNAL_ERROR. On v4 the identical path has the total log blackout and
  INTERNAL_ERROR-for-everything semantics.
- Launcher bootstrap on this workspace: `uv not found in base environment; attempting to
  install it...` → 5× pip retries `Temporary failure in name resolution` (PyPI, by design)
  → `WARNING: could not obtain uv; falling back to pip` → benign with `dependencies: []`
  (setup completed in 10s). Noise, not a failure — but it burns ~30s/run of retries.
- Implication if the v5 log delivery reproduces: the v4 blackout may be env-image-specific,
  not purely network-side — sharpens the eng escalation considerably.

### Env v5 survey + cuBLAS burn backend (2026-07-24, 2026-07-25T03:2x–03:3xZ)

Engagement decision: **v5 everywhere going forward** (owner call, 2026-07-24).

Survey (✅ run 491958602140255, 1×A10, `env5_survey.py`, results read from logs — v5 ships
them; MLflow receipt also OK): python 3.12.3; **no torch**; numpy 2.1.3, mlflow 3.8.1,
nvidia-ml-py native, pandas/pyarrow present (265 dists incl. `databricks.serverless_gpu`,
`parambench-train-comms`); **full CUDA 12.9 toolkit on the image**: libcublas.so.12(.9.1.4),
libcudart.so.12(.9.79), driver 580.126.16, nvidia-smi. Log-delivery finding REPRODUCED
(2 for 2 on v5).

burn.py rewritten with dual load backend: torch (v4) / **ctypes+cublasGemmEx fp16-in
fp32-acc** (v5, no vendoring needed — toolkit is on the image). Receipt schema unchanged
+ new `backend` param. Pre-flight ✅ run 135521403481587 (1×A10, v5): quoted from logs —
`RESULT gpu=0 name='NVIDIA A10G' tflops=67.1 max_temp_c=57 max_power_w=244
hw_throttle_samples=0 ecc_uncorrected_delta=0`, `RESULT burn=PASS` (67.1 TFLOPS cublas vs
63.2 torch-v4 on the same shape class — measured, same MATMUL_N=8192 loop).

Coverage driver v2 (`burn_waves2.py`): 16 node-runs, ≤4 concurrent (quota bucket),
quota-refusals retried after 90s backoff (release-lag observed: run 567319117365393
refused while 3 prior runs were still tearing down); non-quota FAILED counts as a real
node outcome. UUID aggregation at the end gives the allocation map. Superseded by
`coverage_driver.py` (committed): adaptive width probing for cap raises.

### Quota raised 4 → 20 mid-window; full saturation launched (2026-07-25T04:0xZ)

Eng config change (resource-gatekeeper, `max-num-nodes-gpu-8xh100` 4 → 20 for this
workspace; propagates ~15–20 min post-merge). Verified live by admission: adaptive driver
had 5 concurrent admitted at 04:00Z, then 15 more submitted at 04:0x — **19 of 20
admitted; the 20th (job run 830940860440388, 04:04Z) was quota-refused at the propagation
edge** (verbatim: "Workspace has exceeded its GPU quota for GPU_8xH100"), leaving a
one-run receipt of the propagation window itself. Peak measured concurrency: **19×8xH100
nodes simultaneously**. Combined with the earlier max=4 refusals this brackets the quota
behavior end-to-end (P1/P2 evidence).

### ✅ A1 NODE ACCEPTANCE COMPLETE — 20/20 nodes PASS (aggregated 2026-07-30)

Aggregation over every receipted 8-GPU node-run in MLflow experiment `air-lab-gpu-burn`
(all batches 2026-07-24/25: wave 1 v4-torch, v4 stragglers, v5-cublas driver + saturation):

| Claim | Evidence (MLflow receipts, aggregated) |
|---|---|
| All 20 physical nodes covered | **160 distinct GPU UUIDs across 31 receipted node-runs** (8/node × 20 nodes; UUID = hardware identity). Nodes 01–10 visited 2–3× by early ≤4-wide waves (scheduler reuse under the old quota); nodes 11–20 first reached by the saturation batch — the quota raise was NECESSARY for full coverage |
| Every node passes the burn bar | 31/31 receipts `burn=PASS`; **zero HW-throttle samples and zero uncorrected-ECC deltas across all 248 GPU-burn instances** |
| Performance uniform | per-GPU fp16 matmul 641–774 TFLOPS (measured, MATMUL_N=8192, smoke-grade; spread partly backend: torch-v4 n=7 vs cublas-ctypes-v5 n=24) |
| Both env paths validated | v4/torch and v5/ctypes-cuBLAS backends both produce PASS receipts on the same bar |

One anomaly: MLflow runs for admission-refused jobs are created but never terminated —
they sit in status RUNNING with empty params forever (e.g. 165bfe11…, the 04:04 refusal).
Filter on `gpus_visible` receipts, not MLflow status, when aggregating.

**Multinode-at-scale status:** the 16-node/128-GPU ctypes-NCCL stress run (M1+M3 at max
scale) was staged but NOT fired on 07-25 (pool drained after the session ended). It is
one command away (`workloads/rdma-m1-soak.example.yaml` + num_accelerators=128 override)
— coordinate before firing during the shared UAT window.
