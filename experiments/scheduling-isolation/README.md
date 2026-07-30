# Scheduling & isolation UAT

Two customer questions this family answers with receipts (experiment
`air-lab-scheduling-isolation`):

## 1. Co-tenancy / isolation — "if user A and user B each run a workload, can they land on the same node?"

Submit ≥2 `workloads/placement-probe.example.yaml` concurrently (HOLD_SECONDS keeps them
overlapped; `hold_start/end_utc` params prove the overlap). Compare params across runs:

| Signal | Reading |
|---|---|
| same `boot_id` | co-located on one node/VM — **boot_id is the ONLY reliable node discriminator**: `machine_id` and `hostname` are baked into the container image (verified identical across different boots, runs 884203635921670/1032780469039538) |
| disjoint `gpu_uuids`, same `boot_id` | GPU-granular bin-packing on a shared node |
| `gpu_count_visible` > requested GPUs | workload sees other tenants' GPUs — isolation gap |
| `other_gpu_procs` > 0 at start | foreign compute processes visible via NVML |

**Dry-run receipt (2026-07-30, same-user 2×A10, runs 884203635921670 + 1032780469039538,
holds overlapped 14:09–14:12Z):** different `boot_id` (separate nodes), each saw exactly its
own 1 GPU (distinct UUIDs), 0 foreign GPU procs, 16-process PID namespace. Clean isolation
for this pair — but A10 on-demand placement ≠ reserved-pool bin-packing; the H100-pool and
cross-user variants are the ones that answer the customer question.

Same-user pairs run today; the **cross-user variant needs a teammate** to submit the second
probe (identical command) — permissions and isolation may differ by principal.

## 2. Reserved-pool scheduling — single-node submits vs the pool

- **Pool free + 1×H100 submit** (`--override compute.accelerator_type=GPU_1xH100`): does a
  1-GPU workload claim a whole 8-GPU reserved node? Measure by submitting K probes with
  long HOLDs: if max concurrent RUNNING == node count, claim granularity = whole node; if
  ≈ 8× nodes, GPU-granular. (gpu_count_visible per run cross-checks.)
- **Pool full + one more submit** (plan item P2): occupy all nodes with held probes
  (`--override compute.accelerator_type=GPU_8xH100 env_variables.HOLD_SECONDS=900`, one per
  node), then submit one more. Expected: PENDING/queued or a **legible capacity error** —
  NOT silent on-demand spillover or cross-region fallback (compliance-relevant). Record
  submit→state timeline + any state_message verbatim.

Cost discipline: dry-run everything on A10 pairs first; the pool-scale versions are
window-day runs — announce before occupying the pool.
