# RDMA / fabric stress — lab notebook

Five methods for stressing and *proving* the inter-node RDMA path on the reserved pool,
all runnable under this workspace's constraints (no PyPI egress; env v5 standard; CUDA
12.9 + libnccl 2.29.7 on the image). MLflow experiment: `air-lab-rdma-stress`.

Prior receipts this builds on (2026-07-25): ctypes-NCCL probe verified end-to-end on
2×A10 v5 (run 683501653815678, `MULTINODE_NCCL_V5_OK`, correctness-asserted); RDMA
GDRDMA/EFA path previously confirmed on the torch path (run 505819227973807,
e2-demo-field-eng: aws-ofi-nccl 1.15.0, efa-direct, 32 NICs/node).

**torch-on-v5 correction (2026-07-25):** the v5 *default* interpreter lacks torch, but the
image carries the full AI env at `/opt/databricks-environments/databricks-ai` (survey run
96419244890099 found `…/site-packages/torch`). Use that interpreter (M5 does) or
PYTHONPATH it. Earlier "no torch on v5" notes are scoped to the default interpreter.
Verification of the AI-env interpreter is owned by a parallel workstream (duplicate probe
364037812147073 cancelled).

## Method matrix

| # | Method | Asset | Status |
|---|---|---|---|
| M1 | Sustained fabric soak: 1GB all-reduces, 10 min, drift tracking | `workloads/rdma-m1-soak.example.yaml` (2×8xH100) → `nccl_allreduce_ctypes.py` STRESS_SECONDS/BUF_MB | ⚠️ 60s native-task smoke passed; full 10 min still staged |
| M2a | RDMA isolated from NVLink: 1 GPU/node communicator | `rdma-m2a-fabric-only.example.yaml` (4 nodes), FABRIC_ONLY=1 | 🧪 staged |
| M2b | Directed p2p send/recv ring across nodes | `rdma-m2b-p2p-ring.example.yaml`, P2P_RING=1 (busbw = per-rank algbw, no 2(n−1)/n) | 🧪 staged |
| M3 | Prove-it's-RDMA counters: /sys/class/infiniband hw_counters deltas | embedded in every stress run (`rdma_counters()`); A10 and H100 containers exposed no counters; use explicit NCCL OFI/EFA/GDRDMA logs as the path receipt | ❌ counters unavailable on H100 run 970758228903276 |
| M4 | Defensible benchmark: nccl-tests built from vendored source with image toolchain | `rdma-m4-nccl-tests.example.yaml` + `build_and_run_nccl_tests.sh` + `nccl-tests-src/` (v2.13.13, BSD) | 🧪 staged; single-node only until MPI exists on image (script reports mpirun) |
| M5 | parambench-train-comms (in image package list; needs torch → databricks-ai env) | `rdma-m5-parambench.example.yaml` + `parambench_probe.py` | 🧪 staged, probe-grade |

## Pre-registered success criteria (written before the first H100 stress run)

- M1: run SUCCESS + `MULTINODE_NCCL_V5_OK`; sustained busbw within 30% of the smoke busbw
  for the full soak; `stress_window_drift_pct` < 20% (larger = thermal/fabric instability
  → investigate, that's the point of the test).
- M2a/M2b: same sentinel; busbw interpretation label: fabric-only numbers are per-EFA-path,
  NOT comparable to NVLink-diluted full-world busbw.
- M3: if H100 containers expose hw_counters → byte-ish deltas ≈ expected traffic volume
  (order-of-magnitude check) = RDMA-path proof; if not exposed → documented limitation,
  fall back to inference from bandwidth (TCP can't hit these rates).
- M4: build completes with image nvcc; all_reduce_perf 8-GPU sweep prints; numbers are the
  ones allowed on customer decks (label: intra-node/NVLink until MPI).
- M5: module imports under the AI env + CLI runs → upgrade to real config; import failure
  is a finding (package listed but broken).

All numbers from M1/M2 remain **smoke-grade** per the verification skill; M4 output is the
defensible tier.

## DAB-native M1 smoke — pre-registered 2026-09-15

Question: can the checked-in `bundles/ai-runtime-recipes` DAB package and execute the existing M1
recipe as a native Jobs `ai_runtime_task` across two 8xH100 nodes?

Target and bounded shape: `fevm-forrest-2`, Databricks CLI v1.16.0, environment v5,
2×`GPU_8xH100` (`accelerator_count=16`), `STRESS_SECONDS=60`, 1 GiB buffers, no retries, and a
15-minute task timeout. Evidence lands in the existing `air-lab-rdma-stress` MLflow experiment.

A pass requires all of the following:

1. The deployed Jobs resource contains native `ai_runtime_task`, the uploaded `.tgz` code path,
   the uploaded M1 smoke command, `GPU_8xH100`, and `accelerator_count=16`.
2. The bundle-started run terminates `SUCCESS`.
3. Both node logs report distinct node ranks, `NUM_NODES=2`, `LOCAL_WORLD_SIZE=8`, and world size 16.
4. Both nodes print `CORRECTNESS_OK all elements == 16`; node 0 prints the pass-gated
   `MULTINODE_NCCL_V5_OK` sentinel.
5. Logs explicitly select RDMA/EFA rather than socket fallback. Hardware-counter lists may remain
   empty; the prior native-task run established that those counters are not container-visible here.

| Claim | Required evidence |
|---|---|
| DAB persisted the requested native task and shape | raw Jobs resource JSON |
| The bundle run executed correctly on two nodes | terminal state plus independent node logs |
| NCCL used the RDMA path | explicit NCCL transport/provider/channel lines from both nodes |

### Attempt 1 — code-source root mismatch

❌ **FAILED 2026-09-15, Jobs run 288681717574238, `fevm-forrest-2`.** Workspace MLflow run
`3c880ed4a51a40f69173570df7f439ba`. The DAB-created Jobs resource and run metadata contained the
native task, uploaded artifact/command paths, `GPU_8xH100`, and `accelerator_count=16`, but both
nodes exited before starting NCCL:

```text
Code source experiments available at /databricks/code_source/experiments
python: can't open file '/databricks/code_source/experiments/experiments/node-acceptance/nccl_allreduce_ctypes.py': [Errno 2] No such file or directory
ERROR: Script failed with exit code 2 after 2s
```

The `tgz` includes only paths below `experiments/`, so AI Runtime chose that common directory as
`CODE_SOURCE_PATH`. The command adapters incorrectly appended another `experiments/`. They now
resolve `node-acceptance/` and `rdma-stress/` directly below `CODE_SOURCE_PATH`; the retry keeps the
same pre-registered shape and pass criteria.

## Native `ai_runtime_task` M1 smoke — pre-registered 2026-09-15

Question: can the native Jobs `ai_runtime_task` path that passed two-node A10 DDP also run the
existing M1 NCCL/RDMA recipe on two 8xH100 nodes?

Submission target and bounded shape: `fevm-forrest-2`, AIR CLI v1.1.0, environment v5,
2×`GPU_8xH100` (16 GPUs), M1 with `STRESS_SECONDS=60`, `BUF_MB=1024`, no retries, and a
20-minute timeout. This is a shortened plumbing/RDMA smoke, **not** the recipe's full 10-minute
stability acceptance run.

A pass requires all of the following:

1. Dry-run and submitted Jobs metadata contain native `ai_runtime_task` with
   `accelerator_type=GPU_8xH100` and `accelerator_count=16`.
2. The run terminates `SUCCESS`; both node logs report distinct `NODE_RANK` values,
   `NUM_NODES=2`, `LOCAL_WORLD_SIZE=8`, and a 16-rank NCCL communicator.
3. Both nodes print `CORRECTNESS_OK all elements == 16`, and rank 0 prints the pass-gated
   `MULTINODE_NCCL_V5_OK` sentinel.
4. RDMA evidence is explicit: NCCL reports the AWS OFI/EFA transport, and the embedded M3
   hardware-counter probe prints positive byte/data counter deltas. If counters are unavailable,
   that criterion fails and the exact fallback evidence is reported rather than promoted to RDMA.
5. MLflow records the sentinel/topology parameters plus smoke/stress metrics.

| Claim | Required evidence |
|---|---|
| Native task has the requested H100 multi-node shape | quoted raw Jobs task JSON |
| All 16 ranks completed correct collectives | quoted output from node 0 and node 1 plus sentinel |
| Traffic used RDMA rather than socket fallback | quoted OFI/EFA selection and positive M3 counters |

Local static pre-flight:

```text
$ python3 -c '... ast.parse(nccl_allreduce_ctypes.py) ...'
PYTHON_PARSE_OK
```

Target-workspace dry-run passed on 2026-09-15:

```text
"ai_runtime_task": {
  "deployments": [{
    "compute": {
      "accelerator_type": "GPU_8xH100",
      "accelerator_count": 16
    }
  }]
}
"variables": {"STRESS_SECONDS": "60", "BUF_MB": "1024"}
{"data": {"status": "DRY_RUN_OK", "dry_run": true}}
```

### Observed

⚠️ **PARTIAL 2026-09-15, Jobs run 970758228903276, `fevm-forrest-2`.** The native
task/RDMA smoke passed; the separately pre-registered M3 hardware-counter criterion failed. The
run terminated `SUCCESS` after 142 seconds. Workspace MLflow run:
[`3d659da31a484414bbf45472463ee7e9`](https://fevm-forrest-serverless-stable-2.cloud.databricks.com/ml/experiments/3516368535099647/runs/3d659da31a484414bbf45472463ee7e9?o=7474645252241925).

The raw Jobs response proves the native task type and shape:

```json
{
  "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
  "tasks": [{
    "ai_runtime_task": {
      "deployments": [{
        "compute": {
          "accelerator_count": 16,
          "accelerator_type": "GPU_8xH100"
        }
      }]
    }
  }]
}
```

Independent node receipts:

```text
# node 0
NODE 0/2 local=8 world=16 host=main.host.local uuids=914b82e39ea6,...,aebd03301954
NCCL INFO NET/OFI Using transport protocol RDMA (platform set)
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
NODE 0 CORRECTNESS_OK all elements == 16
NODE 0 all_reduce 1024MB x10: 5.3 ms/iter, algbw 204.1 GB/s, busbw ~382.8 GB/s
NODE 0 STRESS 60s buf=1024MB fabric_only=False p2p_ring=False 12980 iters, sustained busbw ~435.4 GB/s, window drift 16.3% (min 5ms/10it max 5ms/10it)
NODE 0 RDMA counter deltas (raw/1e9): []
MULTINODE_NCCL_V5_OK

# node 1
NODE 1/2 local=8 world=16 host=main.host.local uuids=331d3f090666,...,48afe9bff23d
NCCL INFO NET/OFI Using transport protocol RDMA (platform set)
NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 32 nics)
NODE 1 CORRECTNESS_OK all elements == 16
NODE 1 all_reduce 1024MB x10: 5.2 ms/iter, algbw 204.8 GB/s, busbw ~384.0 GB/s
NODE 1 STRESS 60s buf=1024MB fabric_only=False p2p_ring=False 12980 iters, sustained busbw ~435.5 GB/s, window drift 16.4% (min 5ms/10it max 5ms/10it)
NODE 1 RDMA counter deltas (raw/1e9): []
```

The full logs additionally show inter-node channels explicitly wired as
`NET/Libfabric/<nic>/GDRDMA`. MLflow records:

```text
probe_sentinel=MULTINODE_NCCL_V5_OK
num_nodes=2
local_world_size=8
world_size=16
nccl_version=22907
buf_mb=1024
allreduce_smoke_ms=5.259919166564941
algbw_gbps=204.13656369955606
busbw_gbps=382.7560569366676
stress_iters=12980
stress_seconds=60.01264405250549
stress_sustained_busbw_gbps=435.44476425229254
stress_window_drift_pct=16.27078446567605
```

MLflow contains no `rdma_delta_*` metrics, matching both nodes' empty counter lists. Its artifact
inventory contains `logs/node_0` and `logs/node_1`, so both raw receipts are durable.

| Claim | Verdict | Evidence |
|---|---|---|
| Native task has the requested H100 multi-node shape | PASS | raw Jobs JSON: `ai_runtime_task`, `GPU_8xH100`, `accelerator_count: 16` |
| All 16 ranks completed correct collectives | PASS | both nodes: `local=8 world=16`, `CORRECTNESS_OK`; rank 0: `MULTINODE_NCCL_V5_OK` |
| Traffic used RDMA rather than socket fallback | PASS | both nodes: `Using transport protocol RDMA`, `efa-direct`; channel logs: `GDRDMA` |
| Embedded M3 counters prove byte movement | FAIL | both nodes: `RDMA counter deltas (raw/1e9): []`; no `rdma_delta_*` MLflow metrics |
| Full M1 stability acceptance | NOT RUN | deliberately shortened to 60s; canonical recipe is 600s |

The 435.44 GB/s value is **measured, smoke-grade normalized NCCL bus bandwidth**, not raw EFA
line rate and not customer-deck benchmark data. Window drift was 16.27% over this short run; it is
below the pre-registered 20% threshold but does not replace the full 10-minute acceptance soak.

The MLflow experiment description is set with workspace repro links, exact launch-folder
provenance, pass criteria, and this partial result. Local archiving is deferred because
`experiments/mlflow.db` already contains unrelated uncommitted work in this checkout.
