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
| M1 | Sustained fabric soak: 1GB all-reduces, 10 min, drift tracking | `workloads/rdma-m1-soak.example.yaml` (2×8xH100) → `nccl_allreduce_ctypes.py` STRESS_SECONDS/BUF_MB | 🧪 staged |
| M2a | RDMA isolated from NVLink: 1 GPU/node communicator | `rdma-m2a-fabric-only.example.yaml` (4 nodes), FABRIC_ONLY=1 | 🧪 staged |
| M2b | Directed p2p send/recv ring across nodes | `rdma-m2b-p2p-ring.example.yaml`, P2P_RING=1 (busbw = per-rank algbw, no 2(n−1)/n) | 🧪 staged |
| M3 | Prove-it's-RDMA counters: /sys/class/infiniband hw_counters deltas | embedded in every stress run (`rdma_counters()`); ⚠️ A10 container exposes NONE (survey 96419244890099 — A10 has no EFA); H100 exposure = open question, first H100 stress run answers it | 🧪 partial |
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
