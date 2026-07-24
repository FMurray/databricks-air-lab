# Node acceptance — burn + interconnect bench

Phase-1 acceptance tests for a reserved GPU pool (UAT plan A1/A2): prove every node's GPUs
compute at rate without ECC errors or thermal throttling, and that intra-/inter-node
collectives hit expected bandwidth. GA surface only — standard env + pip, no Docker.

| Script | Workload | What it proves |
|---|---|---|
| `burn.py` | `workloads/gpu-burn.example.yaml` | all GPUs enumerate (NVML), sustained matmul TFLOPS, zero uncorrected ECC, no HW-slowdown throttle |
| `allreduce_bench.py` | `workloads/nccl-allreduce.example.yaml` | all-reduce correctness + bus bandwidth vs size (NVLink intra-node; fabric inter-node) |

Both print `RESULT ...` receipt lines (measured values, not characterizations) and exit
non-zero on assertion failure — a green job run IS the acceptance record.

Pool sweep pattern (acceptance day): submit gpu-burn once per node in parallel
(`for i in $(seq 20); do air run --file workloads/gpu-burn.yaml -p <profile> & done`) —
each 1-node run lands on a distinct node while the pool has capacity.
