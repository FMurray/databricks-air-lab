# Native `ai_runtime_task` multi-node probe

## Question and pre-registered pass criteria

Can an AIR workload submitted as the native Jobs `ai_runtime_task` run DDP training across two
nodes? This is an exploratory capability check, not a change to the repo's CLI-only recommendation:
the submission still goes through the `air` CLI.

Target: `fevm-forrest-2`, two `GPU_1xA10` nodes, environment v5, no retries, 15-minute timeout.

A pass requires all of the following:

1. The submitted Jobs run metadata contains `tasks[0].ai_runtime_task`, with
   `accelerator_type=GPU_1xA10` and `accelerator_count=2`.
2. The run terminates with `result_state=SUCCESS`.
3. Logs from both node 0 and node 1 show distinct node ranks and `NUM_NODES=2`, `WORLD_SIZE=2`.
4. The DDP probe completes a forward/backward/optimizer step, asserts identical parameter checksums
   across ranks, sees node ranks `0,1`, and prints `AI_RUNTIME_MULTINODE_TRAINING_OK`.
5. The linked MLflow run records `probe_sentinel=AI_RUNTIME_MULTINODE_TRAINING_OK`,
   `world_size=2`, `num_nodes=2`, and `nodes_seen=0,1`.

| Claim | Required evidence |
|---|---|
| Jobs uses the requested native task type and shape | quoted `jobs get-run` task JSON |
| The task schedules and runs on two nodes | terminal state plus quoted topology from both node logs |
| Cross-node DDP training completes correctly | pass-gated sentinel, checksum assertion, and MLflow params |

## Pre-flight

Local static pre-flight passed on 2026-09-14:

```text
$ bash -n experiments/ai-runtime-task-multinode/run_probe.sh
[exit 0]
$ python3 -c '... ast.parse(...) ...'
PYTHON_PARSE_OK
```

Target-workspace pre-flight passed on 2026-09-14 with AIR CLI v1.1.0,
`fevm-forrest-2`:

```text
[INFO] dry-run: prepared Jobs API payload:
"ai_runtime_task": {
  "deployments": [{
    "compute": {
      "accelerator_type": "GPU_1xA10",
      "accelerator_count": 2
    }
  }]
}
{"data": {"status": "DRY_RUN_OK", "dry_run": true}}
```

## Observed

✅ **VERIFIED 2026-09-14, Jobs run 1090461481669184, `fevm-forrest-2`.** The run
terminated `SUCCESS` after 112 seconds. Workspace MLflow run:
[`5869814e2fc448d0b99836655093d0b8`](https://fevm-forrest-serverless-stable-2.cloud.databricks.com/ml/experiments/1841843443546104/runs/5869814e2fc448d0b99836655093d0b8?o=7474645252241925).

The raw Jobs response proves both the native task type and requested shape:

```json
{
  "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
  "tasks": [{
    "ai_runtime_task": {
      "deployments": [{
        "compute": {
          "accelerator_count": 2,
          "accelerator_type": "GPU_1xA10"
        }
      }]
    }
  }]
}
```

Independent logs from each node prove the topology:

```text
# node 0
TOPOLOGY NUM_NODES=2 NODE_RANK=0 POD_RANK=0 LOCAL_WORLD_SIZE=1 WORLD_SIZE=2
main:513:534 [0] NCCL INFO comm ... rank 0 nRanks 2 nNodes 2 localRanks 1 localRank 0 MNNVL 0
RANK_RECEIPT rank=0 node_rank=0 local_rank=0 world_size=2 loss=0.89160442
AI_RUNTIME_MULTINODE_TRAINING_OK world_size=2 nodes_seen=[0, 1] parameter_checksum=0.971025196835

# node 1
TOPOLOGY NUM_NODES=2 NODE_RANK=1 POD_RANK=1 LOCAL_WORLD_SIZE=1 WORLD_SIZE=2
main:596:616 [0] NCCL INFO comm ... rank 1 nRanks 2 nNodes 2 localRanks 1 localRank 0 MNNVL 0
RANK_RECEIPT rank=1 node_rank=1 local_rank=0 world_size=2 loss=0.95566493
```

The linked MLflow run contains the durable assertion receipt:

```text
accelerator_type=GPU_1xA10
nodes_seen=0,1
num_nodes=2
probe_sentinel=AI_RUNTIME_MULTINODE_TRAINING_OK
task_type_under_test=ai_runtime_task
torch_version=2.9.0+cu129
world_size=2
parameter_checksum=0.9710251968353987
```

| Claim | Evidence |
|---|---|
| Jobs uses the native task type and requested shape | raw task JSON contains `ai_runtime_task`, `GPU_1xA10`, and `accelerator_count: 2` |
| The task scheduled and ran on two nodes | terminal `SUCCESS`; node 0 and node 1 each report `NUM_NODES=2`, distinct node ranks, and NCCL `nRanks 2 nNodes 2` |
| Cross-node DDP training completed correctly | assertion-gated sentinel plus identical post-step parameter checksum; matching MLflow params `nodes_seen=0,1`, `world_size=2` |

This run is **not** a network-performance result. Measured logs show the A10 nodes could not use
EFA (`NET/OFI No eligible providers were found`) and NCCL fell back to `NET/Socket`; the DDP
correctness and native-task result still passed.

The required MLflow experiment description is set with workspace repro links and observed run IDs.
Local archiving is deferred because `experiments/mlflow.db` already contains unrelated uncommitted
work in this checkout; it was left untouched rather than mixing receipts.
