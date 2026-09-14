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

Pending.
