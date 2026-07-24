# Debug a failed run

Goal: map the symptom you're seeing to the cause we already hit, before burning an afternoon.
Every row below came from a real failure in this lab.

## Symptom → actual cause

| Symptom | Actual cause | Check / fix |
|---|---|---|
| Killed, exit 137, **no Python traceback** | **Host-RAM OOM** (OOM-killer SIGKILL), not GPU OOM — small nodes have small CPU RAM | node memory in MLflow system metrics; smaller batches or a bigger node — [Pick your compute](../getting-started/pick-your-compute.md) |
| CUDA OOM traceback | actual GPU OOM | smaller batch / activation checkpointing / bigger card |
| GPU util reads 0% but the job is working | sampling artifact on sub-second steps | trust wall-time + GPU memory, not the gauge |
| Telemetry "export OK" but zero rows in Delta | Zerobus edge hides auth errors behind `grpc-status: 0` | [Ship telemetry to Delta](ship-telemetry-to-delta.md) — probe auth, count rows |
| Multi-node run "hangs" with no output | you're watching node 0 only | `air logs <run-id> --node N` for every node |
| All nodes log the same hostname | expected — hostnames are identical across nodes | distinguish by `NODE_RANK`, never hostname |
| `env_variables`: "Unknown field" | it's top-level, not under `environment:` (docs lag) | [Submit a workload](submit-a-workload.md) |
| Validation fails on `usage_policy_name` | policy doesn't exist in the target workspace | omit the field, or create the policy first |
| Can't find your run in the Jobs UI | `air run` creates `SUBMIT_RUN`s | look in the **Job runs** tab, never "Jobs" |
| Wrong-workspace PERMISSION_DENIED | ambient `DEFAULT` profile pointed elsewhere | pass `-p <profile>` explicitly; check which host errored |
| Docker build dies on Apple Silicon | `uv` segfaults under qemu | vendor wheels on the host — [Custom Docker](use-a-custom-docker-image.md) |

## The 5-minute triage

1. `air logs <run-id> --download-to /tmp/run-logs` — get everything, all nodes.
2. Read **`databricks-launcher.log`** first: lifecycle markers, secret-resolution lines, and the
   `SGC_DEBUG_INFO` block with every correlation ID
   ([details](stream-logs-and-artifacts.md#the-launcher-log-is-gold)).
3. Check MLflow system metrics for the death signature: host memory climbing to the ceiling
   (exit 137) vs. GPU memory spike (CUDA OOM) vs. nothing at all (never started — read the
   launcher log's gate checks).
4. Escalating? Attach the `SGC_DEBUG_INFO` IDs — it's exactly what the on-call needs.

## Two habits that prevent the debugging

- **`--dry-run` every submission.** It does full validation including workspace API calls.
- **Make probes assert, not observe.** A probe that prints a sentinel only after its assertions
  pass ("exit 0" proves nothing) tells you *what* broke, not just *that* something did — see
  [how to read the receipts](../about-receipts.md).

## Attribute failures carefully

The platform is Preview/Beta, but it isn't always the platform: upstream research code, your YAML,
and docs that may trail current behavior are all live suspects. A failure only counts as a
platform issue once you've reproduced it with a minimal probe that rules the others out.
