# Your first run in 10 minutes

Goal: submit a real (cheap) GPU workload and know where everything landed.

**Prerequisites:** a workspace with serverless GPU enabled, and a `databricks` CLI profile that
points at it (`~/.databrickscfg`).

!!! warning "Real money"
    `air run` submits real GPU workloads. Smoke-test cheap: `GPU_1xA10`, short `timeout_minutes`,
    `max_retries: 0`, your own sandbox workspace — never a customer workspace.

## 1. Install the CLI

```bash
uv tool install databricks-air
air --version   # v0.1.x — findings in this cookbook are version-scoped
```

`air -h config` is the YAML schema source of truth — more current than the web docs.

## 2. Submit a probe workload

From the repo root (`*.example.yaml` are committed templates; live copies are gitignored):

```bash
cp workloads/exec-probe.example.yaml workloads/exec-probe.yaml
air run -f workloads/exec-probe.yaml -p <profile> --dry-run   # full validation, incl. workspace API calls
air run -f workloads/exec-probe.yaml -p <profile> --watch
```

Always `--dry-run` first — it catches schema and workspace problems (nonexistent usage policy,
bad accelerator type) before you spend anything.

## 3. Know where your run went

One submission creates **two linked records** ([details](../cookbook/stream-logs-and-artifacts.md)):

| Record | Where | What it's for |
|---|---|---|
| Job run (`SUBMIT_RUN`) | Jobs UI → **Job runs** tab (never the "Jobs" tab) | Execution truth: status, compute, retries |
| MLflow run | The experiment named in `experiment_name` | Params, metrics, system metrics, log artifacts |

Stream logs live:

```bash
air logs <run-id>            # node 0
air logs <run-id> --node 1   # other nodes (multi-node)
```

## Next

- [Pick your compute](pick-your-compute.md) — before you reach for an H100
- [Submit a workload](../cookbook/submit-a-workload.md) — the YAML, field by field, with the schema quirks
