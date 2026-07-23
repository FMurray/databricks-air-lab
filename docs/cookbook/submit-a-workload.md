# Submit a workload

Goal: write a workload YAML that validates the first time, and know exactly what each field does.

Schema source of truth: `air -h config` (and `air -h config.<section>`), not the web docs.
Everything below is verified against `air` v0.1.0 unless labeled otherwise.

## The YAML, annotated

```yaml
experiment_name: my-training          # names the MLflow experiment

environment:
  version: "4"                        # managed env; if set, `dependencies` must exist — use [] if none
  dependencies: [torch, transformers] # pip/uv style: inline specs, -r requirements.txt, wheels
  # docker_image: {url: docker.io/org/img:tag}   # mutually exclusive with version/dependencies

compute:
  num_accelerators: 8                 # whole-node multiples for multi-GPU types
  accelerator_type: GPU_8xH100        # 16 ⇒ 2 nodes: the multi-node path

code_source:
  type: snapshot
  snapshot:
    root_path: ..                     # ⚠ resolves relative to the YAML file's dir, NOT your CWD
    include_paths: [src/]

command: torchrun --nproc_per_node=8 $CODE_SOURCE_PATH/src/train.py
                                      # arbitrary bash — the escape hatch for everything non-Python

env_variables: {HF_HOME: /tmp/hf}     # ⚠ TOP-LEVEL. Nested under environment: → "Unknown field"
secrets: {HF_TOKEN: my_scope/hf_token}  # ⚠ also top-level; resolved from secret scopes at launch

parameters: {training: {batch_size: 32}}  # materialized to a YAML file; path in $HYPERPARAMETERS_PATH

max_retries: 0                        # keep 0 unless you're testing retry semantics
timeout_minutes: 90
mlflow_run_name: baseline-001
# usage_policy_name: my-team-policy   # ⚠ must exist in the target workspace or validation fails
```

## Submit

```bash
air run -f workload.yaml -p <profile> --dry-run   # always: full validation incl. workspace API calls
air run -f workload.yaml -p <profile> --watch
```

Overrides work as advertised, handy for card sweeps:

```bash
air run -f workload.yaml --override compute.accelerator_type=GPU_1xA10 mlflow_run_name=a10-run
```

## The quirks that cost us hours

| Quirk | Consequence | Receipt |
|---|---|---|
| `env_variables` under `environment:` is rejected | put it at top level (docs showed it nested) | ✅ 2026-07-17, `--dry-run` |
| `environment.version` requires a `dependencies` list | use `dependencies: []` even with preinstalled torch | ✅ 2026-07-22, run 505819227973807 |
| `snapshot.root_path` is relative to the YAML's directory | YAMLs kept in a subdir need `root_path: ..` | ✅ 2026-07-17 |
| `usage_policy_name` must reference an existing policy | validation fails, not a warning — omit if unsure | ✅ 2026-07-17, e2-demo-field-eng |
| no `parameters:` block → no `$HYPERPARAMETERS_PATH` | anything deriving identity/config from it silently gets nothing | ✅ 2026-07-17 |

!!! danger "Never dump raw env in `command`"
    `secrets:` values arrive as environment variables, and `command` output lands in job logs.
    An `env | sort` in a command **printed a live token into the logs** here once. Filter:
    `env | grep -vE 'TOKEN|SECRET|KEY'`.

## Next

- [Run multi-node training](run-multi-node-training.md)
- [Stream logs and artifacts](stream-logs-and-artifacts.md) — where the output of that `command` goes
