# Developer guide: run a GPU workload with a vendored wheelhouse

This is the step-by-step guide for a developer who wants to run an AI Runtime (`ai_runtime_task`)
GPU workload in a workspace that **cannot reach public PyPI**. You build a self-contained wheelhouse
once on classic compute, then submit the GPU run through the Jobs API. For how any of this works
underneath (the uv resolution algorithm, the build output layout, environment profiles), see
[`README.md`](README.md).

Two notebooks do the work:

| Notebook | Runs on | Does |
|---|---|---|
| `build_wheelhouse` | classic cluster with Artifactory access | resolves your `requirements.txt` against a serverless AI base and writes a wheelhouse + `jobs-environment.json` to a UC Volume |
| `run_ai_runtime_job` | classic cluster (or serverless), ambient auth | submits the native `ai_runtime_task` via `jobs/runs/submit`, waits, and prints the result |

```
requirements.txt ──> build_wheelhouse ──> /Volumes/.../builds/<lock-id>/jobs-environment.json
                                                     │
                        your training code ──────────┼──> run_ai_runtime_job ──> GPU run
```

## Before you start

1. **Deploy this whole directory to the workspace, in one folder.** Both notebooks load their
   siblings by path (`build_wheelhouse` does `%run ./resolve_worker`; `run_ai_runtime_job` imports
   `submit_ai_runtime_job.py` from the same directory). Keep `profiles/` next to them.
2. **A captured profile for your base environment.** `databricks_ai_v5` ships with one. For
   `databricks_ai_v4`, `databricks_ai_v6`, or `standard_v5` you must run `capture_profile` once on
   that environment first — it lives in the sibling `orchestrated-dev/` directory (not shipped in
   this bundle); copy it in next to `profiles/` to author one. See
   [README "Capturing a profile"](README.md#capturing-a-profile-for-another-environment).
3. **A UC Volume** you can write to, for the wheelhouse output.
4. **A serverless usage policy** your identity can use (Compute → Usage policies), for the GPU run.

---

## Step 1 — Build the wheelhouse (`build_wheelhouse`)

Attach `build_wheelhouse` to a **classic cluster that can reach Artifactory** and set the widgets:

| Widget | Required | Value |
|---|:---:|---|
| `requirements_file` | yes | Workspace or Volume path to your `requirements.txt` |
| `wheelhouse_volume` | yes | `/Volumes/<catalog>/<schema>/<volume>/<dir>` output directory |
| `air_environment` | no | `databricks_ai_v5` (default), `databricks_ai_v4`, `databricks_ai_v6`, or `standard_v5` — must have a captured profile |
| `index_url` | no | Explicit Artifactory index; blank uses the cluster's pip configuration |

Run it. On success the last cells print two paths:

```
Apply this custom serverless environment file:
  /Volumes/.../builds/<lock-id>/environment.yaml
For a native Jobs ai_runtime_task, embed this JSON object as environments[].spec:
  /Volumes/.../builds/<lock-id>/jobs-environment.json
```

**Copy the `jobs-environment.json` path** — that is the only build output Step 2 needs. (Native
`ai_runtime_task` does not accept the `environment.yaml` file path; it needs the JSON object embedded
inline, which is what Step 2 does for you.) The build is addressed by a hash of your inputs, so the
same `requirements.txt` + profile always writes to the same `<lock-id>` directory.

---

## Step 2 — Run the GPU workload (`run_ai_runtime_job`)

### What you supply

- **`jobs-environment.json`** from Step 1.
- **Your code**, as a `code_source_path`: either a source directory or an existing `.tar.gz`/`.tgz`.
  The archive must contain exactly **one enclosing directory** (`project/train.py`, not `train.py` at
  the root). A directory input is packaged for you; an existing archive is validated. At runtime AIR
  extracts it and sets `CODE_SOURCE_PATH` to that enclosing directory.
- **`command_path`**: a `/Workspace` shell script that launches your code. It runs after environment
  setup with `CODE_SOURCE_PATH` exported, so a one-line `exec python3 "$CODE_SOURCE_PATH/train.py"`
  is typical. (See the probe `run_probe.sh` for a working example.)
- **MLflow target**, split into two fields the Jobs API models separately:
  - `experiment` — the experiment **leaf name only**, e.g. `my-training`.
  - `mlflow_experiment_directory` — its existing `/Workspace` **parent**, e.g. `/Workspace/Shared/air`.
  - The run lands at `<directory>/<experiment>`; `mlflow_run` names the individual run.
- **A usage policy**: set exactly **one** of `usage_policy_name` or `usage_policy_id`.

### Widget reference

| Widget | Required | Default | Notes |
|---|:---:|---|---|
| `jobs_environment_file` | yes | — | `/Volumes/...` path from Step 1 |
| `code_source_path` | yes | — | source directory or `.tar.gz`/`.tgz` |
| `code_source_archive_path` | no | — | where to write the archive when `code_source_path` is a directory; blank = sibling `.tar.gz` |
| `command_path` | yes | — | `/Workspace` launch script |
| `experiment` | yes | — | MLflow experiment **leaf name**, not a path |
| `mlflow_experiment_directory` | yes | — | existing `/Workspace` parent directory |
| `mlflow_run` | yes | — | run display name |
| `usage_policy_name` / `usage_policy_id` | one | — | exactly one |
| `accelerator_type` | yes | `GPU_1xA10` | e.g. `GPU_1xA10`, `GPU_8xH100` |
| `accelerator_count` | no | `1` | total GPUs across nodes; must be a multiple of the per-node count in `accelerator_type` |
| `timeout_seconds` | no | `600` | task timeout |
| `task_key` | yes | `training` | — |
| `wait` | no | `true` | `true` polls to completion and fails the cell on a non-`SUCCESS` run; `false` submits and returns |
| `poll_seconds` | no | `10` | poll interval while waiting |
| `workload_parameters` | no | blank | see "Passing parameters" below |
| `workload_parameter_names` | no | blank | see "Passing parameters" below |
| `launch_root` | no | blank | root for the per-run staged launch dir (every run stages); blank = `/Workspace/Users/<you>/.air/jobs_launch` |

Run it. The notebook validates everything **before** calling Jobs (so an input mistake never spends
GPU capacity), prints the resolved payload, then submits and — with `wait=true` — streams
parent/task state, the run URL, MLflow IDs, and task output. The submission's idempotency token
(which stops an SDK retry from launching a duplicate) is always generated per run; there is no
token widget. Only the standalone script exposes an optional `--idempotency-token`, for CI — see
below.

### Where dependencies come from (important)

Your wheelhouse dependencies travel **only** through `environments[].spec` (built from
`jobs-environment.json`); the launcher installs them with uv via `--deps-config`. **Do not rely on a
`requirements.yaml`** — the launcher does not read one from your code archive, and this workflow runs
your command from a staged copy (see below), so a `requirements.yaml` sitting next to your original
script is never beside the running command and is never installed. The runner prints a one-line note
if it finds one, so you aren't surprised; remove it to avoid confusion.

---

## How your command is launched (staging)

Every submission **stages**: the runner creates a fresh per-run directory
`<launch_root>/<experiment>/<run>_<id>/`, copies `command_path` into it, and runs the copy. This
mirrors the AIR CLI, which always uploads your command into a `.air/cli_launch` directory and never
runs it in place. `launch_root` defaults to `/Workspace/Users/<you>/.air/jobs_launch`. Two things
follow:

- Your command runs **alone** in the staged directory, so it must locate your code through
  `$CODE_SOURCE_PATH` (as `run_probe.sh` does), not through paths relative to the script itself.
- Dependency behavior is the same whether or not you pass parameters: a `requirements.yaml` next to
  your original script is never beside the running command, so it is never installed (see above).

These directories accumulate under `launch_root`; prune old ones periodically. For a **scheduled job
whose run-as is a service principal**, set `launch_root` explicitly — the default
`/Workspace/Users/<sp>` home may not exist.

## Passing parameters to the workload

`ai_runtime_task` has no Jobs-parameter channel, so parameters reach your code as a **file**. When you
set either parameter widget, the runner writes a `hyperparameters.yaml` into the staged launch
directory, and the launcher exports `HYPERPARAMETERS_PATH` pointing at it. Your script reads that file
(JSON is valid YAML, so `yaml.safe_load` or OmegaConf both work):

- `workload_parameters` — a JSON object for typed or nested values, e.g. `{"lr": 0.001, "epochs": 3}`.
- `workload_parameter_names` — a comma-separated list, e.g. `lr,epochs`. Each name becomes its own
  widget whose string value is forwarded (this is the channel a persisted launcher job's job
  parameters flow through — see below). A name set in both places is rejected.

Leave both blank and the command still runs from a staged copy, just without a `hyperparameters.yaml`.

---

## Optional — a reusable launcher job (`create_launcher_job.py`)

To schedule the workload or re-run it from the Jobs UI with "Run now (with different parameters)",
persist a Jobs job whose single task is the `run_ai_runtime_job` notebook. Every widget becomes a job
parameter, so run-now can override any of them. (The submission's idempotency token is not a widget
or parameter — it is always generated per run.) This is a **CLI script**, not a notebook:

```bash
python create_launcher_job.py \
  --name "air-wheelhouse-training" \
  --notebook-path /Workspace/Shared/air/run_ai_runtime_job \
  --serverless \
  --jobs-environment-file /Volumes/.../builds/<lock-id>/jobs-environment.json \
  --code-source-path /Workspace/Shared/air/train.tgz \
  --command-path /Workspace/Shared/air/run.sh \
  --experiment my-training \
  --mlflow-experiment-directory /Workspace/Shared/air \
  --mlflow-run baseline \
  --usage-policy-name "AIR Team" \
  --param lr=0.001 --param epochs=3
```

- Pick compute with exactly one of `--serverless` or `--existing-cluster-id <id>`.
- `--param NAME=DEFAULT` (repeatable) exposes a named workload parameter as its own job parameter.
- `--job-id <id>` resets an existing job instead of creating a new one.
- `--dry-run` prints the `JobSettings` without calling Jobs.
- Run with `--help` for every launcher default flag.

---

## Run outside a notebook (optional)

Both the submitter and the launcher-builder run as plain scripts with Databricks SDK auth, e.g.
`python submit_ai_runtime_job.py --help`. In that mode `--code-source-path` must already be a valid
`.tar.gz`/`.tgz` (the directory auto-packaging is a notebook convenience), and `--profile <name>`
selects a local Databricks profile; omit it in the workspace to use ambient auth.

The script also accepts `--idempotency-token` for CI that may retry a submission: a reused token
returns the existing run instead of launching a duplicate. Because every submission stages a fresh
launch directory first, a reused token returns the *original* run and the newly staged
command/parameters are unused — omit it (the default auto-generates one per run) whenever you want
changes to take effect.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Unsupported base environment ... AI Runtime supports only the databricks-ai base environment` | You passed an `environment.yaml` **file path** as the base environment. Use `jobs-environment.json` (Step 1); the notebook embeds the managed base ID + deps inline. |
| `could not determine top level component from tarball` | Your archive has files at its root. Package the enclosing directory (`project/train.py`), not its contents. |
| `no captured profile for '<env>'` | Run `capture_profile` on that environment first. Only `databricks_ai_v5` ships with a profile. |
| Run sits in `Waiting for GPU compute capacity`, then `TIMEDOUT` | No A10/H100 capacity for the policy at submit time. Re-run when capacity frees up; a reused `idempotency_token` returns the timed-out run, so use a fresh one. |
| Environment resolution fails with no command stdout | Resolution happens **before** your code starts, so there is no stdout. Read the **Jobs task state message**, not the logs. |
| Wrong package versions at runtime | Dependencies come only from the wheelhouse spec. A stray `requirements.yaml` next to your script is **not** installed (the command runs from a staged copy), so look at the resolved `environments[].spec`, not that file. |

For the resolution algorithm, build directory layout, and environment profiles, see
[`README.md`](README.md).
