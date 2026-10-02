# Build a serverless AI wheelhouse on classic compute

> **New here?** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) is the step-by-step walkthrough of the two
> notebooks. This README is the reference for how the build works underneath.

The developer supplies two values:

1. A `requirements.txt` path.
2. A UC Volume directory for the wheelhouse builds.

Run `build_wheelhouse.py` on a classic cluster that can reach Artifactory. The notebook uses the
checked-in package baseline and platform metadata for the selected serverless AI environment. It
does not run a freeze or resolver notebook on AIR.

## Use it

Deploy this directory to the workspace, attach `build_wheelhouse` to classic compute, and set:

| Widget | Required | Value |
|---|---:|---|
| `requirements_file` | yes | Workspace or Volume path to the developer's file |
| `wheelhouse_volume` | yes | `/Volumes/<catalog>/<schema>/<volume>/<directory>` |
| `air_environment` | no | `databricks_ai_v5` (default), `databricks_ai_v4`, `databricks_ai_v6`, or `standard_v5` — must have a captured profile (see below) |
| `index_url` | no | Explicit Artifactory index; blank uses the cluster's pip configuration |

The notebook prints the generated `environment.yaml`. Apply that path as the custom serverless base
environment where that surface supports custom files. A managed AI profile selects its base with
exactly one, fully qualified selector:

```yaml
base_environment: workspace-base-environments/databricks_ai_v5
```

A standard profile instead emits `environment_version: "5"`. Jobs rejects a spec containing both
selectors before user code starts.

Native `ai_runtime_task` does **not** accept a Workspace or Volume environment file as
`spec.base_environment`. The builder therefore also emits `jobs-environment.json`; load that JSON
object and embed it directly as `environments[].spec` in the Jobs API request:

```yaml
tasks:
  - task_key: train
    environment_key: wheelhouse
    ai_runtime_task: { ... }
environments:
  - environment_key: wheelhouse
    spec:
      base_environment: workspace-base-environments/databricks_ai_v5
      dependencies:
        - --no-index
        - --require-hashes
        - --find-links /Volumes/<catalog>/<schema>/<volume>/builds/<lock-id>/wheelhouse
        - -r /Volumes/<catalog>/<schema>/<volume>/builds/<lock-id>/environment.lock
```

Do not put the path to `environment.yaml` in that field. `f-classic` rejected that shape before
launch on 2026-09-22 (Jobs run `835858704824011`): `AI Runtime supports only the databricks-ai base
environment`. The `jobs_serverless_managed_base_environments` workspace preview is not required when
every task using the environment is an `ai_runtime_task`; Jobs exempts that case. Environment
resolution happens before the AIR launcher, so a failure here has no command stdout; inspect the Jobs
task state message instead.

`environments[].spec` is the only dependency channel the runners use. AI Training forwards
`spec.dependencies` verbatim to the launcher as `--deps-config`, and `install_deps.py` installs them
with uv; the task log shows `Installing N inline environment dependencies`. Do not put a
`requirements.yaml` in the code archive; the launcher never reads it. The launcher would install a
`<command directory>/requirements.yaml` ahead of the spec, but the runners always run the command
from a staged copy (see [Staged launch directory](#staged-launch-directory)), so a `requirements.yaml`
next to the original script is never beside the running command and is never installed — both runners
just print a note if they see one.

### Submit from the workspace with the Jobs API

Deploy this directory to the workspace and run `run_ai_runtime_job` as a notebook on classic
compute. Supply the generated `/Volumes/.../jobs-environment.json`, code source, command path, and
MLflow widget values. `code_source_path` can be a source directory visible from the notebook or an
existing `.tar.gz`/`.tgz` archive. For a directory, the notebook creates a sibling archive by
default; set `code_source_archive_path` to choose another output path. It validates the archive
before calling Jobs.

The archive must have exactly one enclosing directory:

```text
wheelhouse-probe/
  verify_environment.py
```

An archive with `verify_environment.py` directly at its root fails before user code with
`could not determine top level component from tarball`. AIR extracts the enclosing directory and
sets `CODE_SOURCE_PATH` to it, so `run_probe.sh` can execute
`$CODE_SOURCE_PATH/verify_environment.py`. This is why a directory input is packaged as the
directory itself, rather than packaging only its contents.

Select exactly one `usage_policy_name` or `usage_policy_id`; the notebook resolves a name through
the workspace policy API and sends the resulting top-level `usage_policy_id` in the Jobs request.
Policy names and IDs are shown under **Compute > Usage policies**, and the notebook identity must
have access to the selected policy.

The MLflow fields have separate roles:

- `experiment` is the experiment's name, not a path.
- `mlflow_experiment_directory` is its parent workspace directory and must start with `/Workspace`.
- `mlflow_run` is the display name of this individual run inside the experiment.

For example, directory `/Workspace/Shared/air-experiments` plus experiment `wheelhouse-probe`
targets `/Workspace/Shared/air-experiments/wheelhouse-probe`. The separate directory field is
required because the native task API stores the parent workspace location and the experiment leaf
name separately. It controls where the experiment appears in the MLflow workspace tree; it is not
another experiment and has no relationship to the code source directory.

Before submission, both entry points inspect the archive and query Workspace `get-status` for the
MLflow parent. They reject a Python file used as `code_source_path`, a missing/malformed/multi-root
archive, a full experiment path used as `experiment`, a missing or non-directory parent, and a
parent that already ends in the experiment leaf. These checks run before policy lookup and before
`jobs/runs/submit`, so an input error cannot consume GPU capacity.

The notebook invokes `submit_ai_runtime_job.py` with ambient workspace
authentication, submits `POST /api/2.2/jobs/runs/submit`, and polls the Jobs API. Every submission
carries an `idempotency_token` that is **always auto-generated per run**, so an SDK retry after a
timeout cannot launch a duplicate while a deliberate re-run always gets a new run — the notebook and
the launcher job expose no token widget. It prints both the parent and task state messages, MLflow
IDs, the run URL, and task output; a failed or timed-out run raises in the notebook even when the
workload produced no stdout.

The script is also runnable outside a notebook with Databricks SDK authentication. For example,
`python submit_ai_runtime_job.py --help` lists its arguments. `--profile f-classic` selects that
local profile; omit `--profile` in the workspace so ambient authentication is used. The script's
`--code-source-path` must already be a valid workspace/Volume `.tar.gz` or `.tgz`; automatic
directory packaging is the notebook runner's convenience. The script downloads the remote archive
only for validation; Jobs still receives the original workspace/Volume path. Only the script exposes
an optional `--idempotency-token`, as the AIR CLI does, for CI that retries a submission and wants
the existing run back instead of a duplicate; because every submission stages a fresh launch
directory first (see [Staged launch directory](#staged-launch-directory)), a reused token returns
the original run and leaves the newly staged files unused, so a changed parameter or command does
not take effect under it — omit it to apply changes. Neither path invokes the AIR CLI.

The build cell reads the widgets at execution time. Its manifest records the exact requirements path
and SHA-256 digest, so rerunning only that cell after changing a widget cannot reuse stale input.

### Staged launch directory

Every submission stages, mirroring the AIR CLI: it creates a fresh per-run directory
`<launch_root>/<experiment>/<mlflow_run>_<16 hex>/`, copies `command_path` into it, and sends the
copy as the task's `command_path`. The AIR CLI does the same — it always uploads the command into a
per-run `.air/cli_launch` directory and never runs it in place (universe `5b17751197ffc`,
`cli/sdk/_submit.py`). `launch_root` defaults to `/Workspace/Users/<caller>/.air/jobs_launch`.

Running from a controlled directory is what makes dependency handling uniform across every entry
point. The AI Runtime launcher would install a `<command directory>/requirements.yaml` ahead of the
environment spec, but because the command runs from the staged copy — which contains only the
command and, when set, `hyperparameters.yaml` — a `requirements.yaml` next to the developer's
original script is never beside the running command and is never installed. Dependencies come only
from the wheelhouse `environments[].spec`, with or without parameters. Both runners still print a
one-line note if they see such a file next to the original script, so a developer who placed one
expecting the AIR CLI's behavior is not surprised.

The copied script must find its code through `$CODE_SOURCE_PATH`, not through paths relative to the
script's own location, since it now runs alone in the staged directory; `run_probe.sh` already does.
A fresh directory per run also keeps concurrent runs from sharing a parameter file. These launch
directories accumulate under `launch_root`; prune old ones periodically.

### Pass parameters to the workload

`ai_runtime_task` has no Jobs-parameter channel. Its `dynamic_value_refs` field is
DEVELOPMENT-stage and nothing reads it. Job-level `environment_variables` reach the node, but AIR
reads their raw values from the run snapshot: only `{{secrets/...}}` is substituted, and run-now
cannot override them. (Source: universe `5b17751197ffc`, `AiRuntimeTaskConverter.scala`,
`AiRuntimeTaskHandlers.scala`.) Parameters therefore travel the way the AIR CLI sends its own
`parameters`: as a file. When any parameter is set, the runner writes a `hyperparameters.yaml` into
the staged launch directory (above), and the launcher exports `HYPERPARAMETERS_PATH` to it because
`<command directory>/hyperparameters.yaml` exists (`entry_script.sh`).

Supply parameters either way, or both:

- `workload_parameters` (`--workload-parameters`): a JSON object, so it can hold typed and nested
  values.
- `workload_parameter_names` (`--param NAME=VALUE`): named string values. In the notebook, each
  listed name becomes a widget of that name, so a Jobs parameter with the same name sets it.

A name set in both places, a launcher widget name used as a workload parameter, and NaN values are
rejected before anything is written. The file is JSON, which is valid YAML, so the workload can read
it with `yaml.safe_load` or `json.load`:

```python
import json, os
params = json.load(open(os.environ["HYPERPARAMETERS_PATH"])) if "HYPERPARAMETERS_PATH" in os.environ else {}
epochs = int(params.get("epochs", 1))  # named parameters arrive as strings
```

### Persisted launcher job

A persisted job cannot put `ai_runtime_task` in its own task list and still forward Jobs
parameters. Instead, `create_launcher_job.py` creates a one-task job whose task is the
`run_ai_runtime_job` notebook. Jobs maps each job parameter onto the notebook widget of the same
name. The notebook forwards those values through `hyperparameters.yaml` and submits the AIR run with
`runs/submit`. The launcher task uses no `environments[]` entry, so whether the workspace has
`jobs_serverless_managed_base_environments` doesn't matter. The AIR run's own environment is still
the inline `jobs-environment.json` spec.

```bash
python create_launcher_job.py --profile f-classic \
  --name air-wheelhouse-launcher \
  --notebook-path /Workspace/Shared/orchestrated/run_ai_runtime_job \
  --existing-cluster-id <classic-cluster-id> \
  --jobs-environment-file /Volumes/<c>/<s>/<v>/builds/<lock-id>/jobs-environment.json \
  --code-source-path /Workspace/Shared/wheelhouse-probe.tar.gz \
  --command-path /Workspace/Shared/run_probe.sh \
  --experiment wheelhouse-probe --mlflow-experiment-directory /Workspace/Shared \
  --mlflow-run 'nightly-{{job.trigger.time.iso_date}}' \
  --usage-policy-name <policy> \
  --param epochs=3 --param lr=0.001
```

- Every launcher widget becomes a job parameter, and each `--param` becomes one more. The
  submission's idempotency token is not among them: a persisted token would make every run return
  the same AIR run, so the notebook always generates a new one each run.
- Job-parameter defaults may contain dynamic value references such as
  `{{job.trigger.time.iso_date}}`; Jobs resolves them before the notebook reads its widgets.
- Use `--serverless` instead of `--existing-cluster-id` to run the launcher notebook on serverless.
  The job's top-level `usage_policy_id` is then the same policy as the AIR run's.
- `--dry-run` prints the settings without calling Jobs, and `--job-id N` resets an existing job
  instead of creating a new one.
- The launcher task never retries: a retry would submit a second GPU run.
- With `wait=true` (the default), the launcher task polls until the AIR run finishes and fails if it
  did. The launcher compute stays up for the whole AIR run, queueing included.

Override any parameter per run:

```bash
databricks jobs run-now <job-id> --profile f-classic \
  --json '{"job_parameters": {"epochs": "10", "mlflow_run": "epochs-10"}}'
```

## Resolution algorithm

1. Start uv with the selected environment's static package pins as preferences.
2. Resolve the developer requirements for that environment's Python version and Linux platform.
3. Keep every compatible baseline version. Let uv replace a pin when the requested graph requires
   another version, including transitive conflicts such as OpenAI requiring a newer jiter.
4. Compare the complete resolved closure with the baseline by normalized package name and version.
5. Download one compatible wheel for every new or changed package from Artifactory.
6. Write the full lock, hashed wheel delta, manifest, and serverless environment YAML to a build
   directory addressed by the resolved lock's hash.

There are no overrides and no package protection list. The baseline influences version selection;
it does not overconstrain the solve. A baseline package appears in the wheelhouse only when the
resolved graph proves that its installed version must change.

## Output

Each resolution produces:

```text
<wheelhouse-volume>/
  builds/<lock-id>/
    requirements.txt       # original developer input
    resolved.lock          # complete resolved dependency closure
    delta.lock             # exact changed/new packages with wheel hashes
    environment.lock       # complete resolved closure with wheel hashes
    wheelhouse/*.whl       # wheels named by environment.lock
    environment.yaml       # apply this as the custom serverless base environment
    jobs-environment.json  # embed this object as Jobs environments[].spec for ai_runtime_task
    manifest.json          # counts, target, paths, wheel tags, and hashes
  requests/<request-id>.json
```

`environment.yaml` selects either the managed AI base or the standard environment version, then installs
the fully hashed `environment.lock` with `--no-index --require-hashes --find-links <wheelhouse>`.
The wheelhouse contains the complete resolved closure, including unchanged packages such as Typer
when the requested graph uses them. `delta.lock` separately records what differs from AI v5.
Serverless startup therefore makes no Artifactory or public PyPI request.
`delta.lock` carries the same package versions with SHA-256 hashes for audit and offline verification.

`lock-id` is derived from the selected AIR environment and normalized resolution, so the same
resolution has the same output path even if it came from a different input spelling. `request-id`
also includes the exact requirements and profile files and provides a stable success or failure
receipt for the submitted request.

## Static environment profiles

Profiles live under `profiles/<environment>/`:

- `constraints.txt` is the exact managed environment package baseline.
- `target_env.json` records the Python, ABI, manylinux platforms, and marker environment used for
  cross-resolution and wheel validation.

The checked-in `databricks_ai_v5` profile contains 433 exact package pins from the verified AIR v5
baseline and targets CPython 3.12 on x86_64 manylinux. Add a new profile when the managed serverless
environment changes; application developers should not regenerate it per build.

The v5 baseline was captured on `fevm-forrest-2` on 2026-09-10 by AIR run `52417241507965`.

### Capturing a profile for another environment

`build_wheelhouse` also lists `databricks_ai_v4`, `databricks_ai_v6`, and `standard_v5`, but each
needs a one-time captured profile before it can be selected (the notebook fails fast with the list
of available profiles otherwise). `capture_profile.py` is not part of this minimal bundle — it
lives beside the tests in the sibling `orchestrated-dev/` directory. To author a profile, copy it
in next to this `profiles/` directory and run **`capture_profile`** on the target environment —
attach a serverless notebook set to that environment version, or point its `target_python` widget
at the environment's interpreter. It runs `pip freeze --all`, probes the interpreter's wheel tags,
and writes a drop-in `profiles/<profile_id>/{constraints.txt, target_env.json}`; commit those.

`target_env.json` carries one optional field beyond the v5 schema: `base_environment`.

- **AI runtimes** (`databricks_ai_v4/v5/v6`) set it to their own name, so the rendered specs emit
  `base_environment: workspace-base-environments/databricks_ai_vN` and omit `environment_version`.
  Freeze the AI-env interpreter at
  `/opt/databricks-environments/databricks-ai/bin/python`.
- **`standard_v5`** sets it to `null`; the emitted `environment.yaml` then carries only
  `environment_version` + `dependencies` (no AI base). Confirm that shape on first apply — the
  standard serverless custom-environment spec is not `databricks_ai_*`-based. Freeze the standard
  serverless interpreter (blank `target_python`).

## Local verification

The tests live in the sibling `orchestrated-dev/` directory (kept out of this minimal bundle); each
loads the module it checks from here via `BUNDLE = HERE.parent / "orchestrated"`.

The integration test creates a private local wheel source. It covers the OpenAI/jiter conflict, a
compatible baseline dependency that uv must retain, a hidden parent conflict that requires
backtracking, hashed offline installation, and `pip check`. It does not contact PyPI.

```bash
python3 -B -m unittest \
  experiments/env-flexibility/vendored-wheels/orchestrated-dev/test_resolve_worker.py \
  experiments/env-flexibility/vendored-wheels/orchestrated-dev/test_resolve_worker_integration.py
```

The Jobs runner and launcher-job tests use fake clients, so they need no workspace:

```bash
python3 -B -m unittest \
  experiments/env-flexibility/vendored-wheels/orchestrated-dev/test_submit_ai_runtime_job.py \
  experiments/env-flexibility/vendored-wheels/orchestrated-dev/test_create_launcher_job.py
```

Wheel tags establish Python ABI and platform compatibility. Packages coupled to external native
libraries such as CUDA, Torch, or MPI still need an import or workload smoke test on the target
runtime.
