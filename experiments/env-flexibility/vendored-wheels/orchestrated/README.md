# Build a serverless AI wheelhouse on classic compute

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
environment`. Managed bases may also require the `jobs_serverless_managed_base_environments`
workspace preview. Environment resolution happens before the AIR launcher, so a failure here has no
command stdout; inspect the Jobs task state message instead.

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
authentication, submits `POST /api/2.2/jobs/runs/submit`, and polls the Jobs API. It prints both the
parent and task state messages, MLflow IDs, the run URL, and task output; a failed or timed-out run
raises in the notebook even when the workload produced no stdout.

The script is also runnable outside a notebook with Databricks SDK authentication. For example,
`python submit_ai_runtime_job.py --help` lists its arguments. `--profile f-classic` selects that
local profile; omit `--profile` in the workspace so ambient authentication is used. The script's
`--code-source-path` must already be a valid workspace/Volume `.tar.gz` or `.tgz`; automatic
directory packaging is the notebook runner's convenience. The script downloads the remote archive
only for validation; Jobs still receives the original workspace/Volume path. Neither path invokes
the AIR CLI.

The build cell reads the widgets at execution time. Its manifest records the exact requirements path
and SHA-256 digest, so rerunning only that cell after changing a widget cannot reuse stale input.

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
of available profiles otherwise). Run **`capture_profile`** on the target environment — attach a
serverless notebook set to that environment version, or point its `target_python` widget at the
environment's interpreter. It runs `pip freeze --all`, probes the interpreter's wheel tags, and
writes a drop-in `profiles/<profile_id>/{constraints.txt, target_env.json}`; commit those.

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

The integration test creates a private local wheel source. It covers the OpenAI/jiter conflict, a
compatible baseline dependency that uv must retain, a hidden parent conflict that requires
backtracking, hashed offline installation, and `pip check`. It does not contact PyPI.

```bash
python3 -B -m unittest \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker.py \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker_integration.py
```

Wheel tags establish Python ABI and platform compatibility. Packages coupled to external native
libraries such as CUDA, Torch, or MPI still need an import or workload smoke test on the target
runtime.
