# Generated wheelhouse environment via native Jobs API

## Question and pre-registered pass criteria

Does the corrected `environment.yaml` produced by `build_wheelhouse` construct successfully when a
one-time run references it from a native Jobs `ai_runtime_task`?

Target: `f-classic`, one `GPU_1xA10`, environment v5, no retries, 10-minute task timeout. Submission
is directly through the Jobs API (`jobs/runs/submit`), not the AIR CLI. MLflow is the evidence sink.

A pass requires all of the following:

1. The submitted run metadata contains `tasks[0].ai_runtime_task`, `GPU_1xA10`, and
   `accelerator_count=1`.
2. Its job environment points at a generated Volume YAML containing exactly one selector.
3. User code starts and observes the wheelhouse's exact `mlflow==3.16.0`, `pydantic==2.13.5`, and
   `scikit-learn==1.9.1` versions.
4. One A10 is visible, the task terminates `SUCCESS`, and stdout contains
   `JOBS_API_WHEELHOUSE_ENV_OK` plus `VERDICT: ACCEPTED`.
5. The linked MLflow run contains the acceptance receipt parameters.

| Claim | Required evidence |
|---|---|
| Jobs used the native task and requested shape | quoted `jobs/runs/get` task JSON |
| The corrected generated environment constructed | user code starts and exact package-version check passes |
| The probe completed on the requested GPU | pass-gated sentinel, acceptance report, terminal `SUCCESS` |

## Pre-flight

The original 2026-09-22 pre-flight compiled the scripts, but its archive listing exposed an invalid
layout that was not recognized at the time:

```text
$ python3 -B -m py_compile verify_environment.py
[exit 0]
$ bash -n run_probe.sh
[exit 0]
$ tar -tzf jobs-api-wheelhouse-probe.tgz
verify_environment.py
```

`verify_environment.py` at the tar root is not a valid native `ai_runtime_task` code source. AIR
must be able to identify one enclosing component and expects, for example,
`jobs-api-wheelhouse/verify_environment.py`. A submission with the root-level layout fails before
user code with `could not determine top level component from tarball`.

The workspace runner now either packages a directory with its directory name preserved or validates
an existing `.tar.gz`/`.tgz` before submission. Local regression on 2026-09-24:

```text
$ python3 -B -m unittest experiments/env-flexibility/vendored-wheels/orchestrated/test_submit_ai_runtime_job.py
...............
Ran 15 tests in 0.022s
OK
```

The renderer copy matches the canonical acceptance renderer. The Jobs payload is
`jobs-submit.json`; it contains one native `ai_runtime_task`, one A10, no retries, and a 10-minute
timeout. The corrected remote YAML is a non-destructive sibling of build `960b07edc9cf3fc7`; the
original two-selector file is retained as the negative-control receipt. A live run is not a
benchmark; it only tests environment construction and package activation.

## Observed

### Attempt 1 — custom environment file path rejected before launch

❌ **REJECTED 2026-09-22, Jobs run `835858704824011`, task run `1104650716569647`,
`f-classic`.** Direct `POST /api/2.2/jobs/runs/submit`; no AIR CLI. The payload and returned task
metadata both contain native `ai_runtime_task`, `GPU_1xA10`, and `accelerator_count=1`. The run
failed before an MLflow run or user stdout existed:

```text
INTERNAL_ERROR / FAILED
Task wheelhouse_env_probe failed with message: An error occurred during execution of task AI Runtime Task:
Unsupported base environment '/Volumes/forrest_fdm/airlab/deps/builds/960b07edc9cf3fc7/environment.jobs-api-20260922.yaml'.
AI Runtime supports only the databricks-ai base environment, e.g. 'workspace-base-environments/databricks_ai_v5'.
```

| Claim | Evidence |
|---|---|
| Jobs used native AI Runtime at the requested shape | returned task JSON contains `ai_runtime_task`, `GPU_1xA10`, `accelerator_count=1` |
| Correcting the YAML selectors makes its Volume path usable by `ai_runtime_task` | **FAIL** — the task type rejects custom base-environment paths outright |

### Attempt 2 — pre-registered supported shape

`jobs-submit-inline.json` uses the server-returned supported base ID
`workspace-base-environments/databricks_ai_v5` and supplies the generated wheelhouse offline flags
inline in `environments[].spec.dependencies`. Pass criteria remain the exact versions, one visible
A10, terminal `SUCCESS`, sentinel, acceptance report, and MLflow receipt above.

⚠️ **BLOCKED BY CAPACITY 2026-09-22, Jobs run `951241043188348`, task run
`1017402744436040`, `f-classic`; MLflow run `b2cf98c236784d5097fd9a784c75f9c3`.** Direct
`POST /api/2.2/jobs/runs/submit`; no AIR CLI. Jobs accepted and persisted the fully qualified managed
base plus inline dependencies. While queued, `jobs/runs/get-output` repeatedly returned:

```text
STATUS:Waiting for GPU compute capacity to become available.
```

The task never reached environment installation or user code. After 653 seconds it terminated:

```text
task: TERMINATED / TIMEDOUT — Run timed out
parent: INTERNAL_ERROR / FAILED — Task wheelhouse_env_probe failed with message: Run timed out.
MLflow status: KILLED
```

No sentinel, acceptance parameters, or stdout exists, which is expected because the launcher never
started. This proves the supported Jobs API shape passes control-plane validation on `f-classic`;
it does **not** verify that the wheelhouse installs or that its packages import. Re-run the same
payload with a new `idempotency_token` when A10 capacity is available. The workspace MLflow receipt
is archived locally as run `ea23f3540e024720ad0a8ce1e15cda50` in experiment
`/Users/forrest.murray@databricks.com/air-lab-jobs-api-wheelhouse`.

| Claim | Evidence |
|---|---|
| Jobs accepted the supported managed-base shape | returned run JSON persists `workspace-base-environments/databricks_ai_v5` and all three inline offline dependency entries |
| The package environment constructed and activated | **BLOCKED** — capacity wait ended in `TIMEDOUT` before launcher/user code |
