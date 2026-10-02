# Databricks notebook source
# MAGIC %md
# MAGIC # Run a wheelhouse-backed AI Runtime task
# MAGIC
# MAGIC Submit a native `ai_runtime_task` directly through the Jobs API. This notebook uses the
# MAGIC workspace's ambient authentication and the generated `jobs-environment.json`; it does not
# MAGIC invoke the AIR CLI.
# MAGIC
# MAGIC `code_source_path` may be a workspace/Volume directory or an existing `.tar.gz`/`.tgz`.
# MAGIC AIR does not accept a directory directly: its archive must contain one enclosing directory
# MAGIC (`project/script.py`, not `script.py` at the archive root). This notebook packages a source
# MAGIC directory automatically and validates an existing archive before it submits anything.
# MAGIC `CODE_SOURCE_PATH` in the task points at that extracted enclosing directory.
# MAGIC
# MAGIC The MLflow experiment path is assembled as
# MAGIC `<mlflow_experiment_directory>/<experiment>`. For example, directory
# MAGIC `/Workspace/Shared/air-experiments` plus experiment `wheelhouse-probe` creates or reuses
# MAGIC `/Workspace/Shared/air-experiments/wheelhouse-probe`. The directory is required because the
# MAGIC Jobs API models the parent workspace location separately from the experiment's leaf name;
# MAGIC it is not a second experiment and it is unrelated to the code archive. `mlflow_run` names
# MAGIC this individual run inside that experiment. Before submission, the notebook verifies that
# MAGIC the parent exists and is a Workspace directory and that the experiment value is only a leaf
# MAGIC name.
# MAGIC
# MAGIC **Staged launch directory.** Every submission copies `command_path` into a fresh per-run
# MAGIC directory under `launch_root` and runs the copy, mirroring the AIR CLI (which always stages
# MAGIC into `.air/cli_launch` and never runs your command in place). This keeps dependency handling
# MAGIC uniform: a `requirements.yaml` next to your original script is never beside the running
# MAGIC command, so it is never installed — dependencies come only from the wheelhouse environment
# MAGIC spec. `launch_root` defaults to `/Workspace/Users/<you>/.air/jobs_launch`.
# MAGIC
# MAGIC **Workload parameters.** `ai_runtime_task` has no Jobs-parameter channel, so parameters reach
# MAGIC the command as a file: when set, this notebook writes a `hyperparameters.yaml` into the launch
# MAGIC directory, and the launcher exports `HYPERPARAMETERS_PATH` pointing at it. Set
# MAGIC `workload_parameters` to a JSON object and/or list job parameter names in
# MAGIC `workload_parameter_names`; each listed name becomes a widget whose string value is
# MAGIC forwarded. Because job parameters map onto widgets of the same name, this notebook is also
# MAGIC the task of the persisted launcher job that `create_launcher_job.py` creates; override its
# MAGIC job parameters per run with run-now.

# COMMAND ----------
dbutils.widgets.text("jobs_environment_file", "", "Generated jobs-environment.json")
dbutils.widgets.text("code_source_path", "", "Source directory or .tar.gz/.tgz archive")
dbutils.widgets.text(
    "code_source_archive_path", "", "Archive output for a directory (blank = sibling .tar.gz)"
)
dbutils.widgets.text("command_path", "", "Workspace command/script path")
dbutils.widgets.text("experiment", "", "MLflow experiment name (not a path)")
dbutils.widgets.text(
    "mlflow_experiment_directory", "", "Parent workspace directory (/Workspace/...)"
)
dbutils.widgets.text("mlflow_run", "", "MLflow run display name")
dbutils.widgets.text("usage_policy_name", "", "Usage policy name (name or ID required)")
dbutils.widgets.text("usage_policy_id", "", "Usage policy ID (name or ID required)")
dbutils.widgets.text("accelerator_type", "GPU_1xA10", "Accelerator type")
dbutils.widgets.text("accelerator_count", "1", "Accelerator count")
dbutils.widgets.text("timeout_seconds", "600", "Task timeout (seconds)")
dbutils.widgets.text("task_key", "training", "Task key")
dbutils.widgets.dropdown("wait", "true", ["true", "false"], "Wait for completion")
dbutils.widgets.text("poll_seconds", "10", "Poll interval (seconds)")
dbutils.widgets.text("workload_parameters", "", "Workload parameters (JSON object)")
dbutils.widgets.text(
    "workload_parameter_names", "", "Job parameter names forwarded to the workload (a,b,...)"
)
dbutils.widgets.text("launch_root", "", "Launch dir root (blank = ~/.air/jobs_launch)")

# COMMAND ----------
import importlib.util
from pathlib import Path

from databricks.sdk import WorkspaceClient


context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
notebook_path = context.notebookPath().get()
notebook_dir = Path("/Workspace") / Path(notebook_path.lstrip("/")).parent
submitter_path = notebook_dir / "submit_ai_runtime_job.py"
assert submitter_path.is_file(), f"submitter script is missing: {submitter_path}"

module_spec = importlib.util.spec_from_file_location("air_wheelhouse_job_submitter", submitter_path)
assert module_spec and module_spec.loader, f"cannot load submitter script: {submitter_path}"
submitter = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(submitter)

# COMMAND ----------
values = {
    name: dbutils.widgets.get(name).strip() for name in submitter.LAUNCHER_PARAMETER_NAMES
}

# Declaring each named widget lets a job parameter of the same name set it; interactively it
# appears blank for the user to fill in. Jobs delivers these values as strings.
workload_parameter_names = submitter.parse_workload_parameter_names(
    values["workload_parameter_names"]
)
for name in workload_parameter_names:
    dbutils.widgets.text(name, "", f"Workload parameter: {name}")
workload_parameters = submitter.parse_workload_parameters(
    values["workload_parameters"],
    {name: dbutils.widgets.get(name) for name in workload_parameter_names},
)

required = (
    "jobs_environment_file",
    "code_source_path",
    "command_path",
    "experiment",
    "mlflow_experiment_directory",
    "mlflow_run",
    "accelerator_type",
    "task_key",
)
missing = [name for name in required if not values[name]]
assert not missing, f"required widgets are blank: {', '.join(missing)}"
assert bool(values["usage_policy_name"]) != bool(values["usage_policy_id"]), (
    "set exactly one of usage_policy_name or usage_policy_id"
)
assert values["jobs_environment_file"].startswith("/Volumes/"), (
    "jobs_environment_file must be a /Volumes path emitted by build_wheelhouse"
)

# The wheelhouse dependencies reach the node through environments[].spec (built from
# jobs-environment.json in build_payload); the launcher installs them via --deps-config.
# prepare_code_source_archive validates the archive through the notebook's FUSE mount, so only
# the MLflow parent still needs a remote check.
code_source_archive, code_source_component, archive_created = (
    submitter.prepare_code_source_archive(
        values["code_source_path"],
        values["code_source_archive_path"],
    )
)

client = WorkspaceClient()
experiment, mlflow_experiment_directory, mlflow_experiment_path = (
    submitter.validate_mlflow_experiment_parent(
        client, values["experiment"], values["mlflow_experiment_directory"]
    )
)
# The command runs from a staged copy, so a requirements.yaml next to the original is ignored.
submitter.warn_on_colocated_requirements_yaml(client, values["command_path"])
usage_policy_id = submitter.resolve_usage_policy_id(
    client,
    usage_policy_name=values["usage_policy_name"],
    usage_policy_id=values["usage_policy_id"],
)

payload = submitter.build_payload(
    jobs_environment_file=values["jobs_environment_file"],
    code_source_path=code_source_archive,
    command_path=values["command_path"],
    experiment=experiment,
    mlflow_experiment_directory=mlflow_experiment_directory,
    mlflow_run=values["mlflow_run"],
    usage_policy_id=usage_policy_id,
    accelerator_type=values["accelerator_type"],
    accelerator_count=int(values["accelerator_count"]),
    timeout_seconds=int(values["timeout_seconds"]),
    task_key=values["task_key"],
    # No idempotency widget: always auto-generated per run so a re-run never dedups to a stale run.
)

# Staged after the payload is built, so an input error writes nothing to the workspace. Every
# submission stages, so the command always runs from a controlled directory (see the submitter).
command_path, hyperparameters_path = submitter.stage_launch_directory(
    client,
    command_path=values["command_path"],
    experiment=experiment,
    mlflow_run=values["mlflow_run"],
    parameters=workload_parameters,
    launch_root=values["launch_root"],
)
payload = submitter.with_command_path(payload, command_path)

print(f"environment spec: {values['jobs_environment_file']}")
print(f"code source input:   {values['code_source_path']}")
print(f"code source archive: {code_source_archive}")
print(f"archive action:      {'created' if archive_created else 'validated'}")
print(f"CODE_SOURCE_PATH:    <runtime>/{code_source_component}")
print(f"idempotency token:   {payload['idempotency_token']}")
print(f"command source:   {values['command_path']}")
print(f"staged command:   {command_path}")
if hyperparameters_path:
    print(f"HYPERPARAMETERS_PATH: {hyperparameters_path}")
    print(submitter.render_hyperparameters_yaml(workload_parameters), end="")
print(f"compute:          {values['accelerator_count']} x {values['accelerator_type']}")
print(f"usage policy id:  {usage_policy_id}")
print(f"MLflow experiment: {mlflow_experiment_path}")

if values["wait"].lower() == "true":
    exit_code = submitter.submit_and_wait(
        client,
        payload,
        poll_seconds=float(values["poll_seconds"]),
    )
    assert exit_code == 0, "AI Runtime task did not terminate successfully; see state above"
else:
    submitter.submit_without_waiting(client, payload)
