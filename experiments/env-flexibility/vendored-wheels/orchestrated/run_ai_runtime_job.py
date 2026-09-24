# Databricks notebook source
# MAGIC %md
# MAGIC # Run a wheelhouse-backed AI Runtime task
# MAGIC
# MAGIC Submit a native `ai_runtime_task` directly through the Jobs API. This notebook uses the
# MAGIC workspace's ambient authentication and the generated `jobs-environment.json`; it does not
# MAGIC invoke the AIR CLI.
# MAGIC
# MAGIC The MLflow experiment path is assembled as
# MAGIC `<mlflow_experiment_directory>/<experiment>`. For example, directory
# MAGIC `/Workspace/Shared/air-experiments` plus experiment `wheelhouse-probe` creates or reuses
# MAGIC `/Workspace/Shared/air-experiments/wheelhouse-probe`. `mlflow_run` names this individual
# MAGIC run inside that experiment.

# COMMAND ----------
dbutils.widgets.text("jobs_environment_file", "", "Generated jobs-environment.json")
dbutils.widgets.text("code_source_path", "", "Workspace code archive or directory")
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
dbutils.widgets.text("idempotency_token", "", "Optional idempotency token")
dbutils.widgets.dropdown("wait", "true", ["true", "false"], "Wait for completion")
dbutils.widgets.text("poll_seconds", "10", "Poll interval (seconds)")

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
    name: dbutils.widgets.get(name).strip()
    for name in (
        "jobs_environment_file",
        "code_source_path",
        "command_path",
        "experiment",
        "mlflow_experiment_directory",
        "mlflow_run",
        "usage_policy_name",
        "usage_policy_id",
        "accelerator_type",
        "accelerator_count",
        "timeout_seconds",
        "task_key",
        "idempotency_token",
        "wait",
        "poll_seconds",
    )
}

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

client = WorkspaceClient()
usage_policy_id = submitter.resolve_usage_policy_id(
    client,
    usage_policy_name=values["usage_policy_name"],
    usage_policy_id=values["usage_policy_id"],
)

payload = submitter.build_payload(
    jobs_environment_file=values["jobs_environment_file"],
    code_source_path=values["code_source_path"],
    command_path=values["command_path"],
    experiment=values["experiment"],
    mlflow_experiment_directory=values["mlflow_experiment_directory"],
    mlflow_run=values["mlflow_run"],
    usage_policy_id=usage_policy_id,
    accelerator_type=values["accelerator_type"],
    accelerator_count=int(values["accelerator_count"]),
    timeout_seconds=int(values["timeout_seconds"]),
    task_key=values["task_key"],
    idempotency_token=values["idempotency_token"],
)

print(f"environment spec: {values['jobs_environment_file']}")
print(f"code source:      {values['code_source_path']}")
print(f"command:          {values['command_path']}")
print(f"compute:          {values['accelerator_count']} x {values['accelerator_type']}")
print(f"usage policy id:  {usage_policy_id}")
print(
    "MLflow experiment: "
    f"{values['mlflow_experiment_directory'].rstrip('/')}/{values['experiment']}"
)

if values["wait"].lower() == "true":
    exit_code = submitter.submit_and_wait(
        client,
        payload,
        poll_seconds=float(values["poll_seconds"]),
    )
    assert exit_code == 0, "AI Runtime task did not terminate successfully; see state above"
else:
    submitter.submit_without_waiting(client, payload)
