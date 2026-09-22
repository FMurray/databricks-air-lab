"""Submit a wheelhouse-backed ``ai_runtime_task`` through the Databricks Jobs API.

This module intentionally uses ``WorkspaceClient.api_client.do`` instead of the typed Jobs SDK
models. AI Runtime task support can reach the REST API before it reaches an installed SDK version.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping


TERMINAL_LIFE_CYCLE_STATES = {"BLOCKED", "INTERNAL_ERROR", "SKIPPED", "TERMINATED"}


def load_environment_spec(jobs_environment_file: str | Path) -> dict[str, Any]:
    """Load and validate a generated ``jobs-environment.json`` file."""
    path = Path(jobs_environment_file)
    try:
        spec = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"jobs environment file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"jobs environment file is not valid JSON: {path}: {exc}") from exc

    if not isinstance(spec, dict):
        raise ValueError(f"jobs environment file must contain a JSON object: {path}")

    selectors = [
        key for key in ("base_environment", "environment_version") if spec.get(key)
    ]
    if len(selectors) != 1:
        raise ValueError(
            "jobs environment spec must set exactly one non-empty selector: "
            "base_environment or environment_version"
        )

    dependencies = spec.get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) for dependency in dependencies
    ):
        raise ValueError("jobs environment spec dependencies must be a list of strings")

    return spec


def build_payload(
    jobs_environment_file: str | Path,
    code_source_path: str,
    command_path: str,
    experiment: str,
    mlflow_experiment_directory: str,
    mlflow_run: str,
    accelerator_type: str = "GPU_1xA10",
    accelerator_count: int = 1,
    timeout_seconds: int = 600,
    task_key: str = "training",
    idempotency_token: str = "",
) -> dict[str, Any]:
    """Build the raw ``jobs/runs/submit`` request body."""
    required = {
        "code_source_path": code_source_path,
        "command_path": command_path,
        "experiment": experiment,
        "mlflow_experiment_directory": mlflow_experiment_directory,
        "mlflow_run": mlflow_run,
        "accelerator_type": accelerator_type,
        "task_key": task_key,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"required values are blank: {', '.join(missing)}")
    if accelerator_count < 1:
        raise ValueError("accelerator_count must be at least 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1")

    environment_spec = load_environment_spec(jobs_environment_file)
    environment_key = "wheelhouse"
    payload: dict[str, Any] = {
        "run_name": f"{task_key}-{mlflow_run}",
        "tasks": [
            {
                "task_key": task_key,
                "environment_key": environment_key,
                "max_retries": 0,
                "timeout_seconds": timeout_seconds,
                "ai_runtime_task": {
                    "experiment": experiment,
                    "mlflow_experiment_directory": mlflow_experiment_directory,
                    "mlflow_run": mlflow_run,
                    "code_source_path": code_source_path,
                    "deployments": [
                        {
                            "command_path": command_path,
                            "compute": {
                                "accelerator_type": accelerator_type,
                                "accelerator_count": accelerator_count,
                            },
                        }
                    ],
                },
            }
        ],
        "environments": [
            {
                "environment_key": environment_key,
                "spec": environment_spec,
            }
        ],
    }
    if idempotency_token.strip():
        payload["idempotency_token"] = idempotency_token.strip()
    return payload


def _get_run(client: Any, run_id: int) -> dict[str, Any]:
    return client.api_client.do(
        method="GET",
        path="/api/2.2/jobs/runs/get",
        query={"run_id": run_id},
    )


def _get_task_output(client: Any, task_run_id: int) -> dict[str, Any]:
    return client.api_client.do(
        method="GET",
        path="/api/2.1/jobs/runs/get-output",
        query={"run_id": task_run_id},
    )


def submit_run(client: Any, payload: Mapping[str, Any]) -> int:
    """Submit one run and return its parent run ID."""
    response = client.api_client.do(
        method="POST",
        path="/api/2.2/jobs/runs/submit",
        body=dict(payload),
    )
    run_id = response.get("run_id")
    if run_id is None:
        raise RuntimeError(f"Jobs submit response did not contain run_id: {response!r}")
    print(f"parent run id: {run_id}")
    return int(run_id)


def submit_without_waiting(client: Any, payload: Mapping[str, Any]) -> int:
    """Submit one run, print its Jobs URL, and return its parent run ID."""
    run_id = submit_run(client, payload)
    try:
        run = _get_run(client, run_id)
        print(f"run URL: {run.get('run_page_url') or '<not available>'}")
    except Exception as exc:
        print(f"run URL: <not yet available: {type(exc).__name__}: {exc}>")
    return run_id


def _task_for_key(run: Mapping[str, Any], task_key: str) -> Mapping[str, Any]:
    tasks = run.get("tasks") or []
    for task in tasks:
        if task.get("task_key") == task_key:
            return task
    return tasks[0] if tasks else {}


def _state_line(label: str, state: Mapping[str, Any]) -> str:
    life_cycle = state.get("life_cycle_state") or "UNKNOWN"
    result = state.get("result_state") or "PENDING"
    message = state.get("state_message") or "<none>"
    return f"{label}: {life_cycle} / {result} — {message}"


def _find_values(value: Any, key: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if nested_key == key and nested_value not in (None, ""):
                found.append(str(nested_value))
            found.extend(_find_values(nested_value, key))
    elif isinstance(value, list):
        for nested_value in value:
            found.extend(_find_values(nested_value, key))
    return list(dict.fromkeys(found))


def _print_final_details(
    run: Mapping[str, Any], task: Mapping[str, Any], output: Mapping[str, Any]
) -> None:
    print(_state_line("parent", run.get("state") or {}))
    if task:
        print(_state_line("task", task.get("state") or {}))
    print(f"run URL: {run.get('run_page_url') or '<not available>'}")

    combined = {"run": run, "output": output}
    experiment_ids = _find_values(combined, "mlflow_experiment_id")
    mlflow_run_ids = _find_values(combined, "mlflow_run_id")
    print(f"MLflow experiment id: {', '.join(experiment_ids) or '<not available>'}")
    print(f"MLflow run id: {', '.join(mlflow_run_ids) or '<not available>'}")

    printed_output = False
    for key in ("error", "error_trace", "logs"):
        value = output.get(key)
        if value:
            print(f"task output {key}:\n{value}")
            printed_output = True
    if output.get("logs_truncated"):
        print("task output note: logs were truncated by the Jobs API")
    if not printed_output:
        print("task output: <none>")


def submit_and_wait(client: Any, payload: Mapping[str, Any], poll_seconds: float = 10) -> int:
    """Submit, monitor, and report one run. Return zero only for terminal ``SUCCESS``."""
    if poll_seconds < 0:
        raise ValueError("poll_seconds cannot be negative")

    task_key = str(payload["tasks"][0]["task_key"])
    run_id = submit_run(client, payload)
    last_state = None
    printed_task_run_id = None

    while True:
        run = _get_run(client, run_id)
        task = _task_for_key(run, task_key)
        task_run_id = task.get("run_id")
        if task_run_id is not None and task_run_id != printed_task_run_id:
            print(f"task run id: {task_run_id}")
            printed_task_run_id = task_run_id

        state = run.get("state") or {}
        state_line = _state_line("parent", state)
        task_state_line = _state_line("task", task.get("state") or {}) if task else ""
        current_state = (state_line, task_state_line)
        if current_state != last_state:
            print(state_line)
            if task_state_line:
                print(task_state_line)
            last_state = current_state

        if str(state.get("life_cycle_state") or "").upper() in TERMINAL_LIFE_CYCLE_STATES:
            break
        if poll_seconds:
            time.sleep(poll_seconds)

    output: Mapping[str, Any] = {}
    task = _task_for_key(run, task_key)
    task_run_id = task.get("run_id")
    if task_run_id is not None:
        try:
            output = _get_task_output(client, int(task_run_id))
        except Exception as exc:  # Preserve the terminal state when no task output was created.
            output = {
                "error": f"Jobs get-output was unavailable: {type(exc).__name__}: {exc}"
            }
    _print_final_details(run, task, output)

    task_result = (task.get("state") or {}).get("result_state") if task else None
    parent_result = state.get("result_state")
    return 0 if (task_result or parent_result) == "SUCCESS" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-environment-file", required=True)
    parser.add_argument("--code-source-path", required=True)
    parser.add_argument("--command-path", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--mlflow-experiment-directory", required=True)
    parser.add_argument("--mlflow-run", required=True)
    parser.add_argument("--accelerator-type", default="GPU_1xA10")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--task-key", default="training")
    parser.add_argument("--idempotency-token", default="")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--profile", default="", help="Optional local Databricks profile")
    parser.add_argument("--no-wait", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    payload = build_payload(
        jobs_environment_file=args.jobs_environment_file,
        code_source_path=args.code_source_path,
        command_path=args.command_path,
        experiment=args.experiment,
        mlflow_experiment_directory=args.mlflow_experiment_directory,
        mlflow_run=args.mlflow_run,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        timeout_seconds=args.timeout_seconds,
        task_key=args.task_key,
        idempotency_token=args.idempotency_token,
    )
    if args.no_wait:
        submit_without_waiting(client, payload)
        return 0
    return submit_and_wait(client, payload, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
