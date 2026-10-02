"""Create or reset a persisted Jobs job that launches a parameterized ``ai_runtime_task``.

``ai_runtime_task`` cannot read Jobs parameters (``dynamic_value_refs`` is unwired), so the
persisted job's only task is the ``run_ai_runtime_job`` notebook. Jobs maps each job parameter
onto the notebook widget of the same name; the notebook forwards workload parameters through
``hyperparameters.yaml`` and submits the AIR run with ``jobs/runs/submit``. The launcher task
uses no managed base environment, so the workspace's base-environment preview does not apply.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_submitter():
    path = Path(__file__).resolve().with_name("submit_ai_runtime_job.py")
    spec = importlib.util.spec_from_file_location("air_wheelhouse_job_submitter", path)
    assert spec and spec.loader, f"cannot load submitter script: {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submitter = _load_submitter()


def build_job_settings(
    *,
    name: str,
    notebook_path: str,
    launcher_defaults: dict[str, str],
    named_parameters: dict[str, str],
    existing_cluster_id: str = "",
    serverless: bool = False,
    max_concurrent_runs: int = 1,
) -> dict[str, Any]:
    """Build ``JobSettings`` for the launcher job; every widget value becomes a job parameter.

    A serverless launcher also needs a top-level ``usage_policy_id``; ``main`` resolves it.
    """
    if not name.strip():
        raise ValueError("job name is blank")
    if not notebook_path.startswith("/"):
        raise ValueError("notebook_path must be an absolute workspace path")
    if bool(existing_cluster_id.strip()) == serverless:
        raise ValueError("set exactly one of existing_cluster_id or serverless")
    if max_concurrent_runs < 1:
        raise ValueError("max_concurrent_runs must be at least 1")

    unknown = set(launcher_defaults) - set(submitter.LAUNCHER_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"not launcher job parameters: {sorted(unknown)}")
    required = (
        "jobs_environment_file",
        "code_source_path",
        "command_path",
        "experiment",
        "mlflow_experiment_directory",
        "mlflow_run",
    )
    missing = [key for key in required if not (launcher_defaults.get(key) or "").strip()]
    if missing:
        raise ValueError(f"required launcher defaults are blank: {', '.join(missing)}")
    if bool((launcher_defaults.get("usage_policy_name") or "").strip()) == bool(
        (launcher_defaults.get("usage_policy_id") or "").strip()
    ):
        raise ValueError("set exactly one of usage_policy_name or usage_policy_id")
    submitter.normalize_mlflow_experiment_fields(
        launcher_defaults["experiment"], launcher_defaults["mlflow_experiment_directory"]
    )

    names = submitter.parse_workload_parameter_names(",".join(named_parameters))
    # Run the notebook's own merge now so a conflicting default fails here, not on first run.
    submitter.parse_workload_parameters(
        launcher_defaults.get("workload_parameters", ""), named_parameters
    )
    # Every launcher widget becomes a job parameter so run-now can override any of them. The
    # submission's idempotency token is not among them: the notebook always generates one per run.
    defaults = dict(submitter.LAUNCHER_PARAMETER_DEFAULTS)
    defaults.update(launcher_defaults)
    defaults["workload_parameter_names"] = ",".join(names)

    task: dict[str, Any] = {
        "task_key": "launch_ai_runtime_task",
        "notebook_task": {"notebook_path": notebook_path, "source": "WORKSPACE"},
        # A retry would submit a second GPU run; the AIR task's own outcome fails this task.
        "max_retries": 0,
    }
    if existing_cluster_id.strip():
        task["existing_cluster_id"] = existing_cluster_id.strip()

    settings: dict[str, Any] = {
        "name": name.strip(),
        "max_concurrent_runs": max_concurrent_runs,
        "tasks": [task],
        "parameters": [{"name": key, "default": value} for key, value in defaults.items()]
        + [{"name": key, "default": named_parameters[key]} for key in names],
    }
    return settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Persisted launcher job name")
    parser.add_argument(
        "--notebook-path",
        required=True,
        help="Workspace path of the deployed run_ai_runtime_job notebook",
    )
    compute = parser.add_mutually_exclusive_group(required=True)
    compute.add_argument("--existing-cluster-id", default="")
    compute.add_argument(
        "--serverless",
        action="store_true",
        help="Run the launcher notebook on serverless; sets the job's usage_policy_id",
    )
    parser.add_argument("--job-id", type=int, help="Reset this existing job instead of creating")
    parser.add_argument("--max-concurrent-runs", type=int, default=1)
    parser.add_argument("--profile", default="", help="Optional local Databricks profile")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the job settings without calling Jobs"
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=DEFAULT",
        help="Named workload parameter exposed as its own job parameter; repeatable",
    )
    launcher = parser.add_argument_group("launcher job parameter defaults")
    for key, default in submitter.LAUNCHER_PARAMETER_DEFAULTS.items():
        if key == "workload_parameter_names":
            continue
        launcher.add_argument(
            f"--{key.replace('_', '-')}",
            dest=key,
            default=None,
            help=f"default {default!r}" if default else None,
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launcher_defaults = {
        key: value
        for key in submitter.LAUNCHER_PARAMETER_NAMES
        if (value := getattr(args, key, None)) is not None
    }
    named_parameters = submitter.parse_param_arguments(args.param)

    settings = build_job_settings(
        name=args.name,
        notebook_path=args.notebook_path,
        launcher_defaults=launcher_defaults,
        named_parameters=named_parameters,
        existing_cluster_id=args.existing_cluster_id,
        serverless=args.serverless,
        max_concurrent_runs=args.max_concurrent_runs,
    )
    usage_policy_name = (launcher_defaults.get("usage_policy_name") or "").strip()
    usage_policy_id = (launcher_defaults.get("usage_policy_id") or "").strip()
    if args.dry_run:
        if args.serverless and usage_policy_id:
            settings["usage_policy_id"] = usage_policy_id
        elif args.serverless:
            print(f"dry run: usage_policy_id is resolved from {usage_policy_name!r} on create")
        print(json.dumps(settings, indent=2))
        return 0

    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    if args.serverless:
        # Serverless launcher compute bills to the same policy as the AIR run it submits.
        settings["usage_policy_id"] = submitter.resolve_usage_policy_id(
            client, usage_policy_name=usage_policy_name, usage_policy_id=usage_policy_id
        )

    if args.job_id is None:
        response = client.api_client.do(method="POST", path="/api/2.2/jobs/create", body=settings)
        job_id = response.get("job_id")
        if job_id is None:
            raise RuntimeError(f"Jobs create response did not contain job_id: {response!r}")
        print(f"created launcher job: {job_id}")
    else:
        client.api_client.do(
            method="POST",
            path="/api/2.2/jobs/reset",
            body={"job_id": args.job_id, "new_settings": settings},
        )
        job_id = args.job_id
        print(f"reset launcher job: {job_id}")
    print(f"job parameters: {', '.join(item['name'] for item in settings['parameters'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
