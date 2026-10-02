"""Submit a wheelhouse-backed ``ai_runtime_task`` through the Databricks Jobs API.

This module intentionally uses ``WorkspaceClient.api_client.do`` instead of the typed Jobs SDK
models. AI Runtime task support can reach the REST API before it reaches an installed SDK version.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping
from uuid import uuid4


# BLOCKED is not terminal: Jobs moves a blocked run back to PENDING.
TERMINAL_LIFE_CYCLE_STATES = {"INTERNAL_ERROR", "SKIPPED", "TERMINATED"}
MAX_USAGE_POLICIES_PAGE_SIZE = 1000
CODE_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz")
CODE_ARCHIVE_EXCLUDED_NAMES = {".DS_Store", ".git", "__pycache__"}
REQUIREMENTS_YAML_NAME = "requirements.yaml"
HYPERPARAMETERS_YAML_NAME = "hyperparameters.yaml"
# Under the caller's /Workspace/Users/<name>; mirrors the AIR CLI's .air/cli_launch layout.
DEFAULT_LAUNCH_SUBDIRECTORY = ".air/jobs_launch"
# Jobs parameter names allow letters, digits, '_', '-', and '.'.
WORKLOAD_PARAMETER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
# Widgets of run_ai_runtime_job with their defaults; these are also the persisted launcher job's
# parameters. Keep in sync with the notebook's widget cell (a test compares them). A named
# workload parameter cannot reuse one of these names. The submission's idempotency token is not
# here: it is always auto-generated per run (see build_payload). Only the standalone script exposes
# an --idempotency-token override, for CI cross-invocation retry.
LAUNCHER_PARAMETER_DEFAULTS = {
    "jobs_environment_file": "",
    "code_source_path": "",
    "code_source_archive_path": "",
    "command_path": "",
    "experiment": "",
    "mlflow_experiment_directory": "",
    "mlflow_run": "",
    "usage_policy_name": "",
    "usage_policy_id": "",
    "accelerator_type": "GPU_1xA10",
    "accelerator_count": "1",
    "timeout_seconds": "600",
    "task_key": "training",
    "wait": "true",
    "poll_seconds": "10",
    "workload_parameters": "",
    "workload_parameter_names": "",
    "launch_root": "",
}
LAUNCHER_PARAMETER_NAMES = tuple(LAUNCHER_PARAMETER_DEFAULTS)


def _has_code_archive_suffix(path: str | Path) -> bool:
    return str(path).lower().endswith(CODE_ARCHIVE_SUFFIXES)


def _safe_archive_parts(name: str, *, field: str = "member") -> tuple[str, ...]:
    """Return an archive path's parts, rejecting ambiguous or unsafe paths."""
    trimmed = name.rstrip("/")
    raw_parts = trimmed.split("/") if trimmed else []
    if (
        not raw_parts
        or name.startswith("/")
        or "\\" in name
        or any(part in ("", ".", "..") for part in raw_parts)
    ):
        raise ValueError(f"code source archive has unsafe {field} path: {name!r}")
    parts = PurePosixPath(trimmed).parts
    if not parts:
        raise ValueError(f"code source archive has an empty {field} path")
    return parts


def _archive_top_level_component(archive: tarfile.TarFile, source_label: str) -> str:
    components: set[str] = set()
    root_files: list[str] = []
    content_members = 0
    members = archive.getmembers()
    if not members:
        raise ValueError(f"code source archive is empty: {source_label}")
    for member in members:
        parts = _safe_archive_parts(member.name)
        components.add(parts[0])
        if member.ischr() or member.isblk() or member.isfifo():
            raise ValueError(
                "code source archive contains an unsupported special file: "
                f"{member.name!r}"
            )
        if member.issym() or member.islnk():
            _safe_archive_parts(member.linkname, field="link target")
        if len(parts) == 1 and not member.isdir():
            root_files.append(member.name)
        elif len(parts) > 1 and not member.isdir():
            content_members += 1

    if root_files:
        example = next(
            (
                name
                for name in root_files
                if not PurePosixPath(name).name.startswith("._")
            ),
            root_files[0],
        )
        raise ValueError(
            "code source archive must contain one enclosing top-level directory; "
            f"found file at archive root: {example!r}. Package the directory itself "
            f"(for example, project/{PurePosixPath(example).name}), not only its contents"
        )
    if len(components) != 1:
        raise ValueError(
            "code source archive must contain exactly one top-level directory; "
            f"found {sorted(components)}"
        )
    if content_members == 0:
        raise ValueError(
            "code source archive contains no files below its top-level directory"
        )
    return next(iter(components))


def validate_code_source_archive(code_source_archive: str | Path) -> str:
    """Validate AIR's one-enclosing-directory contract and return that directory name.

    AIR extracts the archive and sets ``CODE_SOURCE_PATH`` to its single top-level component.
    A tarball containing files directly at its root therefore cannot be launched.
    """
    path = Path(code_source_archive)
    if not _has_code_archive_suffix(path):
        raise ValueError(
            "code_source_path must be a .tar.gz or .tgz archive; "
            "the runner notebook can package a directory"
        )
    if not path.is_file():
        raise ValueError(f"code source archive does not exist or is not a file: {path}")

    try:
        with tarfile.open(path, mode="r:gz") as archive:
            return _archive_top_level_component(archive, str(path))
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"cannot read code source archive {path}: {exc}") from exc


def validate_code_source_for_submission(client: Any, code_source_path: str | Path) -> str:
    """Validate a local, Workspace, or Volume archive before calling Jobs."""
    path_text = str(code_source_path)
    if not _has_code_archive_suffix(path_text):
        raise ValueError(
            "code_source_path must point to a .tar.gz or .tgz archive, not a Python "
            "script or directory. Put the script inside one enclosing directory in the "
            "archive; command_path selects the shell script that launches it"
        )

    local_path = Path(path_text)
    if local_path.is_file():
        return validate_code_source_archive(local_path)

    try:
        if path_text.startswith("/Workspace/"):
            workspace_path = path_text[len("/Workspace") :]
            stream = client.workspace.download(workspace_path)
        elif path_text.startswith("/Volumes/"):
            response = client.files.download(path_text)
            stream = response.contents
            if stream is None:
                raise RuntimeError("Files API returned no archive content")
        else:
            raise ValueError(
                "code_source_path must be a readable local path, /Workspace path, "
                "or /Volumes path"
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"code source archive does not exist or is not readable: {path_text}: {exc}"
        ) from exc

    try:
        with contextlib.closing(stream):
            with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                return _archive_top_level_component(archive, path_text)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"cannot read code source archive {path_text}: {exc}") from exc


def normalize_mlflow_experiment_fields(
    experiment: str,
    mlflow_experiment_directory: str,
) -> tuple[str, str, str]:
    """Validate MLflow's leaf-name/parent-directory split and return canonical values."""
    experiment_name = experiment.strip()
    if not experiment_name:
        raise ValueError("experiment must be a non-empty MLflow experiment leaf name")
    if experiment_name != experiment:
        raise ValueError("experiment must not have leading or trailing whitespace")
    if "/" in experiment_name or experiment_name in (".", ".."):
        raise ValueError(
            "experiment must be a leaf name, not a path; put only its existing parent "
            "Workspace directory in mlflow_experiment_directory"
        )

    directory = mlflow_experiment_directory.strip()
    if directory != mlflow_experiment_directory:
        raise ValueError(
            "mlflow_experiment_directory must not have leading or trailing whitespace"
        )
    if not (directory == "/Workspace" or directory.startswith("/Workspace/")):
        raise ValueError(
            "mlflow_experiment_directory must be an absolute parent path starting "
            "with /Workspace"
        )
    raw_parts = directory.split("/")[1:]
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(
            "mlflow_experiment_directory must be canonical: no trailing slash, repeated "
            "slashes, '.', or '..' components"
        )
    if PurePosixPath(directory).name == experiment_name:
        raise ValueError(
            "mlflow_experiment_directory appears to include the experiment name already; "
            f"pass its parent directory so the final path is not {directory}/{experiment_name}"
        )

    full_path = f"{directory}/{experiment_name}"
    return experiment_name, directory, full_path


def validate_mlflow_experiment_parent(
    client: Any,
    experiment: str,
    mlflow_experiment_directory: str,
) -> tuple[str, str, str]:
    """Require the MLflow experiment's parent to exist as a Workspace directory."""
    experiment_name, directory, full_path = normalize_mlflow_experiment_fields(
        experiment, mlflow_experiment_directory
    )
    workspace_api_path = "/" if directory == "/Workspace" else directory[len("/Workspace") :]
    try:
        status = client.api_client.do(
            method="GET",
            path="/api/2.0/workspace/get-status",
            query={"path": workspace_api_path},
        )
    except Exception as exc:
        raise ValueError(
            f"MLflow experiment parent does not exist or is not accessible: {directory}. "
            "Create that Workspace directory or choose an existing parent"
        ) from exc
    object_type = str(status.get("object_type") or "").upper()
    if object_type != "DIRECTORY":
        raise ValueError(
            "mlflow_experiment_directory must identify an existing Workspace directory; "
            f"{directory} is {object_type or 'an unknown object type'}"
        )
    return experiment_name, directory, full_path


def validate_submission_inputs(
    client: Any,
    *,
    code_source_path: str | Path,
    experiment: str,
    mlflow_experiment_directory: str,
) -> tuple[str, str, str, str]:
    """Run all remote-aware preflight checks required before a Jobs submission."""
    component = validate_code_source_for_submission(client, code_source_path)
    experiment_name, directory, full_path = validate_mlflow_experiment_parent(
        client, experiment, mlflow_experiment_directory
    )
    return component, experiment_name, directory, full_path


def warn_on_colocated_requirements_yaml(client: Any, command_path: str) -> bool:
    """Note a ``requirements.yaml`` next to ``command_path``; return whether one exists.

    The AI Runtime launcher would install ``<command dir>/requirements.yaml`` before the
    environment spec. In this workflow dependencies come only from the wheelhouse
    ``environments[].spec``, and every submission runs the command from a staged copy (see
    ``stage_launch_directory``), so a ``requirements.yaml`` beside the developer's original
    script is never beside the running command and never installed. A developer who placed one
    expecting the AIR CLI's behavior would otherwise be surprised, so warn.
    """
    if not command_path.startswith("/Workspace/"):
        return False
    sibling = str(PurePosixPath(command_path).parent / REQUIREMENTS_YAML_NAME)
    try:
        client.api_client.do(
            method="GET",
            path="/api/2.0/workspace/get-status",
            query={"path": sibling[len("/Workspace") :]},
        )
    except Exception as exc:  # Absence is the expected case; this check never blocks submission.
        if getattr(exc, "error_code", "") != "RESOURCE_DOES_NOT_EXIST":
            print(f"WARNING: could not check for {sibling}: {type(exc).__name__}: {exc}")
        return False
    print(
        f"WARNING: {sibling} exists but is not used. This workflow installs dependencies only "
        "from the wheelhouse environment spec, and the command runs from a staged copy, so this "
        "file is ignored. Remove it to avoid confusion."
    )
    return True


def parse_workload_parameter_names(names: str) -> list[str]:
    """Parse the comma-separated job parameter names forwarded to the workload."""
    parsed: list[str] = []
    for raw_name in names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if not WORKLOAD_PARAMETER_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"workload parameter name {name!r} must start with a letter or '_' and "
                "contain only letters, digits, '_', '-', or '.'"
            )
        if name in LAUNCHER_PARAMETER_NAMES:
            raise ValueError(
                f"workload parameter name {name!r} is reserved for the launcher; rename it"
            )
        if name in parsed:
            raise ValueError(f"workload parameter name {name!r} is listed twice")
        parsed.append(name)
    return parsed


def parse_workload_parameters(
    parameters_json: str = "",
    named_values: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge a JSON-object string with per-name string values into one parameter mapping.

    The JSON object carries typed or nested values. ``named_values`` carries individual Jobs
    parameters, which Jobs always delivers as strings. A name set in both places is rejected
    rather than silently resolved.
    """
    parameters: dict[str, Any] = {}
    text = parameters_json.strip()
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"workload_parameters must be a JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"workload_parameters must be a JSON object, not {type(parsed).__name__}"
            )
        parameters.update(parsed)
    for name, value in (named_values or {}).items():
        if name in parameters:
            raise ValueError(
                f"workload parameter {name!r} is set both in workload_parameters and as a "
                "named parameter; set it in one place"
            )
        parameters[name] = value
    return parameters


def render_hyperparameters_yaml(parameters: Mapping[str, Any]) -> str:
    """Render workload parameters as the ``hyperparameters.yaml`` AIR exposes to the command.

    JSON is valid YAML, so ``yaml.safe_load`` and OmegaConf read this file unchanged, and this
    module needs no PyYAML dependency. NaN and infinity have no portable YAML spelling.
    """
    try:
        return json.dumps(dict(parameters), indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as exc:
        raise ValueError(f"workload parameters must be finite JSON values: {exc}") from exc


def _launch_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned[:64] or "run"


def default_launch_root(client: Any) -> str:
    """Return ``/Workspace/Users/<caller>/.air/jobs_launch`` for the authenticated identity."""
    me = client.api_client.do(method="GET", path="/api/2.0/preview/scim/v2/Me")
    user_name = str(me.get("userName") or "").strip()
    if not user_name:
        raise ValueError("cannot determine the caller's workspace user name; set launch_root")
    return f"/Workspace/Users/{user_name}/{DEFAULT_LAUNCH_SUBDIRECTORY}"


def stage_launch_directory(
    client: Any,
    *,
    command_path: str,
    experiment: str,
    mlflow_run: str,
    parameters: Mapping[str, Any] | None = None,
    launch_root: str = "",
) -> tuple[str, str | None]:
    """Copy ``command_path`` into a fresh per-run launch directory; return the staged paths.

    Every submission stages, mirroring the AIR CLI, which always uploads the command into a
    per-run ``.air/cli_launch`` directory and never runs it in place. Running from a controlled
    directory makes dependency handling uniform across entry points: a ``requirements.yaml`` or
    ``env_vars.json`` sitting next to the developer's original script is never beside the running
    command, so it is never installed, whether or not parameters are set. When ``parameters`` is
    non-empty a ``hyperparameters.yaml`` is written alongside the command, and the launcher
    exports ``HYPERPARAMETERS_PATH`` to it — the AIR parameter channel, since ``ai_runtime_task``
    has no Jobs-parameter field (``dynamic_value_refs`` is unwired). Returns the staged command
    path and the hyperparameters path, or ``None`` for the latter when no parameters were set.

    Staging always writes, before the submit. Reusing a stable ``idempotency_token`` (only the
    standalone script's ``--idempotency-token`` can) therefore still creates a fresh launch dir,
    but Jobs returns the pre-existing run pointing at the earlier staged command, so the new files
    go unused and a changed parameter or command does not take effect. Omit it (the default) to
    apply changes.
    """
    if not command_path.startswith("/Workspace/"):
        raise ValueError("command_path must be a /Workspace path to stage the launch directory")
    root = launch_root.strip().rstrip("/") or default_launch_root(client)
    if not root.startswith("/Workspace/") or any(
        part in ("", ".", "..") for part in root.split("/")[1:]
    ):
        raise ValueError(f"launch_root must be a canonical /Workspace directory: {root!r}")
    parameters = dict(parameters or {})
    # Render before any workspace write so a bad parameter value fails before staging.
    hyperparameters_text = render_hyperparameters_yaml(parameters) if parameters else None

    try:
        stream = client.workspace.download(command_path[len("/Workspace") :])
        with contextlib.closing(stream):
            command_content = stream.read()
    except Exception as exc:
        raise ValueError(
            f"command script does not exist or is not readable: {command_path}: {exc}"
        ) from exc

    directory = (
        f"{root}/{_launch_path_component(experiment)}/"
        f"{_launch_path_component(mlflow_run)}_{uuid4().hex[:16]}"
    )
    staged_command_path = f"{directory}/{PurePosixPath(command_path).name}"
    staged: list[tuple[str, bytes]] = [(staged_command_path, command_content)]
    hyperparameters_path: str | None = None
    if hyperparameters_text is not None:
        hyperparameters_path = f"{directory}/{HYPERPARAMETERS_YAML_NAME}"
        staged.append((hyperparameters_path, hyperparameters_text.encode("utf-8")))

    client.api_client.do(
        method="POST",
        path="/api/2.0/workspace/mkdirs",
        body={"path": directory[len("/Workspace") :]},
    )
    for path, content in staged:
        client.api_client.do(
            method="POST",
            path="/api/2.0/workspace/import",
            body={
                "path": path[len("/Workspace") :],
                "format": "AUTO",
                "content": base64.b64encode(content).decode("ascii"),
                "overwrite": False,
            },
        )
    return staged_command_path, hyperparameters_path


def create_code_source_archive(
    code_source_directory: str | Path,
    output_archive: str | Path = "",
) -> tuple[str, str]:
    """Package a directory in AIR's required ``directory_name/...`` archive layout."""
    source = Path(code_source_directory)
    if not source.is_dir():
        raise ValueError(f"code source directory does not exist: {source}")
    if source.name in ("", ".", ".."):
        raise ValueError(f"code source directory must have a usable name: {source}")

    output = (
        Path(output_archive)
        if str(output_archive).strip()
        else source.with_name(f"{source.name}.tar.gz")
    )
    if not _has_code_archive_suffix(output):
        raise ValueError("code source archive output must end in .tar.gz or .tgz")
    if not output.parent.is_dir():
        raise ValueError(f"code source archive parent directory does not exist: {output.parent}")

    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise ValueError("code source archive output must be outside the source directory")

    temporary = output.with_name(f".{uuid4().hex}.{output.name}")

    def archive_filter(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = PurePosixPath(member.name).parts
        if any(part in CODE_ARCHIVE_EXCLUDED_NAMES for part in parts):
            return None
        if any(part.startswith("._") for part in parts):
            return None
        if member.name.endswith((".pyc", ".pyo")):
            return None
        return member

    try:
        with tarfile.open(temporary, mode="w:gz") as archive:
            archive.add(source, arcname=source.name, recursive=True, filter=archive_filter)
        component = validate_code_source_archive(temporary)
        os.replace(temporary, output)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return str(output), component


def prepare_code_source_archive(
    code_source_path: str | Path,
    output_archive: str | Path = "",
) -> tuple[str, str, bool]:
    """Validate an archive or package a source directory for notebook submission."""
    source = Path(code_source_path)
    if source.is_dir():
        archive_path, component = create_code_source_archive(source, output_archive)
        return archive_path, component, True
    if str(output_archive).strip():
        raise ValueError(
            "code_source_archive_path is only used when code_source_path is a directory"
        )
    return str(source), validate_code_source_archive(source), False


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


def list_usage_policies(client: Any) -> list[dict[str, Any]]:
    """Return every serverless usage policy visible to the workspace identity."""
    policies: list[dict[str, Any]] = []
    page_token = ""
    while True:
        query: dict[str, Any] = {"page_size": MAX_USAGE_POLICIES_PAGE_SIZE}
        if page_token:
            query["page_token"] = page_token
        response = client.api_client.do(
            method="GET",
            path="/api/2.0/serverless-policies",
            query=query,
        )
        policies.extend(response.get("policies") or [])
        page_token = str(response.get("next_page_token") or "")
        if not page_token:
            return policies


def resolve_usage_policy_id(
    client: Any,
    *,
    usage_policy_name: str = "",
    usage_policy_id: str = "",
) -> str:
    """Resolve exactly one usage-policy name or ID to the ID required by Jobs."""
    name = usage_policy_name.strip()
    policy_id = usage_policy_id.strip()
    if bool(name) == bool(policy_id):
        raise ValueError("set exactly one of usage_policy_name or usage_policy_id")
    if policy_id:
        return policy_id

    policies = list_usage_policies(client)
    matches = [
        policy
        for policy in policies
        if str(policy.get("policy_name") or "").strip().casefold() == name.casefold()
    ]
    if len(matches) == 1 and matches[0].get("policy_id"):
        return str(matches[0]["policy_id"])
    if len(matches) > 1:
        raise ValueError(
            f"multiple accessible usage policies are named {name!r}; select one by ID"
        )

    available = sorted(
        str(policy["policy_name"])
        for policy in policies
        if policy.get("policy_name")
    )
    suffix = f" Accessible policies: {available}." if available else ""
    raise ValueError(f"no accessible usage policy is named {name!r}.{suffix}")


def build_payload(
    jobs_environment_file: str | Path,
    code_source_path: str,
    command_path: str,
    experiment: str,
    mlflow_experiment_directory: str,
    mlflow_run: str,
    usage_policy_id: str,
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
        "usage_policy_id": usage_policy_id,
        "accelerator_type": accelerator_type,
        "task_key": task_key,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"required values are blank: {', '.join(missing)}")
    if not _has_code_archive_suffix(code_source_path):
        raise ValueError(
            "code_source_path sent to Jobs must be a .tar.gz or .tgz archive with "
            "one enclosing top-level directory"
        )
    experiment, mlflow_experiment_directory, _ = normalize_mlflow_experiment_fields(
        experiment, mlflow_experiment_directory
    )
    if accelerator_count < 1:
        raise ValueError("accelerator_count must be at least 1")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least 1")

    environment_spec = load_environment_spec(jobs_environment_file)
    environment_key = "wheelhouse"
    payload: dict[str, Any] = {
        "run_name": f"{task_key}-{mlflow_run}",
        "usage_policy_id": usage_policy_id.strip(),
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
    # The SDK re-sends a POST after a timeout or dropped connection. Jobs returns the existing
    # run for a repeated token, so a per-invocation default keeps those retries from launching
    # a second GPU run (the AIR CLI does the same). Pass a stable token to dedupe across reruns.
    payload["idempotency_token"] = idempotency_token.strip() or str(uuid4())
    return payload


def with_command_path(payload: Mapping[str, Any], command_path: str) -> dict[str, Any]:
    """Return a copy of a built payload whose single deployment runs ``command_path``.

    Build the payload first so every input is validated before ``stage_launch_directory``
    writes to the workspace, then point it at the staged copy.
    """
    updated = json.loads(json.dumps(payload))
    (deployment,) = updated["tasks"][0]["ai_runtime_task"]["deployments"]
    deployment["command_path"] = command_path
    return updated


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
    parser.add_argument(
        "--code-source-path",
        required=True,
        help=(
            "Workspace/Volume .tar.gz or .tgz archive containing exactly one enclosing "
            "top-level directory"
        ),
    )
    parser.add_argument("--command-path", required=True)
    parser.add_argument(
        "--experiment",
        required=True,
        help="MLflow experiment name only, without a workspace path",
    )
    parser.add_argument(
        "--mlflow-experiment-directory",
        required=True,
        help="Parent workspace directory; experiment is appended to this path",
    )
    parser.add_argument("--mlflow-run", required=True, help="MLflow run display name")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument(
        "--usage-policy-name",
        default="",
        help="Exact accessible usage-policy name; resolved to its ID before submission",
    )
    policy.add_argument(
        "--usage-policy-id",
        default="",
        help="Usage-policy UUID from Compute > Usage policies",
    )
    parser.add_argument("--accelerator-type", default="GPU_1xA10")
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--task-key", default="training")
    parser.add_argument(
        "--idempotency-token",
        default="",
        help=(
            "CI safe-retry key: a reused token returns the existing run instead of launching a "
            "duplicate. Blank (the default) generates one per invocation. A reused token ignores "
            "changed parameters/command (the staged dir is unused)"
        ),
    )
    parser.add_argument(
        "--workload-parameters",
        default="",
        help="JSON object written to the command's hyperparameters.yaml",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="String workload parameter; repeatable, merged with --workload-parameters",
    )
    parser.add_argument(
        "--launch-root",
        default="",
        help=(
            "Workspace directory for the per-run launch dir each submission stages "
            f"(default /Workspace/Users/<caller>/{DEFAULT_LAUNCH_SUBDIRECTORY})"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--profile", default="", help="Optional local Databricks profile")
    parser.add_argument("--no-wait", action="store_true")
    return parser


def parse_param_arguments(params: list[str]) -> dict[str, str]:
    """Parse repeated ``NAME=VALUE`` arguments into named workload parameters."""
    named: dict[str, str] = {}
    for param in params:
        name, separator, value = param.partition("=")
        if not separator or not name.strip() or "," in name:
            raise ValueError(f"--param must be NAME=VALUE, got {param!r}")
        name = parse_workload_parameter_names(name)[0]
        if name in named:
            raise ValueError(f"--param {name!r} is set twice")
        named[name] = value
    return named


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parameters = parse_workload_parameters(
        args.workload_parameters, parse_param_arguments(args.param)
    )
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()

    component, experiment, experiment_directory, experiment_path = (
        validate_submission_inputs(
            client,
            code_source_path=args.code_source_path,
            experiment=args.experiment,
            mlflow_experiment_directory=args.mlflow_experiment_directory,
        )
    )
    # The command runs from a staged copy, so a requirements.yaml next to the original is ignored.
    warn_on_colocated_requirements_yaml(client, args.command_path)
    usage_policy_id = resolve_usage_policy_id(
        client,
        usage_policy_name=args.usage_policy_name,
        usage_policy_id=args.usage_policy_id,
    )
    payload = build_payload(
        jobs_environment_file=args.jobs_environment_file,
        code_source_path=args.code_source_path,
        command_path=args.command_path,
        experiment=experiment,
        mlflow_experiment_directory=experiment_directory,
        mlflow_run=args.mlflow_run,
        usage_policy_id=usage_policy_id,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        timeout_seconds=args.timeout_seconds,
        task_key=args.task_key,
        idempotency_token=args.idempotency_token,
    )
    # Staged after the payload is built, so an input error writes nothing to the workspace.
    command_path, hyperparameters_path = stage_launch_directory(
        client,
        command_path=args.command_path,
        experiment=experiment,
        mlflow_run=args.mlflow_run,
        parameters=parameters,
        launch_root=args.launch_root,
    )
    payload = with_command_path(payload, command_path)
    print(f"staged command: {command_path}")
    if hyperparameters_path:
        print(f"HYPERPARAMETERS_PATH: {hyperparameters_path} ({len(parameters)} parameters)")
    print(f"code archive root: {component}")
    print(f"MLflow experiment: {experiment_path}")
    if args.no_wait:
        submit_without_waiting(client, payload)
        return 0
    return submit_and_wait(client, payload, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
