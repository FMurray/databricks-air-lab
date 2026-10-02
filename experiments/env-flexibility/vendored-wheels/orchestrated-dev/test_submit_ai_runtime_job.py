import base64
import contextlib
import importlib.util
import io
import json
import re
import tarfile
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
# The runnable modules live in the sibling minimal bundle; tests stay out of it.
BUNDLE = HERE.parent / "orchestrated"
SPEC = importlib.util.spec_from_file_location(
    "air_wheelhouse_job_submitter", BUNDLE / "submit_ai_runtime_job.py"
)
submitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submitter)


class FakeApiClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def do(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeNotFound(Exception):
    error_code = "RESOURCE_DOES_NOT_EXIST"


class FakeWorkspaceClient:
    def __init__(self, responses):
        self.api_client = FakeApiClient(responses)


class FakeWorkspaceDownload:
    def __init__(self, contents):
        self.contents = contents
        self.paths = []

    def download(self, path):
        self.paths.append(path)
        self.contents.seek(0)
        return self.contents


class FakeMissingDownload:
    def download(self, path):
        raise FakeNotFound(f"missing: {path}")


class SubmitAiRuntimeJobTest(unittest.TestCase):
    def _environment_file(self, spec):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "jobs-environment.json"
        path.write_text(json.dumps(spec))
        self.addCleanup(temporary.cleanup)
        return path

    def _payload(self):
        environment_file = self._environment_file(
            {
                "base_environment": "workspace-base-environments/databricks_ai_v5",
                "dependencies": [
                    "--no-index",
                    "--find-links /Volumes/catalog/schema/wheels/wheelhouse",
                ],
            }
        )
        return submitter.build_payload(
            jobs_environment_file=environment_file,
            code_source_path="/Workspace/Shared/probe.tgz",
            command_path="/Workspace/Shared/run_probe.sh",
            experiment="wheelhouse-probe",
            mlflow_experiment_directory="/Workspace/Shared",
            mlflow_run="test-run",
            usage_policy_id="policy-123",
            idempotency_token="stable-token",
        )

    def _temporary_directory(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def _valid_code_archive(self):
        root = self._temporary_directory()
        project = root / "wheelhouse-probe"
        project.mkdir()
        (project / "verify_environment.py").write_text("print('ok')\n")
        archive_path = root / "probe.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(project, arcname=project.name)
        return archive_path

    def test_build_payload_embeds_generated_environment_spec(self):
        payload = self._payload()

        self.assertEqual(
            payload["environments"][0]["spec"]["base_environment"],
            "workspace-base-environments/databricks_ai_v5",
        )
        self.assertEqual(payload["tasks"][0]["environment_key"], "wheelhouse")
        self.assertEqual(
            payload["tasks"][0]["ai_runtime_task"]["deployments"][0]["compute"],
            {"accelerator_type": "GPU_1xA10", "accelerator_count": 1},
        )
        self.assertEqual(payload["idempotency_token"], "stable-token")
        self.assertEqual(payload["usage_policy_id"], "policy-123")

    def test_experiment_is_a_name_and_directory_is_its_workspace_parent(self):
        with self.assertRaisesRegex(ValueError, "experiment must be a leaf name"):
            submitter.build_payload(
                jobs_environment_file=self._environment_file(
                    {"environment_version": "5", "dependencies": []}
                ),
                code_source_path="/Workspace/Shared/probe.tgz",
                command_path="/Workspace/Shared/run_probe.sh",
                experiment="/Workspace/Shared/full-path",
                mlflow_experiment_directory="/Workspace/Shared",
                mlflow_run="test-run",
                usage_policy_id="policy-123",
            )

    def test_usage_policy_name_resolves_to_id(self):
        client = FakeWorkspaceClient(
            [
                {
                    "policies": [
                        {"policy_name": "Other team", "policy_id": "other-id"},
                        {"policy_name": "AIR Team", "policy_id": "air-id"},
                    ]
                }
            ]
        )

        policy_id = submitter.resolve_usage_policy_id(
            client, usage_policy_name="air team"
        )

        self.assertEqual(policy_id, "air-id")
        self.assertEqual(
            client.api_client.calls,
            [
                {
                    "method": "GET",
                    "path": "/api/2.0/serverless-policies",
                    "query": {"page_size": 1000},
                }
            ],
        )

    def test_usage_policy_requires_exactly_one_name_or_id(self):
        client = FakeWorkspaceClient([])
        for name, policy_id in (("", ""), ("AIR Team", "air-id")):
            with self.subTest(name=name, policy_id=policy_id):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    submitter.resolve_usage_policy_id(
                        client,
                        usage_policy_name=name,
                        usage_policy_id=policy_id,
                    )

    def test_environment_requires_exactly_one_selector(self):
        for environment_spec in (
            {"dependencies": []},
            {
                "base_environment": "workspace-base-environments/databricks_ai_v5",
                "environment_version": "5",
                "dependencies": [],
            },
        ):
            with self.subTest(environment_spec=environment_spec):
                path = self._environment_file(environment_spec)
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    submitter.load_environment_spec(path)

    def test_environment_dependencies_must_be_strings(self):
        path = self._environment_file(
            {"environment_version": "5", "dependencies": ["--no-index", 42]}
        )

        with self.assertRaisesRegex(ValueError, "list of strings"):
            submitter.load_environment_spec(path)

    def test_valid_code_source_archive_has_one_enclosing_directory(self):
        archive_path = self._valid_code_archive()

        component = submitter.validate_code_source_archive(archive_path)

        self.assertEqual(component, "wheelhouse-probe")

    def test_python_script_is_rejected_as_code_source_with_actionable_message(self):
        client = FakeWorkspaceClient([])

        with self.assertRaisesRegex(
            ValueError, "must point to a .tar.gz or .tgz archive, not a Python script"
        ):
            submitter.validate_code_source_for_submission(
                client, "/Workspace/Shared/verify_environment.py"
            )

        self.assertEqual(client.api_client.calls, [])

    def test_workspace_archive_is_downloaded_and_validated(self):
        archive_bytes = io.BytesIO()
        data = b"print('ok')\n"
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            directory = tarfile.TarInfo("wheelhouse-probe")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            member = tarfile.TarInfo("wheelhouse-probe/verify_environment.py")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        client = FakeWorkspaceClient([])
        client.workspace = FakeWorkspaceDownload(archive_bytes)

        component = submitter.validate_code_source_for_submission(
            client, "/Workspace/Shared/probe.tgz"
        )

        self.assertEqual(component, "wheelhouse-probe")
        self.assertEqual(client.workspace.paths, ["/Shared/probe.tgz"])

    def test_submission_preflight_requires_existing_mlflow_parent_directory(self):
        archive_path = self._valid_code_archive()
        client = FakeWorkspaceClient([{"object_type": "DIRECTORY"}])

        result = submitter.validate_submission_inputs(
            client,
            code_source_path=archive_path,
            experiment="wheelhouse-probe",
            mlflow_experiment_directory="/Workspace/Shared/air-experiments",
        )

        self.assertEqual(
            result,
            (
                "wheelhouse-probe",
                "wheelhouse-probe",
                "/Workspace/Shared/air-experiments",
                "/Workspace/Shared/air-experiments/wheelhouse-probe",
            ),
        )
        self.assertEqual(
            client.api_client.calls,
            [
                {
                    "method": "GET",
                    "path": "/api/2.0/workspace/get-status",
                    "query": {"path": "/Shared/air-experiments"},
                }
            ],
        )

    def test_mlflow_parent_rejects_non_directory_workspace_object(self):
        client = FakeWorkspaceClient([{"object_type": "NOTEBOOK"}])

        with self.assertRaisesRegex(ValueError, "is NOTEBOOK"):
            submitter.validate_mlflow_experiment_parent(
                client,
                "wheelhouse-probe",
                "/Workspace/Shared/not-a-directory",
            )

    def test_mlflow_parent_must_exist_and_be_accessible(self):
        client = FakeWorkspaceClient([])

        with self.assertRaisesRegex(
            ValueError, "parent does not exist or is not accessible"
        ):
            submitter.validate_mlflow_experiment_parent(
                client,
                "wheelhouse-probe",
                "/Workspace/Shared/missing-directory",
            )

    def test_mlflow_parent_rejects_noncanonical_path(self):
        client = FakeWorkspaceClient([])

        with self.assertRaisesRegex(ValueError, "must be canonical"):
            submitter.validate_mlflow_experiment_parent(
                client,
                "wheelhouse-probe",
                "/Workspace/Shared/air-experiments/",
            )

        self.assertEqual(client.api_client.calls, [])

    def test_mlflow_parent_must_not_repeat_experiment_leaf(self):
        client = FakeWorkspaceClient([])

        with self.assertRaisesRegex(ValueError, "include the experiment name already"):
            submitter.validate_mlflow_experiment_parent(
                client,
                "wheelhouse-probe",
                "/Workspace/Shared/wheelhouse-probe",
            )

        self.assertEqual(client.api_client.calls, [])

    def test_code_source_archive_rejects_file_at_root(self):
        root = self._temporary_directory()
        source = root / "verify_environment.py"
        source.write_text("print('ok')\n")
        archive_path = root / "probe.tgz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(source, arcname=source.name)

        with self.assertRaisesRegex(ValueError, "found file at archive root"):
            submitter.validate_code_source_archive(archive_path)

    def test_code_source_archive_rejects_multiple_top_level_directories(self):
        root = self._temporary_directory()
        archive_path = root / "probe.tar.gz"
        for name in ("one", "two"):
            directory = root / name
            directory.mkdir()
            (directory / "main.py").write_text("print('ok')\n")
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(root / "one", arcname="one")
            archive.add(root / "two", arcname="two")

        with self.assertRaisesRegex(ValueError, "exactly one top-level directory"):
            submitter.validate_code_source_archive(archive_path)

    def test_code_source_archive_rejects_unsafe_member_path(self):
        root = self._temporary_directory()
        archive_path = root / "probe.tgz"
        data = b"print('unsafe')\n"
        member = tarfile.TarInfo("../escape.py")
        member.size = len(data)
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.addfile(member, io.BytesIO(data))

        with self.assertRaisesRegex(ValueError, "unsafe member path"):
            submitter.validate_code_source_archive(archive_path)

    def test_directory_is_packaged_with_its_name_as_archive_root(self):
        root = self._temporary_directory()
        project = root / "wheelhouse-probe"
        project.mkdir()
        (project / "verify_environment.py").write_text("print('ok')\n")
        cache = project / "__pycache__"
        cache.mkdir()
        (cache / "verify_environment.pyc").write_bytes(b"ignored")
        (project / "._verify_environment.py").write_bytes(b"ignored AppleDouble metadata")
        output = root / "prepared.tgz"

        archive_path, component, created = submitter.prepare_code_source_archive(
            project, output
        )

        self.assertTrue(created)
        self.assertEqual(Path(archive_path), output)
        self.assertEqual(component, "wheelhouse-probe")
        with tarfile.open(output, "r:gz") as archive:
            members = archive.getnames()
        self.assertIn("wheelhouse-probe/verify_environment.py", members)
        self.assertFalse(any("__pycache__" in name for name in members))
        self.assertFalse(any(Path(name).name.startswith("._") for name in members))

    def test_blank_idempotency_token_is_generated_per_payload(self):
        environment_file = self._environment_file(
            {"environment_version": "5", "dependencies": []}
        )
        arguments = dict(
            jobs_environment_file=environment_file,
            code_source_path="/Workspace/Shared/probe.tgz",
            command_path="/Workspace/Shared/run_probe.sh",
            experiment="wheelhouse-probe",
            mlflow_experiment_directory="/Workspace/Shared",
            mlflow_run="test-run",
            usage_policy_id="policy-123",
        )

        first = submitter.build_payload(**arguments)["idempotency_token"]
        second = submitter.build_payload(**arguments, idempotency_token="  ")["idempotency_token"]

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertNotEqual(first, second)

    def test_colocated_requirements_yaml_is_reported(self):
        client = FakeWorkspaceClient([{"object_type": "FILE"}])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            found = submitter.warn_on_colocated_requirements_yaml(
                client, "/Workspace/Shared/probe/run_probe.sh"
            )

        self.assertTrue(found)
        self.assertIn("/Workspace/Shared/probe/requirements.yaml exists", output.getvalue())
        self.assertEqual(
            client.api_client.calls[0]["query"], {"path": "/Shared/probe/requirements.yaml"}
        )

    def test_missing_colocated_requirements_yaml_is_silent(self):
        client = FakeWorkspaceClient([FakeNotFound("not found")])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            found = submitter.warn_on_colocated_requirements_yaml(
                client, "/Workspace/Shared/probe/run_probe.sh"
            )

        self.assertFalse(found)
        self.assertEqual(output.getvalue(), "")

    def test_colocated_check_failure_warns_without_blocking(self):
        client = FakeWorkspaceClient([PermissionError("denied")])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            found = submitter.warn_on_colocated_requirements_yaml(
                client, "/Workspace/Shared/probe/run_probe.sh"
            )

        self.assertFalse(found)
        self.assertIn("could not check", output.getvalue())

    def test_colocated_check_skips_non_workspace_command_path(self):
        client = FakeWorkspaceClient([])

        found = submitter.warn_on_colocated_requirements_yaml(client, "/Volumes/c/s/v/run.sh")

        self.assertFalse(found)
        self.assertEqual(client.api_client.calls, [])

    def test_payload_requires_a_gzip_tar_code_source_path(self):
        with self.assertRaisesRegex(ValueError, "must be a .tar.gz or .tgz"):
            submitter.build_payload(
                jobs_environment_file=self._environment_file(
                    {"environment_version": "5", "dependencies": []}
                ),
                code_source_path="/Workspace/Shared/source-directory",
                command_path="/Workspace/Shared/run_probe.sh",
                experiment="wheelhouse-probe",
                mlflow_experiment_directory="/Workspace/Shared",
                mlflow_run="test-run",
                usage_policy_id="policy-123",
            )

    def test_submit_and_wait_reports_success_and_uses_raw_jobs_endpoints(self):
        payload = self._payload()
        running = {
            "run_id": 100,
            "run_page_url": "https://workspace/#job/100",
            "state": {"life_cycle_state": "RUNNING", "state_message": "Launching"},
            "tasks": [
                {
                    "task_key": "training",
                    "run_id": 200,
                    "state": {"life_cycle_state": "RUNNING"},
                }
            ],
        }
        succeeded = {
            **running,
            "state": {
                "life_cycle_state": "TERMINATED",
                "result_state": "SUCCESS",
                "state_message": "",
            },
            "tasks": [
                {
                    "task_key": "training",
                    "run_id": 200,
                    "state": {
                        "life_cycle_state": "TERMINATED",
                        "result_state": "SUCCESS",
                        "state_message": "",
                    },
                    "ai_runtime_task": {"mlflow_experiment_id": "123"},
                }
            ],
        }
        client = FakeWorkspaceClient(
            [
                {"run_id": 100},
                running,
                succeeded,
                {"logs": "VERDICT: ACCEPTED", "mlflow_run_id": "abc"},
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = submitter.submit_and_wait(client, payload, poll_seconds=0)

        self.assertEqual(exit_code, 0)
        self.assertIn("parent run id: 100", output.getvalue())
        self.assertIn("task run id: 200", output.getvalue())
        self.assertIn("MLflow experiment id: 123", output.getvalue())
        self.assertIn("MLflow run id: abc", output.getvalue())
        self.assertIn("VERDICT: ACCEPTED", output.getvalue())
        self.assertEqual(
            [(call["method"], call["path"]) for call in client.api_client.calls],
            [
                ("POST", "/api/2.2/jobs/runs/submit"),
                ("GET", "/api/2.2/jobs/runs/get"),
                ("GET", "/api/2.2/jobs/runs/get"),
                ("GET", "/api/2.1/jobs/runs/get-output"),
            ],
        )

    def test_submit_and_wait_keeps_polling_through_blocked(self):
        payload = self._payload()
        blocked = {
            "run_id": 100,
            "state": {"life_cycle_state": "BLOCKED", "state_message": "Waiting"},
            "tasks": [
                {
                    "task_key": "training",
                    "run_id": 200,
                    "state": {"life_cycle_state": "BLOCKED"},
                }
            ],
        }
        succeeded = {
            **blocked,
            "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
            "tasks": [
                {
                    "task_key": "training",
                    "run_id": 200,
                    "state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"},
                }
            ],
        }
        client = FakeWorkspaceClient([{"run_id": 100}, blocked, succeeded, {}])

        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = submitter.submit_and_wait(client, payload, poll_seconds=0)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call["path"] for call in client.api_client.calls].count("/api/2.2/jobs/runs/get"), 2
        )

    def test_submit_and_wait_returns_nonzero_and_prints_prelaunch_failure(self):
        payload = self._payload()
        failed = {
            "run_id": 100,
            "run_page_url": "https://workspace/#job/100",
            "state": {
                "life_cycle_state": "INTERNAL_ERROR",
                "result_state": "FAILED",
                "state_message": "Task failed before user code started",
            },
            "tasks": [
                {
                    "task_key": "training",
                    "run_id": 200,
                    "state": {
                        "life_cycle_state": "INTERNAL_ERROR",
                        "result_state": "FAILED",
                        "state_message": "Unsupported base environment",
                    },
                }
            ],
        }
        client = FakeWorkspaceClient([{"run_id": 100}, failed, {}])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = submitter.submit_and_wait(client, payload, poll_seconds=0)

        self.assertEqual(exit_code, 1)
        self.assertIn("Unsupported base environment", output.getvalue())
        self.assertIn("task output: <none>", output.getvalue())

    def test_get_output_error_does_not_hide_terminal_state(self):
        payload = self._payload()
        failed = {
            "run_id": 100,
            "state": {
                "life_cycle_state": "INTERNAL_ERROR",
                "result_state": "FAILED",
                "state_message": "Environment installation failed",
            },
            "tasks": [
                {
                    "task_key": "training",
                    "run_id": 200,
                    "state": {
                        "life_cycle_state": "INTERNAL_ERROR",
                        "result_state": "FAILED",
                        "state_message": "Dependency resolution failed",
                    },
                }
            ],
        }
        client = FakeWorkspaceClient([{"run_id": 100}, failed, RuntimeError("no output")])
        original_do = client.api_client.do

        def raising_do(**kwargs):
            response = original_do(**kwargs)
            if isinstance(response, Exception):
                raise response
            return response

        client.api_client.do = raising_do

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = submitter.submit_and_wait(client, payload, poll_seconds=0)

        self.assertEqual(exit_code, 1)
        self.assertIn("Dependency resolution failed", output.getvalue())
        self.assertIn("Jobs get-output was unavailable: RuntimeError: no output", output.getvalue())


    def _staging_client(self, responses, command=b"#!/bin/bash\necho hi\n"):
        client = FakeWorkspaceClient(responses)
        client.workspace = FakeWorkspaceDownload(io.BytesIO(command))
        return client

    def test_workload_parameter_names_are_parsed_and_validated(self):
        self.assertEqual(
            submitter.parse_workload_parameter_names(" epochs, lr ,,data.split "),
            ["epochs", "lr", "data.split"],
        )
        self.assertEqual(submitter.parse_workload_parameter_names(""), [])
        with self.assertRaisesRegex(ValueError, "reserved for the launcher"):
            submitter.parse_workload_parameter_names("epochs,mlflow_run")
        with self.assertRaisesRegex(ValueError, "listed twice"):
            submitter.parse_workload_parameter_names("epochs,epochs")
        with self.assertRaisesRegex(ValueError, "must start with a letter"):
            submitter.parse_workload_parameter_names("learning rate")

    def test_workload_parameters_merge_json_with_named_strings(self):
        parameters = submitter.parse_workload_parameters(
            '{"epochs": 3, "optimizer": {"name": "adamw"}}', {"lr": "0.001"}
        )

        self.assertEqual(
            parameters, {"epochs": 3, "optimizer": {"name": "adamw"}, "lr": "0.001"}
        )
        self.assertEqual(submitter.parse_workload_parameters("  ", {}), {})

    def test_workload_parameters_reject_non_objects_and_conflicts(self):
        with self.assertRaisesRegex(ValueError, "JSON object, not list"):
            submitter.parse_workload_parameters("[1, 2]")
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            submitter.parse_workload_parameters("{epochs: 3}")
        with self.assertRaisesRegex(ValueError, "set both"):
            submitter.parse_workload_parameters('{"epochs": 3}', {"epochs": "4"})

    def test_param_arguments_split_on_first_equals(self):
        self.assertEqual(
            submitter.parse_param_arguments(["filter=a=b", "empty="]),
            {"filter": "a=b", "empty": ""},
        )
        with self.assertRaisesRegex(ValueError, "NAME=VALUE"):
            submitter.parse_param_arguments(["epochs"])
        with self.assertRaisesRegex(ValueError, "set twice"):
            submitter.parse_param_arguments(["epochs=1", "epochs=2"])

    def test_hyperparameters_yaml_is_json_and_rejects_nan(self):
        rendered = submitter.render_hyperparameters_yaml({"lr": "0.001", "epochs": 3})

        self.assertEqual(json.loads(rendered), {"lr": "0.001", "epochs": 3})
        with self.assertRaisesRegex(ValueError, "finite JSON values"):
            submitter.render_hyperparameters_yaml({"lr": float("nan")})

    def test_stage_launch_directory_copies_command_beside_hyperparameters(self):
        client = self._staging_client([{}, {}, {}])

        command_path, hyperparameters_path = submitter.stage_launch_directory(
            client,
            command_path="/Workspace/Shared/probe/run_probe.sh",
            parameters={"epochs": 3},
            experiment="wheelhouse-probe",
            mlflow_run="lr sweep/1",
            launch_root="/Workspace/Shared/launch/",
        )

        directory = command_path.rsplit("/", 1)[0]
        self.assertRegex(
            directory, r"^/Workspace/Shared/launch/wheelhouse-probe/lr-sweep-1_[0-9a-f]{16}$"
        )
        self.assertEqual(command_path, f"{directory}/run_probe.sh")
        self.assertEqual(hyperparameters_path, f"{directory}/hyperparameters.yaml")
        self.assertEqual(client.workspace.paths, ["/Shared/probe/run_probe.sh"])
        mkdirs, command_import, hyperparameters_import = client.api_client.calls
        self.assertEqual(mkdirs["path"], "/api/2.0/workspace/mkdirs")
        self.assertEqual(mkdirs["body"], {"path": directory[len("/Workspace") :]})
        self.assertEqual(command_import["body"]["path"], command_path[len("/Workspace") :])
        self.assertEqual(
            base64.b64decode(command_import["body"]["content"]), b"#!/bin/bash\necho hi\n"
        )
        self.assertFalse(command_import["body"]["overwrite"])
        self.assertEqual(
            json.loads(base64.b64decode(hyperparameters_import["body"]["content"])),
            {"epochs": 3},
        )

    def test_stage_launch_directory_without_parameters_copies_only_command(self):
        client = self._staging_client([{}, {}])

        command_path, hyperparameters_path = submitter.stage_launch_directory(
            client,
            command_path="/Workspace/Shared/probe/run_probe.sh",
            experiment="wheelhouse-probe",
            mlflow_run="run",
            launch_root="/Workspace/Shared/launch",
        )

        self.assertIsNone(hyperparameters_path)
        directory = command_path.rsplit("/", 1)[0]
        self.assertEqual(command_path, f"{directory}/run_probe.sh")
        mkdirs, command_import = client.api_client.calls
        self.assertEqual(mkdirs["path"], "/api/2.0/workspace/mkdirs")
        self.assertEqual(command_import["body"]["path"], command_path[len("/Workspace") :])
        self.assertFalse(
            any("hyperparameters.yaml" in str(call) for call in client.api_client.calls)
        )

    def test_stage_launch_directory_defaults_to_caller_home(self):
        client = self._staging_client([{"userName": "someone@example.com"}, {}, {}, {}])

        command_path, _ = submitter.stage_launch_directory(
            client,
            command_path="/Workspace/Shared/run_probe.sh",
            parameters={"epochs": 3},
            experiment="wheelhouse-probe",
            mlflow_run="run",
        )

        self.assertTrue(
            command_path.startswith(
                "/Workspace/Users/someone@example.com/.air/jobs_launch/wheelhouse-probe/run_"
            )
        )
        self.assertEqual(client.api_client.calls[0]["path"], "/api/2.0/preview/scim/v2/Me")

    def test_stage_launch_directory_rejects_bad_paths_before_writing(self):
        for command_path, launch_root, message in (
            ("/Volumes/c/s/v/run.sh", "/Workspace/Shared/launch", "command_path must be"),
            ("/Workspace/Shared/run.sh", "/Workspace/Shared/../launch", "launch_root must"),
            ("/Workspace/Shared/run.sh", "/Volumes/c/s/v", "launch_root must"),
        ):
            client = self._staging_client([])
            with self.subTest(command_path=command_path, launch_root=launch_root):
                with self.assertRaisesRegex(ValueError, message):
                    submitter.stage_launch_directory(
                        client,
                        command_path=command_path,
                        parameters={"epochs": 3},
                        experiment="wheelhouse-probe",
                        mlflow_run="run",
                        launch_root=launch_root,
                    )
                self.assertEqual(client.api_client.calls, [])

    def test_stage_launch_directory_reports_unreadable_command(self):
        client = FakeWorkspaceClient([])
        client.workspace = FakeMissingDownload()

        with self.assertRaisesRegex(ValueError, "command script does not exist"):
            submitter.stage_launch_directory(
                client,
                command_path="/Workspace/Shared/run_probe.sh",
                parameters={"epochs": 3},
                experiment="wheelhouse-probe",
                mlflow_run="run",
                launch_root="/Workspace/Shared/launch",
            )
        self.assertEqual(client.api_client.calls, [])

    def test_with_command_path_replaces_only_the_deployment_command(self):
        payload = self._payload()

        updated = submitter.with_command_path(payload, "/Workspace/Shared/launch/run/run_probe.sh")

        deployment = updated["tasks"][0]["ai_runtime_task"]["deployments"][0]
        self.assertEqual(deployment["command_path"], "/Workspace/Shared/launch/run/run_probe.sh")
        self.assertEqual(
            payload["tasks"][0]["ai_runtime_task"]["deployments"][0]["command_path"],
            "/Workspace/Shared/run_probe.sh",
        )
        self.assertEqual(updated["idempotency_token"], payload["idempotency_token"])

    def test_launcher_defaults_match_notebook_widgets(self):
        notebook = (BUNDLE / "run_ai_runtime_job.py").read_text()
        widget_cell = notebook.split("# COMMAND ----------")[1]
        widgets = dict(
            re.findall(
                r'dbutils\.widgets\.(?:text|dropdown)\(\s*"([^"]+)",\s*"([^"]*)"', widget_cell
            )
        )

        self.assertEqual(widgets, submitter.LAUNCHER_PARAMETER_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
