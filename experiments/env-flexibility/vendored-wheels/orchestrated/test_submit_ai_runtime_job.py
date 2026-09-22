import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "air_wheelhouse_job_submitter", HERE / "submit_ai_runtime_job.py"
)
submitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submitter)


class FakeApiClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def do(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeWorkspaceClient:
    def __init__(self, responses):
        self.api_client = FakeApiClient(responses)


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
            idempotency_token="stable-token",
        )

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


if __name__ == "__main__":
    unittest.main()
