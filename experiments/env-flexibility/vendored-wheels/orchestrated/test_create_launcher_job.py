import contextlib
import importlib.util
import io
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "air_wheelhouse_launcher_job", HERE / "create_launcher_job.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

LAUNCHER_DEFAULTS = {
    "jobs_environment_file": "/Volumes/c/s/v/builds/abc/jobs-environment.json",
    "code_source_path": "/Workspace/Shared/probe.tar.gz",
    "command_path": "/Workspace/Shared/run_probe.sh",
    "experiment": "wheelhouse-probe",
    "mlflow_experiment_directory": "/Workspace/Shared",
    "mlflow_run": "nightly-{{job.trigger.time.iso_date}}",
    "usage_policy_name": "gpu-research",
    "workload_parameters": '{"optimizer": {"name": "adamw"}}',
}


class CreateLauncherJobTest(unittest.TestCase):
    def _settings(self, **overrides):
        arguments = dict(
            name="air-wheelhouse-launcher",
            notebook_path="/Workspace/Shared/orchestrated/run_ai_runtime_job",
            launcher_defaults=dict(LAUNCHER_DEFAULTS),
            named_parameters={"epochs": "3", "lr": "0.001"},
            existing_cluster_id="0101-abc",
        )
        arguments.update(overrides)
        return launcher.build_job_settings(**arguments)

    def test_every_launcher_widget_and_named_parameter_is_a_job_parameter(self):
        settings = self._settings()
        parameters = {item["name"]: item["default"] for item in settings["parameters"]}

        expected_launcher = set(launcher.submitter.LAUNCHER_PARAMETER_NAMES)
        self.assertEqual(set(parameters), expected_launcher | {"epochs", "lr"})
        # The submission's idempotency token is always auto-generated, never a job parameter.
        self.assertNotIn("idempotency_token", parameters)
        self.assertEqual(parameters["workload_parameter_names"], "epochs,lr")
        self.assertEqual(parameters["epochs"], "3")
        self.assertEqual(parameters["accelerator_type"], "GPU_1xA10")
        self.assertEqual(parameters["mlflow_run"], "nightly-{{job.trigger.time.iso_date}}")

    def test_launcher_task_is_a_single_non_retrying_notebook(self):
        settings = self._settings()

        (task,) = settings["tasks"]
        self.assertEqual(
            task["notebook_task"],
            {
                "notebook_path": "/Workspace/Shared/orchestrated/run_ai_runtime_job",
                "source": "WORKSPACE",
            },
        )
        self.assertEqual(task["max_retries"], 0)
        self.assertEqual(task["existing_cluster_id"], "0101-abc")
        self.assertNotIn("environments", settings)
        self.assertNotIn("usage_policy_id", settings)

    def test_serverless_launcher_has_no_cluster(self):
        settings = self._settings(existing_cluster_id="", serverless=True)

        self.assertNotIn("existing_cluster_id", settings["tasks"][0])

    def test_invalid_launcher_settings_fail_before_jobs(self):
        cases = (
            ({"existing_cluster_id": "", "serverless": False}, "exactly one of existing_cluster"),
            ({"named_parameters": {"wait": "false"}}, "reserved for the launcher"),
            (
                {"named_parameters": {"optimizer": "sgd"}},
                "set both in workload_parameters",
            ),
            (
                {"launcher_defaults": {**LAUNCHER_DEFAULTS, "idempotency_token": "fixed"}},
                "not launcher job parameters",
            ),
            (
                {"launcher_defaults": {**LAUNCHER_DEFAULTS, "usage_policy_id": "p-1"}},
                "exactly one of usage_policy_name",
            ),
            (
                {"launcher_defaults": {**LAUNCHER_DEFAULTS, "command_path": ""}},
                "required launcher defaults are blank: command_path",
            ),
            (
                {
                    "launcher_defaults": {
                        **LAUNCHER_DEFAULTS,
                        "experiment": "/Workspace/Shared/wheelhouse-probe",
                    }
                },
                "must be a leaf name",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._settings(**overrides)

    def test_dry_run_prints_settings_without_a_client(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = launcher.main(
                [
                    "--name",
                    "air-wheelhouse-launcher",
                    "--notebook-path",
                    "/Workspace/Shared/orchestrated/run_ai_runtime_job",
                    "--serverless",
                    "--dry-run",
                    "--jobs-environment-file",
                    LAUNCHER_DEFAULTS["jobs_environment_file"],
                    "--code-source-path",
                    LAUNCHER_DEFAULTS["code_source_path"],
                    "--command-path",
                    LAUNCHER_DEFAULTS["command_path"],
                    "--experiment",
                    "wheelhouse-probe",
                    "--mlflow-experiment-directory",
                    "/Workspace/Shared",
                    "--mlflow-run",
                    "nightly",
                    "--usage-policy-id",
                    "policy-123",
                    "--param",
                    "epochs=3",
                ]
            )

        self.assertEqual(exit_code, 0)
        settings = json.loads(output.getvalue())
        self.assertEqual(settings["usage_policy_id"], "policy-123")
        parameters = {item["name"]: item["default"] for item in settings["parameters"]}
        self.assertEqual(parameters["usage_policy_id"], "policy-123")
        self.assertEqual(parameters["workload_parameter_names"], "epochs")


if __name__ == "__main__":
    unittest.main()
