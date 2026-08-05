"""Receipt-backed environment/dependency recipes injected into every brokered submission.

Each recipe encodes a sharp edge the UAT already paid for (receipts in docs/06-uat-suite.md
and docs/cookbook/) so users never re-hit it:
  - env v4 job-plane egress is broken on hardened workspaces  -> pin v5
  - managed 'databricks_ai_v5' is unexpressable for Jobs      -> custom base-env file
  - v5 default interpreter has no torch                        -> AI-env prelude
  - PyPI unreachable by design                                 -> vendored-wheels PYTHONPATH
"""

from __future__ import annotations

AI_ENV_ROOT = "/opt/databricks-environments/databricks-ai"

# jobs/runs/submit `environments` entry for notebook tasks (proven: run 118585768584418)
def job_environment(base_env_workspace_path: str | None = None) -> dict:
    spec = ({"base_environment": base_env_workspace_path} if base_env_workspace_path
            else {"environment_version": "5"})
    return {"environment_key": "hub", "spec": spec}


def command_prelude(needs_torch: bool = False, vendor_path: str | None = None) -> str:
    """Lines prepended to an AIR CLI workload command."""
    lines = []
    if needs_torch:
        # torch 2.9.0+cu129 lives here on env v5 (torch-v5-probe receipt)
        lines.append(f'export PATH="{AI_ENV_ROOT}/bin:$PATH"')
        lines.append(f'export PYTHONPATH="{AI_ENV_ROOT}/lib/python3.12/site-packages:$PYTHONPATH"')
    if vendor_path:
        lines.append(f'export PYTHONPATH="{vendor_path}:$PYTHONPATH"')
    return "\n".join(lines)


def notebook_first_cell_hint(needs_torch: bool = False) -> str:
    """Shown in the UI next to notebook submissions."""
    if not needs_torch:
        return ""
    return (f'import sys; sys.path.insert(0, "{AI_ENV_ROOT}/lib/python3.12/site-packages")'
            "  # env v5 default interpreter has no torch; the AI env is baked in at this path")


AIR_YAML_RULES = [
    'environment.version must be "5" (v4 job-plane egress is broken on hardened workspaces)',
    "dependencies must be [] on no-PyPI workspaces — vendor wheels (docs/cookbook/install-packages-from-a-volume.md)",
    "multinode goes through the air CLI only (never @distributed)",
    "always mlflow.end_run() in workload scripts (hung metrics-monitor thread mimics platform hangs)",
]
