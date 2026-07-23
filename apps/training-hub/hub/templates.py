"""AIR workload YAML generation. Schema mirrored from workloads/*.example.yaml
(verified against air v0.1.0: env_variables is rejected — set env inline in command)."""

from __future__ import annotations

import shutil
import subprocess

import yaml

# TODO: verify the full accelerator catalog against the current CLI.
ACCELERATOR_TYPES = ["GPU_1xA10", "GPU_8xA10", "GPU_1xH100", "GPU_8xH100"]

TEMPLATES = {
    "single-node-finetune": {
        "description": "One node, snapshot code source — the default starting point.",
        "num_accelerators": 1,
        "accelerator_type": "GPU_8xH100",
        "command": "python $CODE_SOURCE_PATH/train.py",
    },
    "multi-node-distributed": {
        "description": "Multi-node via CLI (the only multi-node path). torchrun in command.",
        "num_accelerators": 16,
        "accelerator_type": "GPU_8xH100",
        "command": (
            "torchrun --nnodes=$NUM_NODES --nproc-per-node=$LOCAL_WORLD_SIZE "
            "--node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT "
            "$CODE_SOURCE_PATH/train.py"
        ),
    },
}


def build_workload(
    experiment_name: str,
    template: str,
    command: str,
    dependencies: list[str],
    num_accelerators: int,
    accelerator_type: str,
    timeout_minutes: int = 120,
    usage_policy_name: str | None = None,
    snapshot_root: str = ".",
) -> dict:
    workload = {
        "experiment_name": experiment_name,
        "environment": {"version": "4", "dependencies": dependencies},
        "compute": {
            "num_accelerators": num_accelerators,
            "accelerator_type": accelerator_type,
        },
        "code_source": {"type": "snapshot", "snapshot": {"root_path": snapshot_root}},
        "command": command,
        "max_retries": 0,
        "timeout_minutes": timeout_minutes,
    }
    if usage_policy_name:  # must exist in target workspace or validation fails
        workload["usage_policy_name"] = usage_policy_name
    return workload


def to_yaml(workload: dict) -> str:
    return yaml.safe_dump(workload, sort_keys=False)


def submit(yaml_path: str) -> tuple[bool, str]:
    """Submit via the air CLI when present (deployed app won't have it — see README)."""
    if not shutil.which("air"):
        return False, "air CLI not found on PATH — download the YAML and submit manually"
    proc = subprocess.run(
        ["air", "run", "-f", yaml_path],  # verified against air v0.1.0 (2026-07-22)
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr
