# UAT suite configuration — consumed by the DRIVER notebook (plain Python: the default
# serverless env has no yaml module and PyPI is blocked on this workspace, so no pip).
#
# Each check notebook is executed as a one-time serverless notebook job per shape, with the
# accelerator pinned explicitly via the Jobs API:
#   tasks[].compute.hardware_accelerator: GPU_1xA10 | GPU_1xH100 | GPU_8xH100
#   environments[].spec.environment_version: "5" — v4 job-plane egress is broken on this ws (docs/06)
# "CPU" means plain serverless (no compute block).
#
# NB (2026-07-24): keep dependencies [] until the PyPI-egress blocker is fixed — any pip
# dependency currently fails environment build on this workspace (docs/06-uat-suite.md).
#
# Do NOT rename this to environment.yml / put an environment.yml here: that filename is a
# live Databricks convention that binds notebook environments folder-wide (it broke the
# CPU driver, run 893396989001873).
#
# Multi-node shapes (e.g. 2 nodes x 8xH100) are deliberately ABSENT: distributed multi-node
# goes through the air CLI only (engagement rule — see RUNNING-UAT.md ground rules);
# use workloads/multinode-*.yaml and workloads/nccl-allreduce.example.yaml.

UAT_CONFIG = {
    "environment_version": "5",
    "dependencies": [],
    "checks": [
        {
            "path": "checks/network-blockers",
            # CPU baseline + the AIR-plane view (the two DIFFER here — docs/06 plane differential)
            "shapes": ["CPU", "GPU_1xA10"],
            "timeout_minutes": 15,
        },
        {
            "path": "checks/gpu-smoke",
            "shapes": ["GPU_1xA10", "GPU_1xH100", "GPU_8xH100"],
            "timeout_minutes": 25,
            "expect_gpus": {"GPU_1xA10": 1, "GPU_1xH100": 1, "GPU_8xH100": 8},
        },
    ],
}
