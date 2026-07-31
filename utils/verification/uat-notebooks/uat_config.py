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

# ── Per-target values — REQUIRED, committed blank on purpose ──────────────────────────
# No workspace identifiers live in git: fill these in on the DEPLOYED copy of this file
# (uat/uat_config.py in the target workspace) before running the DRIVER — it fails fast,
# before submitting anything, listing whatever is still blank.
TARGET = {
    # catalog (or catalog.schema.table) whose CLOUD BUCKET the network check must read —
    # use the catalog the customer's workloads will actually write to
    "uc_catalog": "",
    # workspace root-storage bucket host (parse it from an MLflow artifact-upload error,
    # or ask the workspace admin) — the log/artifact delivery path
    "root_storage_host": "",
}

UAT_CONFIG = {
    "environment_version": "5",
    "dependencies": [],
    "checks": [
        {
            "path": "checks/network-blockers",
            # CPU baseline + the AIR-plane view (the two DIFFER here — docs/06 plane differential)
            "shapes": ["CPU", "GPU_1xA10"],
            "timeout_minutes": 15,
            # Per-target values come from TARGET above (the check itself is generic; run
            # standalone it can also auto-discover, but the DRIVER requires explicit values).
            "params": dict(TARGET),
        },
        {
            "path": "checks/air-cli-from-notebook",
            # CPU: the CLI needs no GPU; submit mode makes the check itself submit one
            # 1xA10 envvar-probe (12-min workload timeout) and poll it to terminal.
            "shapes": ["CPU"],
            "timeout_minutes": 30,
            "params": {"air_mode": "submit"},
        },
        {
            "path": "checks/gpu-smoke",
            "shapes": ["GPU_1xA10", "GPU_1xH100", "GPU_8xH100"],
            "timeout_minutes": 25,
            "expect_gpus": {"GPU_1xA10": 1, "GPU_1xH100": 1, "GPU_8xH100": 8},
        },
        {
            "path": "checks/pool-readiness",
            # One-notebook 20-node readiness: burn sweep (receipts + UUID distinctness)
            # + NCCL fabric probe, all submitted via the vendored air CLI (UAT #17 path).
            # DEFAULT IS A NO-COST SKIP: flip confirm_pool to "yes" (and coordinate in the
            # team channel) for the real sweep. Distributed still goes through the CLI —
            # this check merely fronts it from a CPU notebook.
            "shapes": ["CPU"],
            "timeout_minutes": 90,
            "params": {"confirm_pool": "no", "pool_nodes": "20",
                       "burn_seconds": "300", "fabric_nodes": "2"},
        },
    ],
}
