"""Verify a generated wheelhouse environment on a native Jobs ai_runtime_task."""

# acceptance report — see the acceptance-report skill
# ==========================================================================================
# CANONICAL acceptance-report renderer — copy this block verbatim into a workload.
#
# This is the single source of truth for the report code. When adding a report to a new
# workload, COPY these definitions into the workload's script (do NOT import them): each AIR
# YAML snapshots ONLY its own experiment directory, so there is no shared module at runtime.
# When the format changes, edit THIS file first, then re-sync every workload that copied it
# (train_fsdp.py, distributed_correctness_probe.py, …) so all reports stay identical.
#
# What you write per workload is the per-`Check` strings and the render_report() call site —
# NOT this rendering code. See the acceptance-report skill for the procedure and format spec.
# ==========================================================================================
from __future__ import annotations

import os
import signal
import textwrap
import traceback as _tb
from dataclasses import dataclass
from datetime import datetime, timezone

# Per-workload: set this to the workload's display name.
WORKLOAD = "JOBS API WHEELHOUSE ENVIRONMENT"

# Status enum — exactly these five (see format spec §"Status enum").
PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
NA = "N/A-at-this-scale"


@dataclass
class Check:
    """One acceptance check. `status` is one of the five enum values; `traceback` is retained
    (never swallowed) and fenced under the verdict when the run has any FAIL."""
    name: str
    status: str
    measured: str
    threshold: str
    what_why: str
    sufficient: str
    likely_means: str = ""
    traceback: str = ""


def _fail_from_exc(name, threshold, what_why, likely_means, exc) -> Check:
    """Turn an exception into a FAIL record (principle 1: record, don't re-raise) so the report
    still renders and the verdict/exit code can be derived from it. Trace is kept verbatim."""
    return Check(name=name, status=FAIL, measured=f"raised {type(exc).__name__}: {exc}",
                 threshold=threshold, what_why=what_why,
                 sufficient="A raised exception means the property could not be established.",
                 likely_means=likely_means, traceback="".join(_tb.format_exception(exc)))


def _wrap(text: str, indent: str = "               ") -> str:
    """Wrap a long field to ~92 cols, hanging-indented under its dotted label."""
    return textwrap.fill(text, width=96, initial_indent="", subsequent_indent=indent)


def _receipt(checks: "list[Check]", verdict: str, exit_code: int, test_id: str) -> None:
    """Dual-sink the verdict into MLflow params (the durable leg). stdout is the report's
    primary sink but it depends on the env's log delivery and expires with job-run retention
    (format spec §"Preconditions") — the receipt makes an absent stdout report disambiguable:
    receipt present = logs didn't ship; receipt absent = the run died before the verdict.
    Client API bound to MLFLOW_RUN_ID (never start_run — resuming the launcher-owned run
    fails silently on the job plane); alarm-guarded so a blocked tracking call can't hang
    the run; skips cleanly when MLFLOW_RUN_ID is unset (local)."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.log_param(run_id, "acceptance_verdict", verdict)
        client.log_param(run_id, "acceptance_exit", exit_code)
        if test_id:
            client.log_param(run_id, "acceptance_test_id", test_id)
        for i, c in enumerate(checks, 1):
            client.log_param(run_id, f"acceptance_check_{i}", f"{c.status} — {c.name}"[:490])
    except Exception as e:                                 # noqa: BLE001 — receipt is best-effort
        print(f"acceptance receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def render_report(checks: "list[Check]", run_id: str, profile: str, shape: str,
                  scope: str, runtime: str, sentinels: str, test_id: str = "") -> int:
    """Render every check identically and DERIVE the exit code last. Returns the exit code:
    any FAIL ⇒ 1; BLOCKED / SKIPPED / N/A alone ⇒ 0. Verdict is generated from scope + statuses
    so a run cannot claim a proof it did not perform (smoke ⇒ capped at ACCEPTED WITH CAVEATS).
    `test_id` is the UAT results-registry id (utils/verification/results/registry.py) — the
    join key shared by the registry row, the sheet row, and the MLflow receipt."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    W = 70
    out = []
    out.append("=" * 20 + f" {WORKLOAD} ACCEPTANCE REPORT " + "=" * 20)
    out.append(f"Run {run_id}   Profile {profile}   Shape {shape} ( {scope} )")
    out.append(f"Runtime {runtime}   When {when}")
    out.append("")
    out.append(_wrap("Attests to what rank 0 observed. On multi-node the CLI streams node 0 "
                     "only (`air logs <id> --node N`). If this report is absent, treat it as a "
                     "failure.", indent="  "))
    out.append("-" * W)

    has_fail = False
    for i, c in enumerate(checks, 1):
        if c.status == FAIL:
            has_fail = True
        out.append(f"CHECK {i} — {c.name}")
        out.append(f"  Status ....... {c.status}")
        out.append(f"  Measured ..... {c.measured}   Threshold: {c.threshold}")
        out.append(f"  What & why ... {_wrap(c.what_why)}")
        out.append(f"  Sufficient ... {_wrap(c.sufficient)}")
        out.append("-" * W)

    # Verdict — derived from scope + statuses (never a parallel narrative).
    softs = [c for c in checks if c.status in (BLOCKED, SKIPPED, NA)]
    if has_fail:
        verdict, exit_code = "NOT ACCEPTED", 1
        vline = "One or more checks did not clear their threshold at this shape."
    elif scope == "smoke" or softs:
        verdict, exit_code = "ACCEPTED WITH CAVEATS", 0
        capped = "smoke scope (single-process): distributed properties are vacuous here" \
            if scope == "smoke" else \
            "some checks were blocked / skipped / not applicable at this scale"
        vline = f"Every check that ran passed, but {capped} — see the rows above."
    else:
        verdict, exit_code = "ACCEPTED", 0
        vline = f"All checks passed at {shape}."
    out.append(f"VERDICT: {verdict}")
    out.append(f"  {vline}   Sentinels: {sentinels}   Test-id: {test_id or '-'}   "
               f"Exit: {exit_code}")

    # On FAIL — plain English first, then the raw trace (format spec §"On FAIL"). Never swallowed.
    if has_fail:
        out.append("")
        out.append("WHAT THIS LIKELY MEANS")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL:
                out.append(_wrap(f"CHECK {i} failed: {c.measured} did not meet "
                                 f"{c.threshold}. {c.likely_means}", indent="  "))
        out.append("")
        out.append("FOR SUPPORT — raw traceback")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL and c.traceback:
                out.append(f"  [CHECK {i} — {c.name}]")
                out.append(c.traceback.rstrip())

    print("\n" + "\n".join(out), flush=True)
    # Receipt AFTER the print: report delivery is priority one; the receipt is the durable leg.
    _receipt(checks, verdict, exit_code, test_id)
    return exit_code


import importlib.metadata
import re
import shlex
import subprocess
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version


SENTINEL = "JOBS_API_WHEELHOUSE_ENV_OK"
REQUEST_LOOKUP_TIMEOUT_SECONDS = 120
# `GPU_8xH100` = 8 H100s per node; accelerator_count is the total across nodes.
ACCELERATOR_TYPE_PATTERN = re.compile(r"GPU_(\d+)x(\w+)")
PACKAGES_CHECK = "Requested environment pins are the installed versions"
GPU_CHECK = "Requested accelerator shape is visible"


def _raise_timeout(_signum, _frame):
    raise TimeoutError(f"lookup exceeded {REQUEST_LOOKUP_TIMEOUT_SECONDS}s")


def load_requested_task():
    """Return ``(task, environment_spec, host)`` as Jobs recorded them for this task run.

    AI Training tags the launcher-owned MLflow run with its Jobs run IDs, so the run's own Jobs
    record states the compute and environment this task actually requested. Alarm-guarded so a
    hung tracking or Jobs call becomes a BLOCKED check instead of a hung task.
    """
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID is unset; not running as an AI Runtime task")
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(REQUEST_LOOKUP_TIMEOUT_SECONDS)
    try:
        from databricks.sdk import WorkspaceClient
        from mlflow.tracking import MlflowClient

        tags = MlflowClient().get_run(run_id).data.tags
        job_run_id = tags["mlflow.databricks.jobRunID"]
        task_run_id = tags["mlflow.databricks.taskRunID"]
        client = WorkspaceClient()
        run = client.api_client.do(
            method="GET", path="/api/2.2/jobs/runs/get", query={"run_id": int(job_run_id)}
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    task = next(
        (task for task in run.get("tasks") or [] if str(task.get("run_id")) == task_run_id), None
    )
    if task is None:
        raise RuntimeError(f"Jobs run {job_run_id} has no task run {task_run_id}")
    environment_key = task.get("environment_key")
    spec = next(
        (
            environment.get("spec") or {}
            for environment in run.get("environments") or []
            if environment.get("environment_key") == environment_key
        ),
        {},
    )
    return task, spec, client.config.host


def requested_pins(dependencies):
    """Return ``({name: version}, sources)`` for every exact pin the environment installs.

    Mirrors how the launcher reads ``spec.dependencies``: one requirements-file line per entry,
    with environment variables expanded. ``-r <file>`` entries are followed so the wheelhouse
    lock's pins are checked, not just the inline lines.
    """
    pins = {}
    sources = []
    for entry in dependencies:
        tokens = shlex.split(os.path.expandvars(entry))
        if not tokens:
            continue
        if tokens[0] in ("-r", "--requirement") and len(tokens) == 2:
            sources.append(tokens[1])
            lines = Path(tokens[1]).read_text().splitlines()
        elif tokens[0].startswith("-"):
            continue  # index, find-links, and hash-mode flags carry no package version
        else:
            sources.append("inline")
            lines = [entry]
        for line in lines:
            text = line.split("#", 1)[0].split(" --", 1)[0].strip()
            if not text or text.startswith("-"):
                continue
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                continue
            exact = [spec.version for spec in requirement.specifier if spec.operator == "=="]
            if len(exact) == 1:
                pins[requirement.name] = exact[0]
    return pins, list(dict.fromkeys(sources))


def check_packages(spec):
    dependencies = spec.get("dependencies") or []
    pins, sources = requested_pins(dependencies)
    if not pins:
        return Check(
            name=PACKAGES_CHECK, status=SKIPPED,
            measured=f"{len(dependencies)} dependency entries, no exact pins",
            threshold="every exact pin in the requested environment is installed",
            what_why="Compares the installed distributions with the versions the run's own "
                     "environment spec asked for.",
            sufficient="Nothing to compare: the requested environment pins no package versions.",
        )
    mismatches = []
    for name, version in sorted(pins.items()):
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        if installed == "missing" or Version(installed) != Version(version):
            mismatches.append(f"{name}: want {version}, have {installed}")
    measured = f"{len(pins) - len(mismatches)}/{len(pins)} pins match"
    if mismatches:
        shown = "; ".join(mismatches[:10])
        extra = f" (+{len(mismatches) - 10} more)" if len(mismatches) > 10 else ""
        measured += f"; {shown}{extra}"
    return Check(
        name=PACKAGES_CHECK, status=FAIL if mismatches else PASS,
        measured=measured,
        threshold=f"all {len(pins)} pins from {', '.join(sources)} installed in {sys.executable}",
        what_why="Reads the versions from the environment spec Jobs recorded for this run (and "
                 "the lock it references) and compares each with what this interpreter imports.",
        sufficient="Every pinned package must be present at exactly its pinned version; one "
                   "missing or different version fails the check.",
        likely_means="The environment install was skipped or failed, or the command runs a "
                     "different interpreter than the one the launcher installed into.",
    )


def _visible_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_accelerators(task):
    compute = task["ai_runtime_task"]["deployments"][0]["compute"]
    accelerator_type = compute["accelerator_type"]
    total = int(compute.get("accelerator_count") or 0)
    match = ACCELERATOR_TYPE_PATTERN.fullmatch(accelerator_type)
    if not match:
        raise ValueError(f"cannot parse accelerator_type {accelerator_type!r}")
    per_node, model = int(match[1]), match[2]
    expected_nodes = total // per_node
    gpus = _visible_gpus()
    observed_nodes = os.environ.get("NUM_NODES")
    shape_ok = len(gpus) == per_node and all(model in name for name in gpus)
    nodes_ok = observed_nodes is None or int(observed_nodes) == expected_nodes
    return Check(
        name=GPU_CHECK, status=PASS if shape_ok and nodes_ok else FAIL,
        measured=f"gpus_on_node={len(gpus)}, names={gpus}, NUM_NODES={observed_nodes or 'unset'}",
        threshold=f"{per_node} x {model} per node, {expected_nodes} node(s) "
                  f"({accelerator_type}, accelerator_count={total})",
        what_why="Compares the GPUs this node sees with the compute the run's Jobs record "
                 "requested, so a CPU-only or mis-scheduled pod cannot pass.",
        sufficient="This node must see exactly the per-node count of the requested model, and the "
                   "launched node count must match accelerator_count / per-node count.",
        likely_means="The task ran on the wrong compute shape or GPU device scoping failed.",
    )


def _blocked(name, threshold, reason, measured="not measured"):
    return Check(
        name=name, status=BLOCKED, measured=measured, threshold=threshold,
        what_why="The expected values come from this run's Jobs record, which could not be read.",
        sufficient=f"No expectation to compare against: {reason}",
    )


def main() -> int:
    checks = []
    task = spec = None
    host = os.environ.get("DATABRICKS_HOST") or "unknown"
    # Running as an ai_runtime_task means Jobs keys task success off this script's exit code, and
    # the probe is expected to verify a real run. If it then cannot read its own Jobs record, it
    # has failed to establish what it exists to prove, so that is a FAIL (exit non-zero) — like any
    # other check whose body raises — not a soft BLOCKED that would pass the Jobs task green while
    # verifying nothing. Only a local run (no MLFLOW_RUN_ID) is a genuine BLOCKED: no task to read.
    running_as_task = bool(os.environ.get("MLFLOW_RUN_ID"))
    try:
        task, spec, host = load_requested_task()
    except Exception as exc:  # noqa: BLE001 — recorded (FAIL or BLOCKED), still rendered
        if running_as_task:
            what_why = ("The expected values come from this run's Jobs record, which could not be "
                        "read, so neither property could be established.")
            likely_means = ("The run's Jobs record was unreadable — the environment may be broken "
                            "enough that the SDK/MLflow import failed, or the Jobs lookup errored.")
            checks.append(_fail_from_exc(PACKAGES_CHECK, "requested environment pins",
                                         what_why, likely_means, exc))
            checks.append(_fail_from_exc(GPU_CHECK, "requested accelerator shape",
                                         what_why, likely_means, exc))
        else:
            reason = f"{type(exc).__name__}: {exc}"
            try:
                observed = f"names={_visible_gpus()}"
            except Exception as gpu_exc:  # noqa: BLE001 — observation only
                observed = f"nvidia-smi raised {type(gpu_exc).__name__}"
            checks.append(_blocked(PACKAGES_CHECK, "requested environment pins", reason))
            checks.append(_blocked(GPU_CHECK, "requested accelerator shape", reason, observed))

    if task is not None:
        for name, threshold, check in (
            (PACKAGES_CHECK, "requested environment pins", lambda: check_packages(spec)),
            (GPU_CHECK, "requested accelerator shape", lambda: check_accelerators(task)),
        ):
            try:
                checks.append(check())
            except Exception as exc:  # noqa: BLE001 — render the failure
                checks.append(_fail_from_exc(
                    name, threshold,
                    "Compares this node with the run's own Jobs record.",
                    "The requested spec or the node's packages/GPUs could not be inspected.",
                    exc,
                ))

    passed = all(check.status == PASS for check in checks)
    sentinels = SENTINEL if passed else "none"
    if passed:
        print(SENTINEL, flush=True)

    try:
        if task is not None:
            compute = task["ai_runtime_task"]["deployments"][0]["compute"]
            shape = (f"{compute.get('accelerator_type')} x {compute.get('accelerator_count')}, "
                     f"nodes={os.environ.get('NUM_NODES', 'unset')}")
        else:
            shape = f"requested shape unknown, nodes={os.environ.get('NUM_NODES', 'unset')}"
        run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_RUN_NAME") or "local"
        return render_report(
            checks,
            run_id=run_id,
            profile=host,
            shape=shape,
            scope="acceptance",
            runtime=f"python={sys.version.split()[0]} ({sys.executable})",
            sentinels=sentinels,
        )
    except Exception:  # noqa: BLE001 — never lose the verdict
        _tb.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
