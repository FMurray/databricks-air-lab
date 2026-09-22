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
import subprocess
import sys


EXPECTED_VERSIONS = {
    "mlflow": "3.16.0",
    "pydantic": "2.13.5",
    "scikit-learn": "1.9.1",
}
SENTINEL = "JOBS_API_WHEELHOUSE_ENV_OK"


def main() -> int:
    checks = []
    try:
        observed = {
            package: importlib.metadata.version(package)
            for package in EXPECTED_VERSIONS
        }
        versions_ok = observed == EXPECTED_VERSIONS
        checks.append(Check(
            name="Generated wheelhouse packages are active",
            status=PASS if versions_ok else FAIL,
            measured=", ".join(f"{name}={version}" for name, version in observed.items()),
            threshold=", ".join(f"{name}={version}" for name, version in EXPECTED_VERSIONS.items()),
            what_why="Imports three exact versions supplied by the generated offline wheelhouse. "
                     "A mismatch means the Jobs environment did not apply that wheelhouse.",
            sufficient="All three versions must match the build manifest exactly; any missing or "
                       "different version fails this check.",
            likely_means="The environment file was rejected, skipped, or installed into a different "
                         "interpreter than the command uses.",
        ))
    except Exception as exc:  # noqa: BLE001 — render the failure
        checks.append(_fail_from_exc(
            "Generated wheelhouse packages are active",
            "all three exact versions import",
            "Proves the Jobs environment installed the generated offline wheelhouse.",
            "The environment file did not finish installing before user code started.",
            exc,
        ))

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        gpu_ok = len(gpus) == 1 and "A10" in gpus[0]
        checks.append(Check(
            name="Requested A10 accelerator is visible",
            status=PASS if gpu_ok else FAIL,
            measured=f"gpu_count={len(gpus)}, names={gpus}",
            threshold="exactly one GPU whose name contains A10",
            what_why="Confirms this is the requested native AI Runtime task shape, not a CPU-only "
                     "environment setup test.",
            sufficient="One visible A10 exactly matches accelerator_count=1; zero, multiple, or a "
                       "different model fails the requested shape.",
            likely_means="The task ran on the wrong compute shape or GPU device scoping failed.",
        ))
    except Exception as exc:  # noqa: BLE001 — render the failure
        checks.append(_fail_from_exc(
            "Requested A10 accelerator is visible",
            "exactly one visible A10",
            "Confirms the Jobs API native AI Runtime task received its requested GPU.",
            "The requested accelerator was not attached or nvidia-smi failed.",
            exc,
        ))

    passed = all(check.status == PASS for check in checks)
    sentinels = SENTINEL if passed else "none"
    if passed:
        print(SENTINEL, flush=True)

    try:
        run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_RUN_NAME") or "local"
        return render_report(
            checks,
            run_id=run_id,
            profile="f-classic",
            shape="world=1, 1xGPU_1xA10",
            scope="acceptance",
            runtime=f"python={sys.version.split()[0]}",
            sentinels=sentinels,
        )
    except Exception:  # noqa: BLE001 — never lose the verdict
        _tb.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
