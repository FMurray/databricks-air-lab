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
# NOT this rendering code. See ../SKILL.md for the procedure and format-spec.md for what a
# rendered report looks like.
# ==========================================================================================
from __future__ import annotations

import textwrap
import traceback as _tb
from dataclasses import dataclass
from datetime import datetime, timezone

# Per-workload: set this to the workload's display name.
WORKLOAD = "<WORKLOAD NAME>"

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


def render_report(checks: "list[Check]", run_id: str, profile: str, shape: str,
                  scope: str, runtime: str, sentinels: str) -> int:
    """Render every check identically and DERIVE the exit code last. Returns the exit code:
    any FAIL ⇒ 1; BLOCKED / SKIPPED / N/A alone ⇒ 0. Verdict is generated from scope + statuses
    so a run cannot claim a proof it did not perform (smoke ⇒ capped at ACCEPTED WITH CAVEATS)."""
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
    out.append(f"  {vline}   Sentinels: {sentinels}   Exit: {exit_code}")

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
    return exit_code


# ------------------------------------------------------------------------------------------
# Call site (copy into the workload; adapt names). Guard so the renderer can never swallow the
# report, and derive the exit code from the rendered verdict — NOT from "the script exited 0".
# ------------------------------------------------------------------------------------------
#     exit_code = 0
#     if rank == 0:
#         scope = "smoke" if world == 1 else "acceptance"
#         shape = f"world={world}, {mesh_dev}"
#         try:
#             # run_id: prefer MLFLOW_RUN_ID (AIR-injected — the join key a confirmer uses to
#             # find this run), fall back to the display name, then "local".
#             run_id = (os.environ.get("MLFLOW_RUN_ID")
#                       or os.environ.get("MLFLOW_RUN_NAME") or "local")
#             exit_code = render_report(
#                 checks, run_id=run_id,
#                 profile=("local-cpu" if args.local else "air"),
#                 shape=shape, scope=scope, runtime=runtime_str,
#                 sentinels=" ".join(sentinels))
#         except Exception:                              # noqa: BLE001 — never lose the verdict
#             _tb.print_exc()
#             exit_code = 1
#     sys.exit(exit_code)
