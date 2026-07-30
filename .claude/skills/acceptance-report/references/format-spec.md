# Acceptance report — format spec

The plain-English *acceptance report* that every workload in this repo prints at the end of its
run. It exists so a customer confirmer — not necessarily a GPU or distributed-ML expert — can
**run the test → pull the log → read the verdict**, without decoding sentinels, NCCL spew, or
tracebacks. Every workload (FSDP, nccl-allreduce, correctness probe, gpu-burn, …) uses this same
format via the shared renderer (`renderer.py`), so all tests read alike.

This file is the **format spec** — what a report looks like. For the procedure to add a report to
a workload, see `../SKILL.md`; the report code is `renderer.py`. Read alongside
`docs/about-receipts.md` (the evidence markers) and the `experiment-verification` skill.

## Principles

1. **Always renders — including on failure.** Checks *record* an outcome; they do not
   `assert`-and-raise. The runner renders from the records, then derives the exit code last.
2. **Derived from what was measured** — never a parallel narrative that can drift from the result.
3. **stdout on rank 0, behind a banner.** It's the one sink `air logs`/`--watch` reliably returns
   (UC volumes are BR-2-blocked, `/tmp` doesn't survive resubmit). A file is optional, not primary.
4. **Honest about scope.** A check vacuous at the current scale says so; the report attests only to
   what rank 0 saw (if another node dies the job may hang and this report may never print).

## Layout

```
==================== <WORKLOAD> ACCEPTANCE REPORT ====================
Run <id>   Profile <profile>   Shape <e.g. world=N, M×TYPE> ( <smoke | acceptance> )
Runtime <versions the workload already prints>   When <UTC>

  Attests to what rank 0 observed. On multi-node the CLI streams node 0 only
  (`air logs <id> --node N`). If this report is absent, treat it as a failure.
----------------------------------------------------------------------
CHECK <n> — <plain-language name>
  Status ....... <PASS | FAIL | BLOCKED | SKIPPED | N/A-at-this-scale>
  Measured ..... <concrete datum>   Threshold: <the bar it had to clear>
  What & why ... <what it tested, in plain terms, and what breaks in a real job without it>
  Sufficient ... <why THIS value is enough — tie it to the threshold; name what a fail looks like>
----------------------------------------------------------------------
VERDICT: <ACCEPTED | ACCEPTED WITH CAVEATS | NOT ACCEPTED>
  <one sentence, scoped to the shape>   Sentinels: <…>   Exit: <0 | non-zero>
```

## The five per-check fields

| Field | Rule |
|---|---|
| **Name** | Plain language. No sentinel strings (those go on the verdict line). |
| **Status** | Exactly one enum value (below). |
| **Measured + Threshold** | A number/fact the confirmer can see, plus the bar. "Sufficient" is a quantity, not an adjective. |
| **What & why** | For someone who's never done this — say what breaks in a real job if the property fails. |
| **Sufficient** | Connect the value to the threshold and name what a *failing* value would look like. |

## Status enum (exactly these five)

- **PASS** — ran, cleared the threshold.
- **FAIL** — ran, did not clear it. Triggers the failure block.
- **BLOCKED** — an external precondition stopped it (name the blocker); not a failure of the thing under test.
- **SKIPPED** — deliberately not run this invocation (say why).
- **N/A-at-this-scale** — vacuous at the current scale (e.g. a distributed property at world=1). *Not* a PASS.

## Scope awareness

When a run can't exercise a check meaningfully (single-GPU smoke, a feature toggled off), that row
is `N/A-at-this-scale` or `SKIPPED` — never PASS — and the Shape line reads `( smoke )` with the
VERDICT capped at `ACCEPTED WITH CAVEATS`. The verdict text is generated from the run's scope, so a
run can't claim a proof it didn't perform.

## On FAIL

After the verdict, append — plain English first, then the trace:

```
WHAT THIS LIKELY MEANS
  CHECK <n> failed: <value> did not meet <threshold>. <1–3 sentences a non-expert can
  act on: the most common cause and the first thing to check / what to send support.>

FOR SUPPORT — raw traceback
  <full traceback, verbatim>
```

Each check carries a static "likely means" string authored when the check is written. The
traceback is retained, never swallowed, and fenced below the plain-English block.
