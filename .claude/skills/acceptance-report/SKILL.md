---
name: acceptance-report
description: >-
  Add (or change) the plain-English acceptance report a workload prints at the end of its run —
  the CHECK/VERDICT block a customer confirmer reads to get "run the test → pull the log → read
  the verdict" without decoding sentinels, NCCL spew, or tracebacks. Use whenever adding a report
  to a new workload (FSDP, nccl-allreduce, correctness probe, gpu-burn, …), editing the report
  format, or wiring a workload's checks/verdict/exit-code. Carries the canonical renderer and the
  format spec (layout, status enum, scope rules) as copyable references.
---

# Acceptance report

Every workload in this repo ends by printing one **acceptance report**: a plain-English
`CHECK … / VERDICT` block a customer confirmer — not a GPU or distributed-ML expert — reads to
**run the test → pull the log → read the verdict**. Every workload uses the *same* renderer so all
reports read alike.

- **Format spec:** `references/format-spec.md` — layout, the five per-check fields, the status
  enum, scope rules, the On-FAIL block. Read that to understand *what a report looks like*.
- **Canonical renderer (for the copy):** `references/renderer.py` — this skill's single source of
  truth for the report *code*.

Both are shareable with a customer confirmer as-is — they explain the report without naming
anything customer-specific.

This skill is the procedure for putting a report into a workload.

## Important: Copy the renderer, do NOT import

The renderer is **duplicated into every workload script, never imported**. Reason: each AIR YAML
snapshots **only its own experiment directory** into the container, so there is no shared module at
runtime — an `import` that works on your laptop vanishes on AIR. `train_fsdp.py` and
`distributed_correctness_probe.py` both carry byte-identical copies for exactly this reason.

Consequences you must respect:

- `references/renderer.py` is the source of truth. To change the format, **edit that file first**,
  then re-sync every workload that copied it. The copies must stay byte-identical (below `WORKLOAD`).
- Never "DRY it up" into a shared import — that reintroduces the runtime-missing-module failure.

## Adding a report to a new workload

1. **Copy the renderer** from `references/renderer.py` into the workload script: the imports,
   `WORKLOAD`, the five status constants, `Check`, `_fail_from_exc`, `_wrap`, `render_report`. Set
   `WORKLOAD` to the display name. Do not edit the rendering logic — only the per-check strings and
   the call site are workload-specific.
2. **Write one `Check` per proof.** Each check *records* an outcome — it does **not**
   `assert`-and-raise. Fill the five fields per the spec (`Name`, `Status`, `Measured`+`Threshold`,
   `What & why`, `Sufficient`) plus `likely_means` (a static string for the On-FAIL block). Wrap the
   body of anything that can throw and turn the exception into a FAIL record via `_fail_from_exc`,
   so the report still renders.
3. **Keep machine sentinels as-is.** The existing grep/receipt sentinels (`FSDP_*_OK`,
   `DISTRIBUTED_CORRECTNESS_OK`, …) stay exactly where they are — this report is a *plain-English
   layer on top of them*, not a replacement. Pass them to `render_report(..., sentinels=…)`.
4. **Render on rank 0 only, and guard it.** Only rank 0 calls `render_report`; wrap the call in
   `try/except` (see the call-site comment in the reference) so the renderer can never itself
   swallow the verdict.
5. **Derive the exit code LAST**, from the returned value — any `FAIL` ⇒ non-zero;
   `BLOCKED`/`SKIPPED`/`N/A` alone ⇒ 0. `sys.exit(exit_code)`. "Exited 0" is not evidence; the
   verdict is.
6. **Reference the skill by path** from a code comment (`# acceptance report — see the
   acceptance-report skill`) and from the workload YAML, so a reader can find the format.

## Guardrails baked into the renderer (don't undo them)

- **Always renders, including on failure** — checks record, they don't raise. Verdict is derived
  from the records, exit code derived last.
- **Verdict is generated from scope + statuses**, never a parallel narrative: `smoke` scope, or any
  `BLOCKED`/`SKIPPED`/`N/A`, caps the verdict at `ACCEPTED WITH CAVEATS`. A run can't claim a proof
  it didn't perform.
- **Traceback retained, never swallowed** — fenced under `FOR SUPPORT` on any FAIL.

## When done

If this is a new workload family adopting the format, note it in the workload's `NOTES.md`. If you
changed the format, update `references/format-spec.md`, edit `references/renderer.py` first, then
re-sync every workload copy.
