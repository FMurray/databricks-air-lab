# How to read the receipts

This cookbook feeds a live customer engagement. A claim that can't be traced to a run is a
liability, so every page follows the same evidence standard. Here's how to read it — and what to
do before adding your own.

## The markers

- **✅ VERIFIED, date, run id, workspace** — reproduced on a real workspace; the raw log is
  archived in the repo next to the finding (`experiments/<family>/run-<id>.log`).
- **"Reported"** — from a doc, thread, or field guide; a hypothesis, not evidence. The product
  docs lag reality in both directions, so a docs statement never upgrades itself to fact.
- **"Unverified" / open-q #N** — known unknown, numbered in
  [Open questions](04-open-questions.md) so gaps stay traceable.

## The number labels

- **Measured** — read directly from output. Quote freely, with the receipt.
- **Derived** — computed from measurements via a stated formula (e.g. busbw from algbw).
- **Inferred** — consistent with observations but not directly shown; alternatives named.
- **Smoke-grade** — a health check (one message size, 10 iterations), not a benchmark. A
  smoke number never travels into a customer deck without this caveat attached.

## Adding to the cookbook

New facts go through the pipeline: raw log archived → finding + claim→evidence mapping in the
experiment's `NOTES.md` → promoted here with the ✅ annotation. Design runs to produce evidence,
not vibes: a probe should assert its claim and print a sentinel that is unreachable unless the
assertions passed — "it exited 0" proves nothing.

The full standard (pre-submit and post-run checklists included) is the `experiment-verification`
skill: `.claude/skills/experiment-verification/SKILL.md`.
