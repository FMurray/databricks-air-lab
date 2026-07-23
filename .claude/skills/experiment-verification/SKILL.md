---
name: experiment-verification
description: Evidence standards for every experiment run in this repo — what counts as verified, how to archive receipts, and how findings get promoted to docs/. Load BEFORE running any AIR workload or writing results into NOTES.md/docs/, and when asked "where's the evidence" about a prior claim.
---

# Experiment verification standards

This repo's findings feed a live customer engagement. A claim that can't be traced to a run is a
liability: it will be repeated to the customer, contradicted later, and burn trust. The test for
every result you write down: **could someone who distrusts you reconstruct it from what you left
behind?**

## The standard: every claim gets a receipt

A finding is **verified** only when it has all of:

1. **Run identity** — Job Run ID, workspace, date. Write it inline where the claim lives:
   `✅ VERIFIED 2026-07-22, run 505819227973807, e2-demo-field-eng`.
2. **Primary evidence** — the exact log line(s), metric, or query output that supports it, quoted
   or line-referenced. Not a paraphrase of what you remember the log saying.
3. **Archived raw output** — copy the full run log into the experiment dir as
   `run-<job_run_id>.log` before the session's temp files vanish. The workspace retains logs too,
   but the repo copy is what makes `git log` a lab notebook.
4. **A claim→evidence mapping for headline results** — in NOTES.md, a short table or list pairing
   each claim with its log line/artifact. If you can't fill the evidence cell, the claim doesn't
   ship.

## Design runs to produce evidence, not vibes

- **Assert, don't observe.** A probe must verify its own success and print a sentinel that is
  unreachable unless the assertions passed (pattern: `MULTINODE_PROBE_OK` printed only after an
  all-reduce correctness assert). "It exited 0" is not evidence the thing you care about happened.
- **Define success criteria before submitting.** Write in NOTES.md what the run is supposed to
  demonstrate and what output will demonstrate it. If the run can't produce that output, redesign
  it before spending GPU time.
- **Smallest shape that yields the evidence.** A10 before H100, 1 node before N, 100 steps before
  500K. Escalate shape only when the question requires it (a multi-node question requires
  multiple nodes; a "does it OOM at 80GB" question requires the 80GB card).
- **Failures are findings.** Record exit codes and the *mechanism* (exit 137 host-OOM-kill ≠ CUDA
  OOM; grpc-status 0 silent drop ≠ delivery). A run that died informatively is a successful probe
  of a failure mode — write it up with the same rigor.

## Label the epistemic status of every number

- **Measured** — read directly from output (`1.4 ms/iter`). Safe to quote with the receipt.
- **Derived** — computed from measurements via a stated formula (busbw from algbw). Quote with
  the formula and inputs.
- **Inferred** — consistent with observations but not directly shown (e.g. "chunking kicked in"
  from a memory drop + time jump). Say "consistent with", name the alternative explanations.
- **Smoke-grade vs defensible** — one message size × 10 iters is a health check; a customer-deck
  bandwidth number needs the proper benchmark (nccl-tests across sizes). Never let a smoke number
  travel without that caveat attached.

## Known observation traps (verified in this repo)

- **Sampled gauges lie on short jobs**: GPU-util read 0% during sub-second forward passes and
  99-100% on long rungs. Don't conclude "idle GPU" from a gauge on a short-step job.
- **The CLI streams node 0 only.** Multi-node claims require checking other nodes:
  `air logs <run_id> --node N`. All nodes report the same hostname (`main.host.local`) — use
  `NODE_RANK`/rank IDs, not hostnames, to distinguish nodes.
- **Docs lag the product in both directions** (features documented that the schema rejects;
  features live that docs call Private Preview). A docs statement is a hypothesis, not evidence.
  Same for Slack: cite channel + date, and treat as "reported", not "verified", until you ran it.
- **Correctness by construction beats spot-checks**: prefer assertions whose success logically
  requires the property (all-reduce of ones summing to world_size proves all ranks participated).

## Reproducibility minimum

Someone re-running your experiment next month needs, committed:

- the `.example.yaml` (schema-current — fold in any quirks you hit) and the exact submit command
  including `--override`s and profile,
- the probe/training script at the SHA that ran (commit before or immediately after the run;
  note the CLI version — `air --version` — findings are version-scoped),
- any dataset/checkpoint identity (OpenML data_id, HF checkpoint filename).

Secrets rail still applies: never dump full `env` in a workload command — filter
(`env | grep -E 'RANK|WORLD|MASTER|...'`) so secret-bearing vars can't land in logs.

## Promotion pipeline

Raw log → `experiments/<family>/run-<id>.log` → finding + receipts in that family's `NOTES.md` →
if it generalizes, promote to `docs/` with the ✅ annotation → if it answers an open-q, update
`docs/04-open-questions.md` in place (✅ ANSWERED / PARTIALLY ANSWERED, date, run id). Claims in
`docs/` without receipts are bugs — fix them when found.

## Pre-submit checklist

- [ ] Success criteria + expected evidence written in NOTES.md
- [ ] Probe asserts its claim and prints a distinctive sentinel
- [ ] Smallest sufficient shape; short `timeout_minutes`; `max_retries: 0` unless testing retries
- [ ] No unfiltered `env` dumps in `command`
- [ ] Dry-run (`air run --dry-run`) passed

## Post-run checklist

- [ ] Raw log archived as `run-<id>.log` in the experiment dir (all nodes if multi-node)
- [ ] Claim→evidence mapping in NOTES.md; numbers labeled measured/derived/inferred
- [ ] ✅ annotations with date + run id + workspace on anything promoted to docs/ or open-qs
- [ ] Example YAML updated with any schema/behavior quirks discovered
- [ ] Committed (git log is the lab notebook)
