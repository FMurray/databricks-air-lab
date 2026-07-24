---
name: experiment-verification
description: Show-your-work standards for experiments in this repo — how an agent demonstrates what a run actually did instead of characterizing it. Load BEFORE running any AIR workload and before writing results into NOTES.md/docs/.
---

# Experiment verification standards

This exists so the human reviewing agent work can see **what actually happened**. The failure
mode it guards against is the agent declaring "great success" — summarizing runs in its own
words, rounding failures up to passes, or asserting properties the output never showed. It is
not a zero-trust forensics system; don't build ceremony, build visibility.

## Core rule: show, don't characterize

Never state a result in your own words without the output that backs it. Every result claim
carries:

- **run identity** — Job Run ID + workspace + date, inline where the claim lives:
  `✅ VERIFIED 2026-07-22, run 505819227973807, e2-demo-field-eng`
- **quoted output** — the actual line(s) the run printed, verbatim, in NOTES.md. If asked
  "where's the evidence" the answer should already be written down.
- For headline results: a short claim→evidence table in NOTES.md. An empty evidence cell means
  the claim doesn't ship.

Report failures exactly as they happened — exit codes, the mechanism (host-OOM kill ≠ CUDA OOM),
what was expected vs observed. A run that failed informatively is a result; a failure narrated
as "mostly working" is the thing this skill exists to prevent.

## Design runs so success is checkable

- **Assert, don't observe.** Probes verify their own claims and print a distinctive sentinel
  that is unreachable unless the assertions passed (`MULTINODE_PROBE_OK`). "Exited 0" is not
  evidence of the property you care about.
- **Write success criteria in NOTES.md before submitting** — what the run should demonstrate and
  what output will show it. This keeps "success" from being defined after seeing the results.
- **Pre-flight locally when feasible** (single-process/CPU mode) before spending GPU money;
  quote the pre-flight output and the command that produced it.
- **Smallest shape that answers the question.** A10 before H100, 1 node before N, 100 steps
  before 500K. Escalate only when the question requires it.

## Where evidence lives: MLflow

**MLflow is the evidence layer** — the platform-recorded workspace run is the source of truth,
and `utils/verification/archive_run.py` mirrors it into the repo-local store
(`sqlite:///experiments/mlflow.db` + `experiments/mlartifacts/`, both committed) so receipts
survive workspace retention and travel with `git clone`:

```
uv run --with mlflow python -m utils.verification.archive_run \
    --profile <profile> --job-run-id <id> --experiment <name> \
    --extra <client-side submission/preflight log> ...
```

The copy carries params, full metric histories, tags, and artifacts, plus provenance tags
(`archive.source_run_id`, `archive.source_workspace`, `archive.archived_at`); `--extra` files
land under `client_logs/` so loose local logs become artifacts OF the linked run instead of
free-floating text. Browse with `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`.

- Log metrics/params from the run; attach files that matter (run logs, result tables, probe
  output) as **run artifacts** so they outlive job-run log retention (~60 days).
- Record the MLflow run ID/URL alongside the Job Run ID in NOTES.md — and the local archive
  run ID once archived.
- NOTES.md is the narrative: quoted excerpts, pointers, conclusions. The repo commits **code,
  predictions, and conclusions — not data**: `run-*.log` / `preflight-*.log` are gitignored
  (local working copies are fine); raw output belongs in MLflow.
- Retrieval for anything not yet in MLflow: `air logs <run_id> [--node N] -p <profile>`.

## Label what kind of number it is

- **Measured** — read directly from output. Quote it.
- **Derived** — computed from measurements (busbw from algbw): state the formula and inputs.
- **Inferred** — consistent with observations but not directly shown: say "consistent with" and
  name alternatives.
- **Smoke-grade vs defensible** — a 10-iter single-size bandwidth check is a health signal; a
  customer-deck number needs the real benchmark (e.g. nccl-tests). Don't let smoke numbers
  travel without that label.

## Known observation traps (all hit in this repo)

- Sampled gauges lie on short jobs: GPU-util read 0% during sub-second steps, 99-100% on long
  rungs. Don't conclude "idle" from a gauge on a short-step job.
- The CLI streams **node 0 only**; check other nodes (`air logs <id> --node N`) before making
  multi-node claims. All nodes report the same hostname — distinguish by `NODE_RANK`/rank IDs.
- Docs lag the product in both directions; Slack is "reported", not "verified". Both are
  hypotheses until you ran it. Note tool versions (`air --version`; torch/NCCL from logs) —
  findings are version-scoped.
- **Log delivery is per-workspace unreliable**: a run can report SUCCESS and stream MLflow
  system metrics while its stdout is unretrievable by every channel (`air logs` streaming +
  download, MLflow artifacts, Jobs API). Verified 2026-07-22: run 938962751074433 on
  fevm-forrest-aws-stable lost all stdout; identical YAML on e2-demo-field-eng (run
  93215537511850) streamed fine. Design probes so critical evidence also lands somewhere
  durable (MLflow params/metrics, a UC volume) — don't stake a finding on stdout alone.

## Pre-submit checklist

- [ ] Success criteria + expected output written in NOTES.md
- [ ] Probe asserts its claim and prints a sentinel
- [ ] Local pre-flight done where feasible, output quoted
- [ ] Smallest sufficient shape; short `timeout_minutes`; `max_retries: 0` unless testing retries
- [ ] No unfiltered `env` dumps in `command` (secrets land in logs)
- [ ] `air run --dry-run` passed

## Post-run checklist

- [ ] Quoted output + run identity in NOTES.md; claim→evidence table for headline results
- [ ] MLflow run ID recorded; artifacts attached if the raw output needs to outlive job-run retention
- [ ] Run archived to the local store (`utils/verification/archive_run.py`, `--extra` for
      client-side logs); local archive run ID added next to the claim in NOTES.md
- [ ] Multi-node claims checked on every node
- [ ] Numbers labeled measured/derived/inferred; failures reported as they happened
- [ ] Generalizable findings promoted to docs/ with ✅ date + run id; open-qs updated in place
- [ ] Example YAML updated with any quirks discovered; committed (code + conclusions, not data)
