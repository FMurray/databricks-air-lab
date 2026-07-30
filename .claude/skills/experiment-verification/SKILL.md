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
  evidence of the property you care about. (This governs how evidence is *produced*; the
  `acceptance-report` skill governs how the in-run verdict is *presented* to a confirmer — its
  `record-don't-raise` rule relaxes the report row but keeps sentinels pass-gated, so the two
  don't conflict.)
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

## Every experiment carries a description (reproducibility contract)

An MLflow experiment with no description is an orphan: a teammate opening the workspace UI sees
a name like `air-lab-envvar-probe` and a pile of runs with no way to rerun or trust any of it.
Set the description when the experiment is created (in practice: right after the first
submission), and update **Observed** whenever a headline result changes.

**Contract — four parts, in this order:**

1. **What it tests** — one or two sentences, the finding/purpose first. State the submission
   path explicitly: only notebook-sourced experiments have a notebook behind them; most here are
   **"submitted via the `air` CLI from a local repo checkout"** — say so, or a teammate will
   hunt for a notebook that doesn't exist.
2. **Repro** — the driving asset, linked (see link rules below): the `*.example.yaml` template +
   the script it invokes for CLI workloads, the workspace notebook for notebook experiments.
   Plus shape (`GPU_1xA10`, …), environment version, and any env vars that matter. One-off
   inline `air run` diagnostics with no committed YAML must say exactly that and link the run
   table that documents them (e.g. `docs/06-uat-suite.md`).
3. **Pass** — the success criteria / sentinel, where they exist.
4. **Observed** — results with run IDs and dates, failures included, workspace-scoped
   ("on this workspace…" vs "on an open workspace…").

**Link rules:** readers are in the MLflow UI, so repro pointers are **workspace links** into the
repo mirror (`/Workspace/Shared/databricks-air-lab`), never local paths. Markdown renders in the
Description panel. The only format verified to open (2026-07-24, fe-sandbox-mkazia-lw2, human
click-tested) is the **object-ID route**:

```
FILE       https://<host>/editor/files/<object_id>?o=<workspace_id>
NOTEBOOK   https://<host>/editor/notebooks/<object_id>?o=<workspace_id>
DIRECTORY  https://<host>/browse/folders/<object_id>?o=<workspace_id>
```

Get the ID + type from `databricks api get /api/2.0/workspace/get-status?path=<url-encoded-path>`
(doubles as the existence check). Both path-style forms **404** on this workspace UI:
`https://<host>/Workspace/<path>` and the legacy hash `https://<host>/?o=…#workspace/<path>`.
Trap: the air CLI (v0.1.x) writes a default run description ("Workload configuration:
[training_config.yaml](/Workspace/…)") whose href is the config's **FUSE mount path** on the
compute node — correct as a filesystem path, 404 as a UI link. Product bug, raised with AIR
eng; don't hand-patch CLI-written run tags — the YAML itself is reachable via
`get-status` on the same path.
Object IDs change if a re-sync deletes/recreates the object — spot-check description links after
syncing the mirror. When unsure a link is right, have a human open it before shipping it.

Only committed content is mirrored — live YAMLs (non-`.example`) are not there; link the
`.example.yaml` template and say "copy + fill in workspace fields"; re-sync the mirror if the
asset is new.

**Exact-submission provenance:** every `air run` uploads its launch dir to
`/Users/<submitter>/.air/cli_launch/<experiment_name>/<run-name>_<id>/` — `training_config.yaml`
(the exact YAML as submitted), `requirements.yaml`, `command.sh`, and `git_diff.patch`. Link the
experiment's launch folder from the description as the authoritative record of what was
submitted (caveat: it lives under the submitter's user dir, so teammates may need access
granted, or mirror the configs to `/Shared`). ⚠️ **`git_diff.patch` is a privacy trap**: it
captures your full uncommitted diff at submission time, which can include content that was
later anonymized or was never meant to leave `docs/private/` context. Never mirror or share
launch dirs without scanning them (customer identifiers, tokens) — and remember they exist at
all: submitting from a dirty tree publishes that dirt to the workspace.

**Setting it** — the description is the `mlflow.note.content` experiment tag; setting it again
overwrites:

```
databricks api post /api/2.0/mlflow/experiments/set-experiment-tag --profile <profile> \
  --json '{"experiment_id": "<id>", "key": "mlflow.note.content", "value": "<markdown>"}'
```

Audit a workspace for undescribed experiments:

```
databricks api post /api/2.0/mlflow/experiments/search --profile <profile> \
  --json '{"max_results": 100}' | jq -r '.experiments[]
  | select(((.tags // []) | map(select(.key == "mlflow.note.content")) | length) == 0) | .name'
```

Descriptions are workspace-visible: the anonymization rule applies (no customer identifiers).

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
- [ ] Experiment description (`mlflow.note.content`) set/updated — what-it-tests + Repro
      (workspace links, submission path stated) + Pass + Observed
- [ ] Run archived to the local store (`utils/verification/archive_run.py`, `--extra` for
      client-side logs); local archive run ID added next to the claim in NOTES.md
- [ ] Multi-node claims checked on every node
- [ ] Numbers labeled measured/derived/inferred; failures reported as they happened
- [ ] Generalizable findings promoted to docs/ with ✅ date + run id; open-qs updated in place
- [ ] Example YAML updated with any quirks discovered; committed (code + conclusions, not data)
