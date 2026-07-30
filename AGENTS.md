# Agent guide — databricks-air-lab

This repo is built to be driven by agent harnesses (Claude Code, Codex, Genie, …). Humans steer;
agents execute. Optimize your work accordingly: prefer the skills and runnable probes below over
re-deriving procedures, and leave verified facts behind in `docs/` when you learn something new.

## What this repo is

Testing ground + utilities for **Databricks AIR** (AI Runtime — serverless GPU training).
Customer-identifying material lives only in
`docs/private/` (gitignored); committed content must stay anonymized ("the customer").

## Read in this order

1. `docs/01-air-feature-surface.md` — what AIR is; verified against real runs, trumps public docs
2. `docs/03-workload-matrix.md` — workload families (FM/RL/classic-ML/multi-language) + test plans
3. `docs/04-open-questions.md` — numbered; cited as "open-q #N" across the repo
4. `docs/05-logging-observability.md` — native logging model (MLflow/job runs/log artifacts)

## Skills (procedures with the sharp edges encoded)

Skills live in `.claude/skills/<name>/SKILL.md`. Claude Code loads them automatically; other
harnesses: read the SKILL.md and follow it — they're written harness-neutral, with bundled
scripts/assets referenced by relative path.

| Skill | Use when |
|---|---|
| `experiment-verification` | **BEFORE any `air run` and before writing results into NOTES.md/docs/** — evidence standards, receipts, archiving, promotion pipeline |
| `air-otel-telemetry` | wiring logs/metrics/GPU telemetry from AIR workloads into Delta via Zerobus OTLP; debugging "export OK but zero rows" |
| `acceptance-report` | adding/changing the plain-English CHECK/VERDICT report a workload prints; carries the canonical renderer to copy (never import) |

(Add new skills for any procedure you had to figure out twice.)

## Conventions

- **No local tmp artifacts — ever.** Every test asset (probe YAMLs, diagnostic notebooks,
  helper scripts) is checked into the repo the moment it's used, so any teammate can pick up
  the work without this machine. Diagnostic workloads live in `workloads/probes/`;
  notebook sources in `utils/verification/`. If it ran against a workspace, it's in git.
- **Ask testers where they want evidence delivered** before a verification round: Google Doc,
  Slack, or MLflow (the default when nobody says otherwise). Google Doc links are
  customer-identifying and must NEVER be committed to the repo — they live in `docs/private/`
  or stay in the delivery channel itself; committed docs reference receipts by run id +
  MLflow experiment only.

- **Show, don't characterize — every result claim carries its receipt.** Product is Preview/Beta;
  docs lag; findings feed a live customer engagement. A result is "verified" only with run id +
  workspace + date and the quoted output that backs it; MLflow (local, synced to managed) is the
  evidence layer for raw output — the repo commits code, predictions, and conclusions, not data.
  Full standard (assertion-based probes, measured/derived/inferred labeling, observation traps,
  pre/post-run checklists): `.claude/skills/experiment-verification/SKILL.md`.
- **Every MLflow experiment carries a description.** The Description panel (`mlflow.note.content`
  tag) is the audit trail teammates see first: what it tests, how to reproduce it, pass criteria,
  observed results with run IDs. Repro pointers are **workspace links** into the
  `/Workspace/Shared/databricks-air-lab` mirror (readers are in the MLflow UI, not on your
  machine), and CLI-submitted experiments say so explicitly — most experiments here have no
  notebook source. Full contract: `.claude/skills/experiment-verification/SKILL.md`.
- Workload YAMLs: `workloads/*.example.yaml` are committed templates; live copies (workspace
  specifics) are gitignored siblings without `.example`.
- Experiments: one dir per family under `experiments/`, findings in its `NOTES.md`; facts that
  generalize get promoted to `docs/`.
- Shared SQL/queries live in `utils/` as importable modules (see `utils/billing/queries.py`
  pattern) — the Training Hub app (`apps/training-hub/`) imports them; don't duplicate SQL.
- Commit early and often with descriptive messages; the git log is the lab notebook.

## Safety rails (live systems!)

- `air run` submits real GPU workloads that cost real money. Smoke tests: `GPU_1xA10`,
  short `timeout_minutes`. Never submit to a customer workspace.
- Never echo tokens/secrets into transcripts, files, or job logs (no `env | sort` in workload
  commands — secrets are env vars there). Secret material goes in Databricks secret scopes or
  gitignored local files (`~/.air-lab-sp-secret.json` pattern).
- Workspace profiles (`~/.databrickscfg`): `fevm-forrest` = personal sandbox (admin, safe);
  `e2-demo-field-eng` = shared demo workspace (no admin, be polite); anything else — ask.
- Local container builds use **podman**, not Docker Desktop (license restrictions;
  cross-build notes in `experiments/docker-otel-zerobus/NOTES.md`).
- **Distributed multi-node = air CLI only.** The notebook `@distributed` API is not
  production-ready for the customer engagement driving this repo — never build UAT items,
  demos, or recommendations on it; document it only as out-of-scope context.

## Environment quirks worth knowing

- `air` CLI (v0.1.x, via `uv tool install databricks-air`): `env_variables`/`secrets` are
  TOP-LEVEL YAML fields; `air run` creates SUBMIT_RUN job runs (Jobs UI → "Job runs" tab, never
  the "Jobs" tab); `air -h config` is the schema source of truth.
- Apple Silicon cross-builds: `uv` segfaults under qemu; vendor linux/amd64 wheels on the host
  (`uv pip install --target vendor --python-platform x86_64-unknown-linux-gnu ...`) and COPY.
