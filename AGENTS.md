# Agent guide — databricks-air-lab

This repo is built to be driven by agent harnesses (Claude Code, Codex, Genie, …). Humans steer;
agents execute. Optimize your work accordingly: prefer the skills and runnable probes below over
re-deriving procedures, and leave verified facts behind in `docs/` when you learn something new.

## What this repo is

Testing ground + utilities for **Databricks AIR** (AI Runtime — serverless GPU training), driven
by an FDE engagement with a large FinServ customer. Customer-identifying material lives only in
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

(Add new skills for any procedure you had to figure out twice.)

## Conventions

- **Every claim gets a receipt — no exceptions.** Product is Preview/Beta; docs lag; findings feed
  a live customer engagement. A result is "verified" only with run id + workspace + date, quoted
  primary evidence, and the raw log archived as `run-<id>.log` in the experiment dir. Full
  standard (assertion-based probes, measured/derived/inferred labeling, known observation traps,
  pre/post-run checklists): `.claude/skills/experiment-verification/SKILL.md`.
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

## Environment quirks worth knowing

- `air` CLI (v0.1.x, via `uv tool install databricks-air`): `env_variables`/`secrets` are
  TOP-LEVEL YAML fields; `air run` creates SUBMIT_RUN job runs (Jobs UI → "Job runs" tab, never
  the "Jobs" tab); `air -h config` is the schema source of truth.
- Apple Silicon cross-builds: `uv` segfaults under qemu; vendor linux/amd64 wheels on the host
  (`uv pip install --target vendor --python-platform x86_64-unknown-linux-gnu ...`) and COPY.
