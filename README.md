# Databricks AIR Lab

A lab for **Databricks AIR** (AI Runtime — serverless GPU training, fka Serverless GPU Compute):
hands-on experiments that verify what the product actually does, utilities built on those findings,
and the **Training Hub** — a web app where users on a shared GPU pool take action based on who they
are (admin vs. team member) and what they want to do (see spend, or build and submit a training run).

Driven by an FDE engagement with a large FinServ customer; committed content is anonymized —
customer-specific material lives in `docs/private/` (gitignored).

**Find your section** ([Diátaxis](https://diataxis.fr)): route by who you are and what you want.

| Who you are | What you want | Go to |
|---|---|---|
| An agent asked to "serve the site" | The docs site running on localhost | [Serve the docs site](#serve-the-docs-site-agents-start-here) |
| New to AIR | Learn what it is and see a workload take shape | [Tutorials](#tutorials-learn) |
| Working a task | Run a workload, add an experiment, wire telemetry | [How-to guides](#how-to-guides-do) |
| Mid-task, need a fact | Repo layout, CLI facts, product status | [Reference](#reference-look-up) |
| Deciding or reviewing | Why this repo exists and how it works | [Explanation](#explanation-understand) |

Agents: read `AGENTS.md` next — repo conventions, skills index, safety rails (live GPUs, real money).

## Serve the docs site (agents: start here)

The HTML site is this repo's docs rendered by MkDocs, with a landing page (`docs/index.md`) that
routes visitors by who they are and what they want to learn or try. One command from the repo
root, nothing to install (verified 2026-07-22):

```bash
uv run --with mkdocs-material -- mkdocs serve
```

No `uv`? `pip install mkdocs-material && mkdocs serve`.

**Done when:** `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000` prints `200` and the
page titled "Databricks AIR Lab" opens on **Start here** with a "Who are you, and what do you
want?" routing table. `docs/private/` (customer material) is excluded via `exclude_docs` in
`mkdocs.yml` — `http://127.0.0.1:8000/private/...` must 404; if it doesn't, stop the server.

### Serve the Training Hub app

The lab's interactive web app (fleet visibility + workload submission). Zero configuration —
falls back to bundled example config and renders demo data (verified 2026-07-22):

```bash
cd apps/training-hub && uv run --with-requirements requirements.txt -- streamlit run app.py --server.headless true
```

**Done when:** HTTP 200 on `http://localhost:8501` with **Fleet** and **Submit a workload** tabs.
Without workspace credentials the Fleet tab's live-data panels show in-app errors — expected.
Full serve/configure/deploy detail: `apps/training-hub/README.md`.

## Tutorials (learn)

1. **What AIR is** — read `docs/01-air-feature-surface.md`. Verified against real runs; trumps
   the public docs, which lag the product.
2. **See a workload take shape without spending anything** — serve the Training Hub (above), open
   **Submit a workload**, pick a template, and watch the AIR YAML update live as you edit fields.
3. **Run your first real workload** — copy `workloads/exec-probe.example.yaml` to a live sibling
   (drop `.example`; gitignored), keep it cheap (`GPU_1xA10`, short `timeout_minutes`), then
   `air run -f <file>.yaml` against your sandbox workspace. Runs appear in the Jobs UI under
   **Job runs** (never the "Jobs" tab).

## How-to guides (do)

- **Submit a workload**: template it in the Training Hub or copy a `workloads/*.example.yaml`;
  submit with `air run -f`. Schema source of truth: `air -h config`.
- **Add an experiment**: new dir under `experiments/<family>/` with a `NOTES.md` of what you ran
  and observed; promote facts that generalize into `docs/` with date + run id.
- **Wire telemetry (logs/metrics/GPU util → Delta)**: use the `air-otel-telemetry` skill
  (`.claude/skills/air-otel-telemetry/SKILL.md`) — don't re-derive; the Zerobus sharp edges are
  encoded there.
- **Query spend/attribution**: import from `utils/billing/queries.py` (shared by the Training
  Hub) — don't duplicate SQL.

## Reference (look up)

### Layout

| Path | Purpose |
|---|---|
| `docs/` | Distilled, verified knowledge: feature surface, workload matrix, open questions (cited as "open-q #N"), logging/observability |
| `docs/private/` | Customer-specific material — gitignored, local-only |
| `apps/training-hub/` | The web app: fleet visibility + workload submission for a shared pool |
| `workloads/` | AIR CLI workload YAMLs; `*.example.yaml` committed, live copies gitignored |
| `experiments/` | Hands-on tests by family: `foundation-models/`, `rl/`, `classic-ml/`, `multi-language/`, `docker-otel-zerobus/` — findings in each `NOTES.md` |
| `utils/` | Importable modules: billing/attribution queries, visibility helpers |
| `AGENTS.md` | Harness-neutral agent guide: conventions, skills, safety rails, environment quirks |

### Product status (as verified here — Preview/Beta, changes fast)

- Single-node AIR: Public Preview. `@distributed` (multi-GPU notebook API): Beta. Custom Docker: Beta.
- Accelerators: `GPU_1xA10`, `GPU_1xH100`, `GPU_8xH100` (multi-node only via CLI, e.g. 16×H100 = 2 nodes)
- Max workload runtime: 7 days (checkpoint + restart for longer)
- `air` CLI v0.1.x (`uv tool install databricks-air`): `env_variables`/`secrets` are top-level
  YAML fields; `air run` creates SUBMIT_RUN job runs

### Links

- Public docs: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
- Internal: go/air/enablement (field guide) · go/sgc-deck · `#ai-runtime-oncall` (eng escalation)
- Customer channels and baseline: see `docs/private/` (local-only)

## Explanation (understand)

**Why this repo exists.** An FDE engagement stresses AIR across heterogeneous workloads — tabular
foundation models, LLM fine-tuning, RL, classic ML (XGBoost), multi-language (Java) training — on
a shared reserved GPU pool. The product is Preview/Beta and its docs lag, so nothing is asserted
to the customer that hasn't been verified hands-on here.

**Why the Training Hub.** Inside a reserved pool the product doesn't yet do per-team access
control or chargeback, so allocation runs through one human today. The hub replaces that
operationally — declared quotas and spend visibility for admins, safe template-based submission
for team members. Rationale and known gaps: `apps/training-hub/README.md`.

**How the repo works.** Experiments produce `NOTES.md` observations; verified facts get promoted
to `docs/` with dates and run ids; repeatable procedures become skills; reusable queries become
`utils/` modules the app imports. The git log is the lab notebook. Open questions are numbered in
`docs/04-open-questions.md` and cited as "open-q #N" so gaps stay traceable.
