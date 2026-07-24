# Databricks AIR Lab

A lab for **Databricks AIR** (AI Runtime — serverless GPU training, fka Serverless GPU Compute):
hands-on experiments that verify what the product actually does, a **tactical cookbook** built on
those findings, and the **Training Hub** — a web app where users on a shared GPU pool take action
based on who they are (admin vs. team member) and what they want to do.

| Who you are | What you want | Go to |
|---|---|---|
| An agent asked to "serve the site" | The cookbook running on localhost | [Serve the cookbook](#serve-the-cookbook-agents-start-here) |
| A practitioner | Verified recipes: first run, multi-node, Docker, debugging | serve the cookbook → **Getting started** / **Cookbook** tabs (source: `docs/`) |
| A GPU allocation manager | Spend by team, attribution, fleet visibility | cookbook **Fleet ops** tab, or [serve the Training Hub](#serve-the-training-hub-app) |
| An agent working in this repo | Conventions, skills, safety rails | `AGENTS.md` (read it before running anything) |

## Serve the cookbook (agents: start here)

The HTML site is the **AIR Lab Cookbook** — field-tested recipes for practitioners and GPU
allocation managers, routed by who they are and what they want to learn or try. It complements
the [public docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/), it does not
replace them. One command from the repo root, nothing to install (verified 2026-07-22):

```bash
uv run --with mkdocs-material -- mkdocs serve
```

No `uv`? `pip install mkdocs-material && mkdocs serve`.

**Done when:** `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000` prints `200` and the
page titled "AIR Lab Cookbook" opens on a **Pick your path** landing with tabs: Getting started,
Cookbook, Fleet ops, Reference. `docs/private/` (customer material) is excluded via `exclude_docs`
in `mkdocs.yml` — `http://127.0.0.1:8000/private/...` must 404; if it doesn't, stop the server.

## Serve the Training Hub app

The lab's interactive web app (fleet visibility + workload submission for a shared pool). Zero
configuration — falls back to bundled example config and renders demo data (verified 2026-07-22):

```bash
cd apps/training-hub && uv run --with-requirements requirements.txt -- streamlit run app.py --server.headless true
```

**Done when:** HTTP 200 on `http://localhost:8501` with **Fleet** and **Submit a workload** tabs.
Without workspace credentials the Fleet tab's live-data panels show in-app errors — expected.
Full serve/configure/deploy detail: `apps/training-hub/README.md`.

## Layout

| Path | Purpose |
|---|---|
| `docs/` | The cookbook (recipes + reference docs; open questions cited as "open-q #N") |
| `docs/private/` | Customer-specific material — gitignored, local-only |
| `apps/training-hub/` | The web app: fleet visibility + workload submission |
| `workloads/` | AIR CLI workload YAMLs; `*.example.yaml` committed, live copies gitignored |
| `experiments/` | Hands-on tests by family; findings + receipts in each `NOTES.md` |
| `utils/` | Importable modules: billing/attribution queries, visibility helpers |
| `.claude/skills/` | Procedures with the sharp edges encoded (harness-neutral SKILL.md files) |
| `AGENTS.md` | Agent guide: conventions, skills index, safety rails, environment quirks |

## How the repo works

Experiments produce `NOTES.md` findings with receipts (date + run id + archived log); verified
facts get promoted into the cookbook; repeatable procedures become skills; reusable queries become
`utils/` modules the app imports. The git log is the lab notebook. The evidence standard lives in
`.claude/skills/experiment-verification/SKILL.md` and is summarized on the cookbook's
"How to read the receipts" page (`docs/about-receipts.md`).
