# databricks-air-lab

**Start with `AGENTS.md`** — the harness-neutral agent guide (repo map, skills index, safety
rails, environment quirks). This file adds only Claude-specific pointers.

Purpose: test, understand, and build utilities for Databricks AIR (AI Runtime — serverless GPU training).
Driving customer context: an FDE engagement with a large FinServ customer (identity and all
customer-specific details live ONLY in `docs/private/`, starting with `customer-refs.md`).

Read first: `AGENTS.md`, then `docs/01-air-feature-surface.md` (what AIR is), `docs/private/customer-baseline.md`
(customer requirements + prep checklist), `docs/03-workload-matrix.md` (workload-family test plans),
`docs/04-open-questions.md` (numbered — referenced as "open-q #N" throughout the repo).

Project skills in `.claude/skills/` (e.g. `air-otel-telemetry`) encode procedures with their sharp
edges — prefer invoking them over re-deriving from docs/experiments.

Customer-specific material lives in `docs/private/` (gitignored, local-only): baseline, meeting notes,
anything naming the customer. Everything committed to the repo should be anonymized/shareable — refer
to "the customer" or generic personas in committed docs.

Conventions:
- AIR CLI workload YAMLs live in `workloads/`; `*.example.yaml` are templates, copies without
  `.example` are live configs (gitignored if they contain workspace specifics).
- Experiments are organized by workload family under `experiments/`; each gets a NOTES.md capturing
  what was run and observed (verified facts move up into `docs/`).
- Verify docs claims hands-on before repeating them to the customer; docs lag the product (Preview/Beta).
- Internal channels: #ai-runtime-oncall (eng escalation); customer-specific channels are listed
  in `docs/private/customer-refs.md` — never name them in committed files.
