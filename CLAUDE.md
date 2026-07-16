# databricks-air-lab

Purpose: test, understand, and build utilities for Databricks AIR (AI Runtime — serverless GPU training).
Driving customer context: [the customer] [customer-org] FDE engagement.

Read first: `docs/01-air-feature-surface.md` (what AIR is), `docs/02-[the customer]-baseline.md` (customer
requirements + prep checklist), `docs/03-workload-matrix.md` (workload-family test plans),
`docs/04-open-questions.md` (numbered — referenced as "open-q #N" throughout the repo).

Conventions:
- AIR CLI workload YAMLs live in `workloads/`; `*.example.yaml` are templates, copies without
  `.example` are live configs (gitignored if they contain workspace specifics).
- Experiments are organized by workload family under `experiments/`; each gets a NOTES.md capturing
  what was run and observed (verified facts move up into `docs/`).
- Verify docs claims hands-on before repeating them to the customer; docs lag the product (Preview/Beta).
- Internal channels: #ai-runtime-oncall (eng escalation), #[the customer]-fde-[customer-org]-gpus, #[the customer]-model-training-gpu-blocker.
