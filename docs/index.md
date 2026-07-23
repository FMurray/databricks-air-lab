# AIR Lab Cookbook

Field-tested recipes for **Databricks AIR** (AI Runtime — serverless GPU training).

This is not a copy of the [public docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/) —
it's the tactical layer on top of them: what actually works, verified on real workspaces, with the
sharp edges the docs don't mention. Every claim carries a receipt (date + run id); where the docs
and our runs disagree, the runs win. See [how to read the receipts](reference/about-receipts.md).

## Pick your path

**I train models** (ML engineer, researcher)

1. [Your first run in 10 minutes](getting-started/index.md) — install the CLI, submit a cheap probe
2. [Pick your compute](getting-started/pick-your-compute.md) — A10 vs H100, with measured memory envelopes
3. [The cookbook](cookbook/index.md) — submit, scale out, observe, debug

**I manage GPU allocation** (platform lead, FinOps)

1. [Fleet ops overview](fleet-ops/index.md) — what you can and can't see today
2. [See spend by team](fleet-ops/see-spend-by-team.md) — ready-made queries over `system.billing.usage`
3. [Attribute usage honestly](fleet-ops/attribute-usage.md) — the attribution ladder, gap by gap

## Featured recipes

- [Run multi-node distributed training](cookbook/run-multi-node-training.md) — verified 2×8×H100, torchrun wires straight in
- [Debug a failed run](cookbook/debug-a-failed-run.md) — symptom → actual cause, from real failures
- [Ship telemetry to Delta tables](cookbook/ship-telemetry-to-delta.md) — SQL over your whole fleet, and the silent-drop bug that eats it
- [Use a custom Docker image](cookbook/use-a-custom-docker-image.md) — including Apple Silicon cross-build traps

!!! note "Product status"
    AIR is Preview/Beta and moves fast — docs lag the product **in both directions** (features
    documented that the schema rejects; features live that docs still call Private Preview).
    Findings here are version-scoped: `air` CLI v0.1.x, checked against the date on each receipt.
