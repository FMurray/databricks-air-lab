# AIR Lab Cookbook

Field-tested recipes for **Databricks AIR** (AI Runtime — serverless GPU training).

This complements the [public docs](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
rather than repeating them: task-shaped recipes, each verified hands-on against a real workspace,
with the practical details a first run teaches you. Every claim carries a receipt (date + run id),
and recipes note where current behavior differs from published docs — the product is evolving
quickly. See [how to read the receipts](about-receipts.md).

## Pick your path

<div class="grid cards" markdown>

-   :material-rocket-launch: **I train models**

    ---

    ML engineer or researcher: from first cheap probe to multi-node H100s.

    1. [Your first run in 10 minutes](getting-started/index.md)
    2. [Pick your compute](getting-started/pick-your-compute.md) — with measured memory envelopes
    3. [The cookbook](cookbook/index.md) — submit, scale out, observe, debug

-   :material-chart-box: **I manage GPU allocation**

    ---

    Platform lead or FinOps: what you can actually see and control on a shared pool.

    1. [Fleet ops overview](fleet-ops/index.md)
    2. [See spend by team](fleet-ops/see-spend-by-team.md) — ready-made billing queries
    3. [Attribute usage honestly](fleet-ops/attribute-usage.md) — the ladder, gap by gap

</div>

## Featured recipes

<div class="grid cards" markdown>

-   **[Run multi-node training](cookbook/run-multi-node-training.md)**

    ---
    Verified 2×8×H100: torchrun wires straight into the injected env; ~359 GB/s busbw measured.

-   **[Debug a failed run](cookbook/debug-a-failed-run.md)**

    ---
    Symptom → actual cause, from failures we actually hit. Exit 137 is not what you think.

-   **[Ship telemetry to Delta](cookbook/ship-telemetry-to-delta.md)**

    ---
    Fleet-wide SQL over logs and GPU gauges — with end-to-end delivery verification built in.

-   **[Use a custom Docker image](cookbook/use-a-custom-docker-image.md)**

    ---
    The proven podman → Docker Hub → `air register` flow, incl. Apple Silicon cross-build traps.

</div>

!!! note "Product status"
    AIR is in Preview/Beta and improving quickly, so published docs can trail current behavior in
    places. Recipes here are dated and version-scoped (`air` CLI v0.1.x) — check the receipt date
    when in doubt, and treat `air -h config` as the schema source of truth.
