# Attribute usage

Goal: attach an owner to every GPU-hour, using the mechanisms available today — each with clearly
understood guarantees.

Attribution is a ladder, not a switch: each rung records identity a different way. Use the
highest rung your context allows, and label anything below it.

## The attribution ladder

| Rung | Mechanism | Guarantee | Status |
|---|---|---|---|
| 1 | Per-principal rows in `system.billing.usage` | platform-recorded | works for on-demand; reservation pools bill aggregate records today (per-workload tagging is roadmap) |
| 2 | `usage_policy_name` on the workload | platform-recorded | what it lands in billing tables is **unverified** (open-q #5) — test before relying on it |
| 3 | Join `air.mlflow_run_id` → `system.lakeflow.job_run_timeline` | platform-recorded creator | ✅ verified; query in `utils/visibility/telemetry_identity.sql` |
| 4 | Per-team service principals for [telemetry](../cookbook/ship-telemetry-to-delta.md) | authenticated (a team's token can only write its tables) | ✅ verified — a downscoped token cannot write another team's table |
| 5 | `air.requester` telemetry attribute | **self-reported** in-container | ✅ works; right for dashboards, not for chargeback |

## Working with aggregate reservation records (rung 1)

Until per-workload tagging inside reserved pools lands (open-q #16):

- Show unattributed spend **as unattributed** — the Training Hub labels it explicitly rather than
  spreading it pro-rata, which keeps the attributed numbers trustworthy.
- Use rung 3 to attribute *runs* even where dollars aggregate: who ran what, when, on which
  accelerator is fully answerable today.
- Declared quotas + visible variance ([Training Hub](index.md)) cover allocation until per-team
  entitlement ships.

## Roadmap items to track

Two upcoming capabilities complete this ladder: server-side principal stamping on Zerobus writes
(upgrades rung 5 from self-reported to platform-recorded) and per-workload tags on reserved-pool
billing records (completes rung 1). Context and status: open-q #15/#16 in
[Open questions](../04-open-questions.md).
