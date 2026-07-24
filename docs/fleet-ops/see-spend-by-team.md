# See spend by team

Goal: answer "who spent what on GPUs" from `system.billing.usage` without writing SQL from scratch.

The canonical query builders live in `utils/billing/queries.py` — the same module the Training Hub
imports. Don't duplicate the SQL; import or run it.

## From the command line

From the repo root, with a profile pointing at the right workspace:

```bash
export HUB_WAREHOUSE_HTTP_PATH="/sql/1.0/warehouses/<id>"   # must belong to that same workspace
python -m utils.billing.queries by-principal --days 30
```

| Query | Answers |
|---|---|
| `daily` | GPU DBU usage by day |
| `by-principal` | chargeback by identity, with list-price join |
| `attribution` | how much usage has an identity attached at all |
| `tags` | what custom tags exist on GPU usage (are teams tagging?) |
| `utilization` | daily usage vs. the reservation |

`--sql-only` prints the SQL instead of executing — paste it into a dashboard or hand it to a
warehouse you don't run from Python. The runner prints which workspace host it's querying —
**check it**; a wrong ambient profile yields a misleading PERMISSION_DENIED from the wrong
workspace.

## Grants

The reader needs `SELECT` on `system.billing.usage` (and the Training Hub's service principal
needs the same, plus CAN_USE on the warehouse).

## Two numbers to treat carefully

!!! warning
    - **Dollar figures**: the list-price join uses an H100 DBU multiplier that is **unverified
      with commercialization** — treat output as relative weight, not invoice math, until that's
      confirmed.
    - **Unattributed usage**: reservation pools bill aggregate records today (per-workload
      tagging is roadmap), so unattributed spend is expected there — not a sign teams are
      untagged. [Attribute usage](attribute-usage.md) covers how to work with it.
