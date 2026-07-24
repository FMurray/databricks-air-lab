# Fleet ops

For the person who owns a shared GPU pool: what you can actually see and control today, and the
tooling this lab built to close the gaps.

## What the platform provides today

Two capabilities are still on the product roadmap (tracked as open-q #15/#16), and they shape how
this job works right now:

1. **Access control** is at the workspace level today — serverless GPU access comes with the
   workspace serverless flag; per-team GPU entitlement is roadmap.
2. **Attribution inside a reserved pool** — reservation pools currently bill as a single
   aggregate record; per-workload tagging is roadmap.

The toolkit below provides operational equivalents in the meantime: declared quotas with
visibility, and attribution built from what the system tables record today.

## Your toolkit

| Want | Tool | Where |
|---|---|---|
| Spend by team / principal, utilization vs. reservation | ready-made SQL builders | [See spend by team](see-spend-by-team.md) |
| A UI: declared quotas, spend, active runs, self-serve YAML submission for your teams | **Training Hub** app | serve command below |
| Attribution with known guarantees | the attribution ladder | [Attribute usage](attribute-usage.md) |
| Structured cross-fleet telemetry (who OOMed this week?) | OTEL → Delta pipe | [Ship telemetry to Delta](../cookbook/ship-telemetry-to-delta.md) |

## Serve the Training Hub

Zero-config demo mode (bundled example teams/quotas), one command from the repo root:

```bash
cd apps/training-hub && uv run --with-requirements requirements.txt -- streamlit run app.py --server.headless true
```

Open http://localhost:8501 — **Fleet** tab for you, **Submit a workload** tab for your teams.
Pointing it at real billing data and deploying it as a Databricks App:
`apps/training-hub/README.md`.

Quotas in the hub are **declared, not enforced** — enforcement arrives with per-team entitlement.
Today's value is making over/under-quota visible, so allocation decisions run on shared data
instead of ad-hoc requests to one person.
