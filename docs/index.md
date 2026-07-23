# Start here

This site is the front door to the **Databricks AIR Lab**: verified knowledge about Databricks AIR
(AI Runtime — serverless GPU training), and the actions you can take with it. Pick the row that
matches who you are and what you want, and follow it ([Diátaxis](https://diataxis.fr)).

## Who are you, and what do you want?

| Who you are | What you want | Do this |
|---|---|---|
| New to AIR | **Learn** what it actually is (verified, not marketing) | Read [AIR feature surface](01-air-feature-surface.md) |
| ML engineer with a workload in mind | **Learn** whether AIR fits it and where the sharp edges are | Find your family in the [workload × AIR fit matrix](03-workload-matrix.md) |
| Anyone, no workspace access needed | **Try** building an AIR workload YAML risk-free | Serve the Training Hub (below), open **Submit a workload**, pick a template |
| ML engineer with sandbox access | **Try** a real GPU run cheaply | Follow [your first real run](#try-your-first-real-run) below |
| Platform lead / admin of a shared pool | **Try** fleet visibility: quotas, spend by team, active runs | Serve the Training Hub (below), open the **Fleet** tab |
| Anyone instrumenting a workload | **Try** shipping logs/metrics/GPU util to Delta tables | Use the `air-otel-telemetry` skill (`.claude/skills/air-otel-telemetry/SKILL.md` in the repo) |
| Debugging or reviewing | **Learn** what's still unverified or broken | Check the numbered [open questions](04-open-questions.md) ("open-q #N" across the repo) |
| Wiring up tracking | **Learn** how native logging works on AIR | Read [logging & observability](05-logging-observability.md) |
| An agent working in this repo | **Learn** the conventions and safety rails first | Read `AGENTS.md` at the repo root |

## Try: the Training Hub

The lab's web app for teams sharing a reserved GPU pool. From the repo root:

```bash
cd apps/training-hub && uv run --with-requirements requirements.txt -- streamlit run app.py --server.headless true
```

Then open <http://localhost:8501>. Zero configuration needed — it renders demo data until you
point it at a real workspace (`apps/training-hub/README.md` has the how-to).

## Try: your first real run

!!! warning
    `air run` submits real GPU workloads that cost real money. Smoke-test cheap: `GPU_1xA10`,
    short `timeout_minutes`, your own sandbox workspace — never a customer workspace.

```bash
uv tool install databricks-air                       # the air CLI (v0.1.x)
cp workloads/exec-probe.example.yaml workloads/exec-probe.yaml   # live copies are gitignored
air run -f workloads/exec-probe.yaml
```

Your run appears in the Jobs UI under **Job runs** (never the "Jobs" tab). Schema questions:
`air -h config` is the source of truth.
