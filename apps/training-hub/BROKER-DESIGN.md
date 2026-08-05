# GPU Broker — training-hub prototype design

Extends the Training Hub into a **submission broker** for a shared reserved GPU pool: the
platform admits or fail-fasts (no queue, no arbitration — verified: 20 submits → 4 admitted,
16 immediate quota FAILs), so the hub becomes the layer that queues, gates, attributes, and
injects known-good configuration. One workspace in scope.

## Requirement → feature map (from the customer-gap review)

| # | Gap | Broker feature |
|---|---|---|
| 1 | capacity is tribal knowledge | **Capacity panel** from declared config: reservation size, per-shape platform quota (e.g. the measured `max=4` admission cap), per-team quotas — plus live in-flight counts against each |
| 2 | no platform queue | **Submission queue**: requests admitted to the platform only when capacity model says a slot exists; FIFO within team, team-quota-aware; everything else waits in QUEUED with visible position |
| 3 | pools workspace-bound | single-workspace prototype (config names the workspace/profile) |
| 6 | no platform access gating | **App-layer gating**: only members of a GPU-enabled team may enqueue; per-team shape allowlist. Honest limit: bypassable by direct API — the broker gates the *paved road*, platform gating remains the eng ask |
| 7 | no per-user/workload chargeback | **Attribution ledger**: every brokered request records team/user/use-case tag/shape/run_id/timestamps; joins to `system.billing.usage` (existing `utils/billing`) and OTEL telemetry for cost views |
| 8 | claim granularity unknown | dispatcher records placement receipts (GPU UUIDs / boot_id via the placement-probe pattern) for brokered runs where enabled |
| 9/10 | env + deps sharp edges | **Recipes injected at submit**: env v5 pin, base-env file reference, AI-env interpreter prelude, vendored-wheels PYTHONPATH — from the receipt-backed patterns in docs/cookbook |
| 11 | multinode = CLI only | brokered submission uses the proven paths: notebook tasks via `runs/submit` (+`compute.hardware_accelerator`), AIR CLI workloads via the vendored `air` CLI subprocess. Never `@distributed`, never persistent-job wrapping of Gen-AI tasks (both receipt-dead) |
| 12 | fleet visibility | existing Fleet tab + queue/ledger views |

## Architecture (prototype)

```
Streamlit UI (tabs: Fleet | Submit | Queue)
        │
hub/queue.py     Broker: enqueue → gate → admission-control → dispatch → track
hub/capacity.py  Capacity model: config buckets vs live in-flight
hub/recipes.py   Env/deps/command recipes (receipt-backed)
hub/config.py    reservation + teams (+ per-shape platform quotas)  [existing, extended]
        │
SQLite (broker.db, app-local)  ── swap-for-Delta interface when catalog storage works
Jobs API runs/submit  /  vendored air CLI
```

State machine per request:
`QUEUED → SUBMITTED → RUNNING → SUCCESS | FAILED | CANCELED` (+ `REJECTED` at gate).
A dispatcher pass (called on UI refresh and by a `tick()` loop) syncs run states via
`jobs/runs/get`, frees capacity, and promotes the queue head.

## Honest prototype limits

- Gating/quotas are broker-enforced, not platform-enforced: direct-API users bypass them.
  The broker's ledger still *sees* platform-wide usage via runs list for the capacity view.
- SQLite is single-instance state; fine for a prototype/app-per-workspace, not HA.
- Apps hosting is disabled on the current sandbox org — serve locally
  (`uv run --with-requirements requirements.txt -- streamlit run app.py`) or on a
  workspace with Apps enabled.
