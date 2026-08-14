# GPU Broker — training-hub prototype design

Extends the Training Hub into a **submission broker** for a shared reserved GPU pool: the
platform admits or fail-fasts (no queue, no arbitration — verified at two different quota
levels; receipts in experiments/node-acceptance/NOTES.md), so the hub becomes the layer that
queues, gates, attributes, and injects known-good configuration. One workspace in scope.

⚠️ **Quota values are settings, not platform constants.** Workspace GPU quotas are
per-workspace, per-gpuType eng-side settings raised via ES ticket (this workspace's 8xH100
bucket went 4 → 20 mid-UAT). The *durable* platform facts are: fail-fast with no queue,
per-shape node buckets, and an ES-mediated increase path. Never bake a measured quota value
into design or code — model it as current-workspace config that can change under you.

## Requirement → feature map (from the customer-gap review)

| # | Gap | Broker feature |
|---|---|---|
| 1 | capacity is tribal knowledge | **Capacity panel** from declared config: reservation size, per-shape workspace quota (an adjustable ES-mediated setting — display the *current* value, flag staleness after any quota change), per-team quotas — plus live in-flight counts against each |
| 2 | no platform queue | **Submission queue**: requests admitted to the platform only when capacity model says a slot exists; FIFO within team, team-quota-aware; everything else waits in QUEUED with visible position |
| 3 | pools workspace-bound | single-workspace prototype (config names the workspace/profile) |
| 6 | no platform access gating | **App-layer gating**, two modes. *Open mode* (prototype today): only members of a GPU-enabled team may enqueue; per-team shape allowlist; bypassable by direct API — gates the paved road only. *Credential-boundary mode* (target): humans hold zero GPU entitlement; per-team SPs hold entitlements + UC grants, secrets readable only by the app; broker dispatches run-as the team SP (user captured via Apps OBO) — bypass becomes a permission error, platform-enforced by construction. Covers the batch/reserved lane; interactive sessions remain user-run under workspace controls, where platform entitlement remains the eng ask |
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

## Problem classification & prior art (2026-08-06)

**Problem type**: multi-tenant batch-job scheduling with hierarchical fair sharing over
discrete, gang-scheduled slots. The broker is an *admission* layer on top of a placement layer
it doesn't control — the classic two-level scheduler split (Mesos lineage). Closest modern
analog: **Kubernetes Kueue** (quota-aware job queueing above the scheduler); adopt its
vocabulary (nominal quota, cohort borrowing, reclaim) when evolving `hub/queue.py`.

**Why this instance is easier than the general problem** — properties to preserve, because
each one removes a hard subproblem:

- *Single resource dimension* (nodes/GPUs of one class) → weighted **max-min fairness**
  suffices; DRF-style multi-resource machinery is unnecessary.
- *Coarse discrete units* (1× and 8× shapes, pool = N node-slots) → capacity math is counting,
  not bin-packing.
- *Small tenant count* (a handful of teams, tens of concurrent jobs) → policy **legibility**
  beats algorithmic sophistication; every ordering decision must be explainable to the team
  that's waiting.
- *Checkpoint discipline is already mandatory* (7-day cap) → **cooperative preemption**
  (drain at next checkpoint, resubmit) is available — the usually-missing ingredient.
- *Deadlines are known in advance* (release-calendar cadence) → **advance reservations**
  are the right mechanism, not priority escalation.

**The standard solution kit that maps onto the queue** (in adoption order):

1. **Min-guarantee + max-cap with elastic borrowing** per team, reclaim at checkpoint
   boundaries (YARN Capacity Scheduler semantics / Kueue cohorts). Upgrades quotas from
   "declared, not enforced" to "honored on the paved road."
2. **Decayed-usage fairshare ordering** (Slurm multifactor priority) instead of plain
   FIFO-within-team: queue priority falls as recent consumption rises. This turns the
   attribution ledger into the scheduling input.
3. **Priority/QOS lane for urgent work**, backed by reserved headroom or reclaim.
4. **Advance reservations** entered from the release calendar.

**App-layer honesty** (extends the limits below): admission, ordering, borrowing, and
accounting are appropriate at this layer because the hub is the submission path. Bypass
*enforcement*, involuntary preemption, and placement are platform-side (existing eng asks);
reconciliation against the runs list + OTEL keeps bypass visible rather than blockable.

**Anti-pattern**: internal markets/bidding for capacity — solves contention among hundreds of
unknowable tenants, at the cost of the explainability a small org needs. Not for this scale.

## Honest prototype limits

- Gating/quotas in the current prototype are broker-enforced, not platform-enforced:
  direct-API users bypass them; the ledger still *sees* platform-wide usage via runs list.
  This is a property of *open mode*, not of the app layer as such — in credential-boundary
  mode (per-team SPs, broker-held secrets; see requirement row 6) enforcement is
  platform-real for the batch lane. That mode raises the stakes: the app becomes a
  privileged submission gateway (security review accordingly), its availability becomes
  critical path (break-glass escrowed creds; SQLite must go), and interactive sessions
  stay outside it.
- SQLite is single-instance state; fine for a prototype/app-per-workspace, not HA.
- Apps hosting is disabled on the current sandbox org — serve locally
  (`uv run --with-requirements requirements.txt -- streamlit run app.py`) or on a
  workspace with Apps enabled.
