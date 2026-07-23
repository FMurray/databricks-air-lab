# Utilities backlog

Ordered by customer pull (see `docs/private/[the customer]-baseline.md`):

1. **`billing/`** — chargeback/attribution queries on `system.billing.usage`. This is the interim answer
   to customer P0 #3 until per-workload tagging in reserved pools exists.
   - `billing/queries.py` — canonical query builders + runner (`python -m utils.billing.queries by-principal --days 30`);
     imported by `apps/training-hub`. Covers: daily usage, per-principal chargeback w/ list-price join,
     attribution coverage, tag inventory, reservation utilization (H100 DBU multiplier UNVERIFIED — confirm
     with commercialization before quoting $).
   - `billing/air_usage.sql` — original standalone query (field-guide provenance).
2. **`visibility/`** — admin job/compute visibility (the July deliverable): who is running what on which
   accelerator, queue depth on shared pools, utilization vs reservation.
3. **`submit/`** — workload authoring helpers ([customer contact]'s "minor lift" gap): YAML generator/validator,
   sensible defaults per workload family, right-sizing hints (did this need an H100?).
4. **`checkpoint/`** — checkpoint/restart harness for the 7-day runtime cap (resume-aware `max_retries` wrapper).
5. **`audit/`** — GPU access audit: who *can* reach serverless GPU in a workspace (until GPU-only entitlement ships).
