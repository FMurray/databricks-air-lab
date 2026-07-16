# Utilities backlog

Ordered by [the customer] pull (see `docs/02-[the customer]-baseline.md`):

1. **`billing/`** — chargeback/attribution queries on `system.billing.usage`. This is the interim answer
   to [the customer] P0 #3 until per-workload tagging in reserved pools exists. Start: `billing/air_usage.sql`.
2. **`visibility/`** — admin job/compute visibility (the July deliverable): who is running what on which
   accelerator, queue depth on shared pools, utilization vs reservation.
3. **`submit/`** — workload authoring helpers ([customer contact]'s "minor lift" gap): YAML generator/validator,
   sensible defaults per workload family, right-sizing hints (did this need an H100?).
4. **`checkpoint/`** — checkpoint/restart harness for the 7-day runtime cap (resume-aware `max_retries` wrapper).
5. **`audit/`** — GPU access audit: who *can* reach serverless GPU in a workspace (until GPU-only entitlement ships).
