"""AIR / serverless-GPU billing queries against system tables.

Single source of truth for the chargeback/attribution SQL (interim answer to the
customer's P0: per-workload tagging in reserved pools doesn't exist yet). Used from
notebooks, the terminal (`python -m utils.billing.queries`), and the training-hub app.

Builders return SQL strings; `run()` executes them (needs databricks-sql-connector +
databricks-sdk, imported lazily so the builders work anywhere).

Filter rationale: product_features identifies AIR usage regardless of SKU naming era
(ai_runtime vs serverless_gpu keys) — see billing/air_usage.sql provenance (field guide).
Known SKUs seen on this engagement:
  ENTERPRISE_MODEL_TRAINING_US_EAST_N_VIRGINIA
  ENTERPRISE_MODEL_TRAINING_SERVERLESS_GPU_COMPUTE_PROVISIONED_CAPACITY  (reserved pool)

Reserved vs on-demand (source of truth: go/airuntime-field-billing-faq, CONFIRMED status):
  reserved  -> compute_type=RESERVED_COMPUTE (the SKU is derived from it: ..._PROVISIONED_CAPACITY,
               plus a _C suffix for Azure-on-AWS; reserved usage before 2025-12-17 billed under the
               plain MODEL_TRAINING SKU), usage_metadata.ai_runtime_pool_id set
  on-demand -> regional ENTERPRISE_MODEL_TRAINING_* sku, compute_type=ON_DEMAND_COMPUTE,
               usage_metadata.ai_runtime_workload_id set (CLI) / serverless_gpu key (notebooks)
Usage bills as reserved only once billing whitelists the pool's channel, and only inside the
contract window (universe: ReservedAirChannelConfig + the airReservedChannels SAFE flag); before
that, pool usage bills ON_DEMAND (what the 2026-07-25 20-node sweep showed on 2026-07-30).

Reserved rows are per NODE, not per workload: the Serverless Platform builds node metering with no
metering record (ServerlessComputeUsageLogger.getNodeUsage), and billing drops the per-pod reserved
records that do carry the workload's usage policy (AiRuntimeLogPreProcessor.shouldDropWithInfo).
So reserved rows have NULL usage_policy_id, no usage-policy tags, and a placeholder provisioner
created_by instead of a user. Verified 2026-10-12 on the engagement account: usage_policy_id NULL
on 100% of RESERVED_COMPUTE rows, while on-demand rows carry the policy's tags. Per-workload
attribution on reserved capacity therefore comes from audit, not billing (reserved_job_appspace).
Reservations are guaranteed capacities keyed by (workspace, accelerator): workloads that omit
provisioned_capacity_id are auto-placed on the workspace's active reservation, so there is no
single pool compute id to filter audit or the Jobs API by.

Schema migration (relates to open-q #5): the newer `serverless_gpu` attribution schema has since
landed. Verified 2026-09-02 on an internal serverless-GPU sandbox account,
`product_features.serverless_gpu.workload_type` (SGC_WORKLOAD_NOTEBOOK / SGC_WORKLOAD_JOBS) now
coexists with the older `ai_runtime` rows over the same window (not a clean date cutover — both
schemas emit concurrently, workspace/flag-gated). The new schema is what carries
`usage_metadata.notebook_id`, so it is what makes *interactive notebook* attribution possible
(see notebook_attribution); old `ai_runtime` rows carry identity but no notebook_id.
"""

from __future__ import annotations

import os
import re

AIR_PREDICATE = """(product_features.ai_runtime.compute_type IS NOT NULL
       OR product_features.serverless_gpu.workload_type IS NOT NULL)"""

# DBU emission per GPU-hour. H100 value is the account team's working number
# (2026-07-22 thread) — UNVERIFIED against billing; A10 unknown. Confirm with the
# commercialization team before quoting $ to the customer. Reserved-capacity SKUs may
# bill flat rather than per-DBU (under investigation) — treat utilization estimates
# built on these as directional.
DBU_PER_GPU_HOUR = {"H100": 8.884, "A10": None}

# AIR rows carry ONLY created_by (verified 2026-07-30: 1530/1530 rows on the engagement
# account — run_as/owned_by always NULL; matches the field billing FAQ schemas). Without
# created_by in the coalesce, everything reads "unattributed".
_IDENTITY = ("COALESCE(identity_metadata.run_as, identity_metadata.owned_by, "
             "identity_metadata.created_by)")

# Reserved/on-demand per go/airuntime-field-billing-faq; SKU is the primary key (survives
# compute_type gaps), the enum is the cross-check.
_CAPACITY_BUCKET = """CASE
  WHEN sku_name LIKE '%PROVISIONED_CAPACITY%'
       OR product_features.ai_runtime.compute_type = 'RESERVED_COMPUTE' THEN 'reserved'
  WHEN product_features.ai_runtime.compute_type = 'ON_DEMAND_COMPUTE'  THEN 'on_demand'
  WHEN product_features.serverless_gpu.workload_type IS NOT NULL       THEN 'on_demand_notebook'
  ELSE 'unknown' END"""

# List-price join: contract rates differ from list; good for relative comparison,
# labeled "est_list_cost" everywhere to avoid being read as an invoice.
_LIST_PRICE_JOIN = """LEFT JOIN system.billing.list_prices p
  ON u.sku_name = p.sku_name
 AND u.cloud = p.cloud
 AND u.usage_start_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)"""


def _days(days: int) -> int:
    days = int(days)
    if not 0 < days <= 730:
        raise ValueError(f"days out of range: {days}")
    return days


def air_usage_daily(days: int = 30) -> str:
    """Daily AIR DBUs at date × workspace × SKU × principal × job grain."""
    return f"""
SELECT
  usage_date,
  workspace_id,
  sku_name,
  {_IDENTITY} AS principal,
  usage_metadata.job_id AS job_id,
  COALESCE(product_features.ai_runtime.compute_type,
           product_features.serverless_gpu.workload_type) AS workload_type,
  SUM(usage_quantity) AS dbus
FROM system.billing.usage
WHERE {AIR_PREDICATE}
  AND usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY ALL
ORDER BY usage_date DESC"""


def by_capacity_bucket(days: int = 30) -> str:
    """Reserved vs on-demand: is AIR spend landing on the reservation or ad hoc?

    Bucket per the field billing FAQ (SKU + compute_type); pool/workload ids shown so a
    reserved row can be tied to its pool. Reserved rows are per node (module docstring), so
    'principals' in the reserved bucket counts the placeholder provisioner, not real users.
    """
    return f"""
SELECT
  {_CAPACITY_BUCKET} AS capacity_bucket,
  sku_name,
  usage_metadata.ai_runtime_pool_id AS pool_id,
  COUNT(DISTINCT usage_metadata.ai_runtime_workload_id) AS cli_workloads,
  COUNT(DISTINCT {_IDENTITY}) AS principals,
  SUM(usage_quantity) AS dbus,
  MIN(usage_date) AS first_seen,
  MAX(usage_date) AS last_seen
FROM system.billing.usage
WHERE {AIR_PREDICATE}
  AND usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY ALL
ORDER BY dbus DESC"""


def by_principal(days: int = 30) -> str:
    """Chargeback rollup per principal, with estimated list cost."""
    return f"""
SELECT
  COALESCE(u.identity_metadata.run_as, u.identity_metadata.owned_by,
           u.identity_metadata.created_by) AS principal,
  SUM(u.usage_quantity) AS dbus,
  SUM(u.usage_quantity * CAST(p.pricing.default AS DOUBLE)) AS est_list_cost_usd,
  COUNT(DISTINCT u.usage_metadata.job_id) AS jobs,
  MAX(u.usage_date) AS last_active
FROM system.billing.usage u
{_LIST_PRICE_JOIN}
WHERE {AIR_PREDICATE}
  AND u.usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY 1
ORDER BY dbus DESC"""


def attribution_coverage(days: int = 30) -> str:
    """How much AIR spend is attributable at all — the P0 evidence query.

    Reserved rows are per-node aggregates with no policy tags (module docstring, open-q #5/#16);
    this measures the identity/tag coverage gap rather than assuming it.
    """
    return f"""
SELECT
  usage_date,
  CASE WHEN {_IDENTITY} IS NULL THEN 'unattributed' ELSE 'attributed' END AS identity_status,
  CASE WHEN size(custom_tags) > 0 THEN 'tagged' ELSE 'untagged' END AS tag_status,
  SUM(usage_quantity) AS dbus
FROM system.billing.usage
WHERE {AIR_PREDICATE}
  AND usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY ALL
ORDER BY usage_date DESC"""


def tag_inventory(days: int = 30) -> str:
    """Which custom_tags keys/values actually land on AIR usage rows (open-q #5)."""
    return f"""
SELECT
  t.key AS tag_key,
  t.value AS tag_value,
  SUM(u.usage_quantity) AS dbus,
  COUNT(DISTINCT u.usage_metadata.job_id) AS jobs
FROM system.billing.usage u
LATERAL VIEW explode(u.custom_tags) t AS key, value
WHERE {AIR_PREDICATE}
  AND u.usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY 1, 2
ORDER BY dbus DESC"""


def reservation_utilization_daily(
    days: int = 30,
    reserved_gpus: int = 160,
    dbu_per_gpu_hour: float = DBU_PER_GPU_HOUR["H100"],
) -> str:
    """Estimated GPU-hours vs reserved capacity per day, split reserved vs on-demand.

    Directional only until dbu_per_gpu_hour is verified (see DBU_PER_GPU_HOUR note).
    reserved_gpus: nodes × GPUs-per-node (e.g. 20 × 8 = 160).
    ⚠️ est_pct_of_reservation is computed from ALL AIR GPU DBUs, because pool usage bills
    on_demand until billing whitelists the reservation (module docstring). Once reserved rows
    appear, read reserved_dbus as the true reservation draw and on_demand_dbus as other
    accelerators / pre-window usage.
    """
    rate = float(dbu_per_gpu_hour)
    return f"""
SELECT
  usage_date,
  SUM(usage_quantity) AS dbus,
  SUM(CASE WHEN {_CAPACITY_BUCKET} = 'reserved' THEN usage_quantity ELSE 0 END) AS reserved_dbus,
  SUM(CASE WHEN {_CAPACITY_BUCKET} <> 'reserved' THEN usage_quantity ELSE 0 END) AS on_demand_dbus,
  ROUND(SUM(usage_quantity) / {rate}, 1) AS est_gpu_hours,
  ROUND(100 * SUM(usage_quantity) / {rate} / ({int(reserved_gpus)} * 24), 1) AS est_pct_of_reservation
FROM system.billing.usage
WHERE {AIR_PREDICATE}
  AND usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY usage_date
ORDER BY usage_date DESC"""


def notebook_attribution(days: int = 30) -> str:
    """Which *notebooks* ran on serverless GPU, and who ran them.

    Interactive notebook attaches create no job, so the Jobs API and the ai-compute-manager
    workloads API don't see them — billing does, but only on the newer serverless_gpu schema:
    workload_type = SGC_WORKLOAD_NOTEBOOK carries usage_metadata.notebook_id, and identity is in
    identity_metadata. The older ai_runtime schema (ON_DEMAND_COMPUTE) carries identity but NO
    notebook_id, so confirm the target workspace has cut over before relying on this (module
    docstring). Verified 2026-09-02 on an internal sandbox account: 648 distinct notebooks across
    235 principals over 90d, account-wide.

    notebook_id is a numeric object id, not a path — resolve it via notebook_paths_from_audit.
    """
    return f"""
SELECT
  usage_metadata.notebook_id AS notebook_id,
  {_IDENTITY} AS principal,
  COUNT(*) AS rows,
  SUM(usage_quantity) AS dbus,
  MIN(usage_date) AS first_seen,
  MAX(usage_date) AS last_seen
FROM system.billing.usage
WHERE product_features.serverless_gpu.workload_type = 'SGC_WORKLOAD_NOTEBOOK'
  AND usage_metadata.notebook_id IS NOT NULL
  AND usage_date >= date_sub(current_date(), {_days(days)})
GROUP BY ALL
ORDER BY dbus DESC"""


def notebook_paths_from_audit(days: int = 30) -> str:
    """Resolve serverless-GPU notebook_ids to workspace paths + attach identity.

    billing.usage carries only the numeric notebook_id; the attach event in system.access.audit
    (service_name='notebook', action_name='attachNotebook') carries the path, the user, and the
    timestamp. Join key: usage.notebook_id == audit.request_params.notebookId. Audit alone can't
    tell GPU from non-GPU (the attach is generic; clusterId isn't labeled) — the billing side is
    the GPU filter. Coverage is partial in any fixed window: attachNotebook fires once while usage
    accrues daily, so widen `days` to raise it. Verified 2026-09-02: 24/230 GPU notebooks resolved
    to a path in a 30d window. Needs SELECT on system.access.audit in addition to billing.usage.
    """
    d = _days(days)
    return f"""
WITH gpu_nb AS (
  SELECT DISTINCT usage_metadata.notebook_id AS notebook_id,
         {_IDENTITY} AS bill_principal
  FROM system.billing.usage
  WHERE product_features.serverless_gpu.workload_type = 'SGC_WORKLOAD_NOTEBOOK'
    AND usage_metadata.notebook_id IS NOT NULL
    AND usage_date >= date_sub(current_date(), {d})
),
attach AS (
  SELECT request_params.notebookId AS notebook_id,
         MAX(request_params.path) AS path,
         MAX(user_identity.email) AS attach_user,
         MAX(event_time) AS last_attach
  FROM system.access.audit
  WHERE service_name = 'notebook' AND action_name = 'attachNotebook'
    AND event_date >= date_sub(current_date(), {d})
  GROUP BY 1
)
SELECT g.notebook_id, a.path, g.bill_principal, a.attach_user, a.last_attach
FROM gpu_nb g
LEFT JOIN attach a ON g.notebook_id = a.notebook_id
ORDER BY (a.path IS NULL), g.notebook_id"""


def _ident(value: str, pattern: str, what: str) -> str:
    # Interpolated into SQL (run() has no parameter binding), so allow a fixed character set only.
    if not re.fullmatch(pattern, value or ""):
        raise ValueError(f"bad {what}: {value!r}")
    return value


def reserved_job_appspace(
    workspace_id: int,
    since: str,
    accelerator: str = "GPU_8xH100",
    tag_key: str = "SealUsecase",
    billing_days: int = 90,
) -> str:
    """Jobs that ran on a reservation, with their usage policy and its tags (audit + billing).

    Reserved billing rows carry no policy (module docstring), so the job → policy link comes from
    audit: jobs create/reset/update/submitRun events whose request has an ai_runtime_task on the
    reserved accelerator, in the reservation's workspace, at/after its billing-window start. The
    policy's tags come from that policy's ON-DEMAND billing rows — there is no system table of
    policies. Billing custom_tags mix policy, job and workspace tags, so only key=value pairs present
    on EVERY row for the policy are kept: job tags vary by job and drop out; workspace tags
    (e.g. ManagedBy, WorkspaceId) are constant and stay in policy_tags.

    Read with care:
      - billing_jobs = 1 → one job's own tags are also constant, so policy_tags may include them;
      - NULL policy_tags → the policy has no on-demand usage in billing_days; get its tags from
        GET /api/2.0/serverless-policies/{policy_id} instead;
      - only jobs with a policy-bearing event since `since` appear (older jobs: move `since` back);
      - the accelerator match assumes every job on that accelerator in the workspace lands on the
        reservation (true when bursting is off and the reservation has no group ACL).

    tag_key names the per-use-case policy tag surfaced as `appspace`. Verified 2026-10-12 on the
    engagement workspace (returned policy tags for reserved jobs); not yet checked against the API.
    """
    from datetime import datetime

    ws = int(workspace_id)
    # Normalize to the full form the query was tested with ('YYYY-MM-DD HH:MM:SS').
    ts = datetime.fromisoformat(
        _ident(since, r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?", "since")
    ).strftime("%Y-%m-%d %H:%M:%S")
    acc = _ident(accelerator, r"[A-Za-z0-9_]+", "accelerator")
    key = _ident(tag_key, r"[A-Za-z0-9_.:-]+", "tag_key")
    return f"""
WITH audit_jobs AS (
  SELECT
    action_name,
    COALESCE(request_params['job_id'],
             get_json_object(response.result, '$.job_id'),
             get_json_object(response.result, '$.run_id')) AS job_or_run_id,
    user_identity.email AS actor,
    COALESCE(request_params['usage_policy_id'],
             request_params['budget_policy_id'],
             get_json_object(request_params['new_settings'], '$.usage_policy_id'),
             get_json_object(request_params['new_settings'], '$.budget_policy_id')) AS policy_id,
    from_json(COALESCE(request_params['tags'],
                       get_json_object(request_params['new_settings'], '$.tags')),
              'map<string,string>') AS job_tags,
    event_time
  FROM system.access.audit
  WHERE workspace_id = {ws}
    AND service_name = 'jobs'
    AND action_name IN ('create', 'reset', 'update', 'submitRun')
    AND event_time >= TIMESTAMP '{ts} UTC'
    AND to_json(request_params) LIKE '%ai_runtime_task%'
    AND to_json(request_params) LIKE '%{acc}%'
),
-- Latest policy per job (partial `update` events often omit it).
reserved_jobs AS (
  SELECT * FROM audit_jobs
  WHERE policy_id IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY job_or_run_id ORDER BY event_time DESC) = 1
),
-- On-demand billing rows for those policies, the only SQL source of policy tags.
policy_rows AS (
  SELECT usage_metadata.usage_policy_id AS policy_id,
         usage_metadata.job_id          AS job_id,
         custom_tags
  FROM system.billing.usage
  WHERE usage_metadata.usage_policy_id IN (SELECT DISTINCT policy_id FROM reserved_jobs)
    AND usage_date >= date_sub(current_date(), {_days(billing_days)})
),
policy_evidence AS (
  SELECT policy_id,
         COUNT(*)               AS billing_rows,
         COUNT(DISTINCT job_id) AS billing_jobs
  FROM policy_rows
  GROUP BY 1
),
-- Keep only tag key=value pairs present on every row for the policy (policy + workspace tags).
policy_tags AS (
  SELECT t.policy_id,
         map_from_entries(collect_list(struct(t.key, t.value))) AS policy_tags
  FROM (
    SELECT policy_id, key, value, COUNT(*) AS n
    FROM policy_rows
    LATERAL VIEW explode(custom_tags) e AS key, value
    GROUP BY 1, 2, 3
  ) t
  JOIN policy_evidence ev USING (policy_id)
  WHERE t.n = ev.billing_rows
  GROUP BY t.policy_id
)
SELECT
  r.job_or_run_id,
  r.actor,
  r.policy_id,
  pt.policy_tags['{key}'] AS appspace,
  pt.policy_tags,
  r.job_tags,
  ev.billing_rows,
  ev.billing_jobs,
  r.event_time AS last_policy_event
FROM reserved_jobs r
LEFT JOIN policy_tags     pt USING (policy_id)
LEFT JOIN policy_evidence ev USING (policy_id)
ORDER BY appspace, r.job_or_run_id"""


# --- execution helpers (lazy deps) ---------------------------------------------------


def workspace_host() -> str:
    """Host ambient SDK auth resolves to. Surface this near results — a DEFAULT
    profile aimed at the wrong workspace yields confusing PERMISSION_DENIED errors."""
    from databricks.sdk.core import Config

    return Config().host


def run(sql: str, http_path: str | None = None):
    """Execute against a SQL warehouse, return a pandas DataFrame."""
    import pandas as pd
    from databricks import sql as dbsql
    from databricks.sdk.core import Config

    http_path = (
        http_path
        or os.environ.get("HUB_WAREHOUSE_HTTP_PATH")
        or os.environ.get("DATABRICKS_WAREHOUSE_HTTP_PATH")
    )
    if not http_path:
        raise RuntimeError(
            "Set HUB_WAREHOUSE_HTTP_PATH (or DATABRICKS_WAREHOUSE_HTTP_PATH) "
            "to a warehouse HTTP path like /sql/1.0/warehouses/<id>"
        )
    cfg = Config()
    with dbsql.connect(
        server_hostname=cfg.host.removeprefix("https://"),
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    ) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])


if __name__ == "__main__":
    import argparse

    builders = {
        "daily": air_usage_daily,
        "capacity": by_capacity_bucket,
        "by-principal": by_principal,
        "attribution": attribution_coverage,
        "tags": tag_inventory,
        "utilization": reservation_utilization_daily,
        "notebooks": notebook_attribution,
        "notebook-paths": notebook_paths_from_audit,
    }
    ap = argparse.ArgumentParser(description="Run an AIR billing query")
    ap.add_argument("query", choices=[*builders, "reserved-jobs"])
    ap.add_argument("--days", type=int, default=30)
    # reserved-jobs takes the reservation's workspace + billing-window start, not --days.
    ap.add_argument("--workspace-id", type=int, help="reservation workspace id (reserved-jobs)")
    ap.add_argument("--since", help="reservation window start, 'YYYY-MM-DD[ HH:MM]' UTC (reserved-jobs)")
    ap.add_argument("--accelerator", default="GPU_8xH100", help="reserved accelerator (reserved-jobs)")
    ap.add_argument("--tag-key", default="SealUsecase", help="per-use-case policy tag (reserved-jobs)")
    ap.add_argument("--sql-only", action="store_true", help="print SQL, don't execute")
    args = ap.parse_args()
    if args.query == "reserved-jobs":
        if not (args.workspace_id and args.since):
            ap.error("reserved-jobs requires --workspace-id and --since")
        stmt = reserved_job_appspace(args.workspace_id, args.since, args.accelerator, args.tag_key)
    else:
        stmt = builders[args.query](args.days)
    if args.sql_only:
        print(stmt)
    else:
        print(f"-- {workspace_host()}")
        print(run(stmt).to_string(index=False))
