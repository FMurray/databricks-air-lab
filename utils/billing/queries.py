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
  reserved  -> sku ..._PROVISIONED_CAPACITY, compute_type=RESERVED_COMPUTE,
               usage_metadata.ai_runtime_pool_id set, NO identity_metadata (the attribution gap)
  on-demand -> regional ENTERPRISE_MODEL_TRAINING_* sku, compute_type=ON_DEMAND_COMPUTE,
               usage_metadata.ai_runtime_workload_id set (CLI) / serverless_gpu key (notebooks)
⚠️ Verified 2026-07-30 on the engagement account: the 2026-07-25 20-node reserved-pool sweep
emitted ON_DEMAND_COMPUTE rows (751.3 DBUs) and ZERO reserved-SKU rows account-wide over 30d —
consistent with pool usage not (yet) stamped as reserved in this PrPr account. Don't equate
"ran on the pool" with "billed as reserved" until by_capacity_bucket shows reserved rows.

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
    reserved row can be tied to its pool. Reserved rows have no identity_metadata — if the
    'reserved' bucket is non-empty, per-user attribution inside it is expected to be NULL.
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

    Reserved pools reportedly emit aggregate records (open-q #5/#16); this measures
    the identity/tag coverage gap rather than assuming it.
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
    ⚠️ est_pct_of_reservation is computed from ALL AIR GPU DBUs, because on this account
    pool usage currently lands stamped on_demand (see module docstring, verified 2026-07-30).
    Once reserved-SKU rows appear, read reserved_dbus as the true reservation draw and
    on_demand_dbus as overflow/spill.
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


def _cluster_id(cluster_id: str) -> str:
    # Interpolated into SQL (run() has no parameter binding), so allow id characters only.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cluster_id or ""):
        raise ValueError(f"bad cluster_id: {cluster_id!r}")
    return cluster_id


def reserved_notebook_sessions(cluster_id: str, days: int = 30) -> str:
    """Notebooks attached to a reserved GPU pool, with owner + attached hours (audit-only).

    Port of the field "Attribution for GPU Pools" spec, Milestone 0 (notebooks). Reserved-pool
    billing rows carry no identity_metadata (module docstring), so this skips billing entirely:
    it pairs attachNotebook/detachNotebook events for the pool's compute id into sessions,
    clips them to the window, and takes the createNotebook user as owner.

    UNVERIFIED — not yet run against a reserved pool. Things to check when testing:
      - audit request_params.clusterId actually equals the pool's compute id;
      - duration_attached_hours is wall-clock attached time, NOT GPU-hours (no GPU-count
        multiplier; idle attached time counts);
      - an attach followed by another attach (no detach between) is dropped, and a session
        ended without a detachNotebook event runs to the window end;
      - all_attachment_events / notebook_creators scan full audit history (no event_date bound).
    """
    cid = _cluster_id(cluster_id)
    return f"""
WITH bounds AS (
  SELECT
    current_timestamp() - INTERVAL {_days(days)} DAYS AS window_start,
    current_timestamp()                    AS window_end
),

-- All attachment-state changes for the specified compute.
all_attachment_events AS (
  SELECT
    a.workspace_id,
    a.request_params['notebookId'] AS notebook_id,
    a.request_params['path']       AS notebook_path,
    a.request_params['clusterId']  AS cluster_id,
    a.action_name,
    a.event_time
  FROM system.access.audit AS a
  CROSS JOIN bounds AS b
  WHERE a.service_name = 'notebook'
    AND a.action_name IN ('attachNotebook', 'detachNotebook')
    AND a.request_params['clusterId'] = '{cid}'
    AND a.request_params['notebookId'] IS NOT NULL
    AND a.event_time <= b.window_end
),

-- Include events inside the window plus the last event before it.
-- The latter determines whether a notebook was already attached when
-- the window began.
relevant_events AS (
  SELECT e.*
  FROM all_attachment_events AS e
  CROSS JOIN bounds AS b
  WHERE e.event_time >= b.window_start

  UNION ALL

  SELECT e.*
  FROM all_attachment_events AS e
  CROSS JOIN bounds AS b
  WHERE e.event_time < b.window_start
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY e.workspace_id, e.notebook_id, e.cluster_id
    ORDER BY e.event_time DESC
  ) = 1
),

ordered_events AS (
  SELECT
    *,
    LEAD(action_name) OVER (
      PARTITION BY workspace_id, notebook_id, cluster_id
      ORDER BY event_time
    ) AS next_action,
    LEAD(event_time) OVER (
      PARTITION BY workspace_id, notebook_id, cluster_id
      ORDER BY event_time
    ) AS next_event_time
  FROM relevant_events
),

-- Construct one interval for each attachment.
attachment_sessions AS (
  SELECT
    e.workspace_id,
    e.notebook_id,
    e.notebook_path,
    GREATEST(e.event_time, b.window_start) AS session_start,
    LEAST(
      CASE
        WHEN e.next_action = 'detachNotebook'
          THEN e.next_event_time
        ELSE b.window_end
      END,
      b.window_end
    ) AS session_end
  FROM ordered_events AS e
  CROSS JOIN bounds AS b
  WHERE e.action_name = 'attachNotebook'
    AND (
      e.next_action = 'detachNotebook'
      OR e.next_action IS NULL
    )
    AND COALESCE(e.next_event_time, b.window_end) > b.window_start
),

-- Treat the user who generated createNotebook as the creator/owner.
notebook_creators AS (
  SELECT
    workspace_id,
    request_params['notebookId'] AS notebook_id,
    user_identity.email          AS owner
  FROM system.access.audit
  WHERE service_name = 'notebook'
    AND action_name = 'createNotebook'
    AND request_params['notebookId'] IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY workspace_id, request_params['notebookId']
    ORDER BY event_time
  ) = 1
)

SELECT
  s.notebook_id,
  COALESCE(
    c.owner,
    'UNKNOWN - creation event unavailable'
  ) AS owner,
  ROUND(
    SUM(
      TIMESTAMPDIFF(SECOND, s.session_start, s.session_end)
    ) / 3600.0,
    2
  ) AS duration_attached_hours
FROM attachment_sessions AS s
LEFT JOIN notebook_creators AS c
  ON  c.workspace_id = s.workspace_id
  AND c.notebook_id  = s.notebook_id
WHERE s.session_end > s.session_start
GROUP BY
  s.notebook_id,
  c.owner
ORDER BY
  duration_attached_hours DESC"""


def reserved_job_runs(cluster_id: str, days: int = 30) -> list[dict]:
    """Job runs that used a reserved GPU pool, with author + duration (Jobs API, not SQL).

    Port of the same spec's Milestone 0 (jobs): page /api/2.2/jobs/runs/list with expanded
    tasks and keep runs whose top-level or task cluster_instance.cluster_id is the pool's id.
    Auth is ambient SDK config (like run()) instead of the spec's DATABRICKS_HOST/TOKEN.
    Workspace-scoped: covers only the workspace the SDK resolves to.

    UNVERIFIED — not yet run against a reserved pool. Check that reserved-pool runs populate
    cluster_instance.cluster_id at all (CLI `air run` SUBMIT_RUN runs expose no compute block
    on the run object — docs/fleet-ops/attribute-usage.md). duration_seconds is wall-clock
    run time, not GPU-hours.
    """
    import time
    from datetime import datetime, timedelta, timezone

    from databricks.sdk import WorkspaceClient

    target = _cluster_id(cluster_id)
    api = WorkspaceClient().api_client
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=_days(days))).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    params = {"start_time_from": cutoff_ms, "expand_tasks": "true", "limit": 25}

    results = []
    while True:
        page = api.do("GET", "/api/2.2/jobs/runs/list", query=params)

        for run in page.get("runs", []):
            cluster_ids = {
                task.get("cluster_instance", {}).get("cluster_id")
                for task in run.get("tasks", [])
            }
            top_level_cluster = run.get("cluster_instance", {}).get("cluster_id")
            if top_level_cluster:
                cluster_ids.add(top_level_cluster)
            cluster_ids.discard(None)

            if target not in cluster_ids:
                continue

            # run_duration is the completed run's duration, including repairs.
            # For an active run, calculate elapsed time.
            if run.get("end_time", 0):
                duration_ms = run.get("run_duration", run["end_time"] - run["start_time"])
            else:
                duration_ms = now_ms - run["start_time"]

            results.append({
                "job_id": run.get("job_id"),
                "run_id": run["run_id"],
                "run_name": run.get("run_name"),
                "author": run.get("creator_user_name"),
                "start_time": datetime.fromtimestamp(
                    run["start_time"] / 1000, timezone.utc
                ).isoformat(),
                "duration_seconds": duration_ms / 1000,
                "result": (
                    run.get("status", {}).get("state")
                    or run.get("state", {}).get("result_state")
                ),
            })

        token = page.get("next_page_token")
        if not token:
            break
        params["page_token"] = token

    return results


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
    # Reserved-pool queries take the pool's compute id; reserved-jobs is a Jobs API call.
    reserved = {"reserved-notebooks": reserved_notebook_sessions, "reserved-jobs": reserved_job_runs}
    ap = argparse.ArgumentParser(description="Run an AIR billing query")
    ap.add_argument("query", choices=[*builders, *reserved])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cluster-id", help="reserved GPU pool compute id (reserved-* queries)")
    ap.add_argument("--sql-only", action="store_true", help="print SQL, don't execute")
    args = ap.parse_args()
    if args.query in reserved and not args.cluster_id:
        ap.error(f"{args.query} requires --cluster-id")
    if args.query == "reserved-jobs":
        import pandas as pd

        print(f"-- {workspace_host()}")
        print(pd.DataFrame(reserved_job_runs(args.cluster_id, args.days)).to_string(index=False))
        raise SystemExit
    if args.query == "reserved-notebooks":
        stmt = reserved_notebook_sessions(args.cluster_id, args.days)
    else:
        stmt = builders[args.query](args.days)
    if args.sql_only:
        print(stmt)
    else:
        print(f"-- {workspace_host()}")
        print(run(stmt).to_string(index=False))
