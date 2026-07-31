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
"""

from __future__ import annotations

import os

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
    }
    ap = argparse.ArgumentParser(description="Run an AIR billing query")
    ap.add_argument("query", choices=builders)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sql-only", action="store_true", help="print SQL, don't execute")
    args = ap.parse_args()
    stmt = builders[args.query](args.days)
    if args.sql_only:
        print(stmt)
    else:
        print(f"-- {workspace_host()}")
        print(run(stmt).to_string(index=False))
