-- AIR / Serverless GPU usage by SKU (from go/air/enablement field guide).
-- Both product_features keys appear depending on rollout era (ai_runtime vs serverless_gpu).
SELECT
  usage_date,
  workspace_id,
  sku_name,
  usage_unit,
  product_features.ai_runtime.compute_type      AS air_compute_type,
  product_features.serverless_gpu.workload_type AS sgc_workload_type,
  SUM(usage_quantity)                           AS usage_qty
FROM system.billing.usage
WHERE (product_features.ai_runtime.compute_type IS NOT NULL
       OR product_features.serverless_gpu.workload_type IS NOT NULL)
  AND usage_date > '2025-09-01'
GROUP BY ALL
ORDER BY usage_date DESC;

-- TODO: join system.billing.list_prices for $ (the customer asked how to find the serverless GPU SKU there).
-- TODO(open-q #5): check what usage_policy_name / identity metadata lands here for reserved pools —
--   today pools reportedly emit one aggregate record (the P0 attribution gap).
