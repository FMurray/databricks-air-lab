-- Requester identity for AIR OTEL telemetry: self-reported vs authoritative.
--
-- Rows carry identity two ways (see experiments/docker-otel-zerobus/app/train.py):
--   1. resource.attributes["air.requester"]  -- derived in-container from HYPERPARAMETERS_PATH
--      (and every log record also carries it as an attribute via OTEL baggage propagation)
--   2. resource.attributes["air.mlflow_run_id"] -- correlation key to platform-side truth
--
-- (1) is convenient but self-reported: any code in the container could write anything.
-- For chargeback-grade attribution, resolve identity from the platform side by joining the
-- MLflow/job run to system tables — the run creator is recorded by Databricks, not the container.

-- Self-reported view (quick):
SELECT
  time,
  service_name,
  resource.attributes:["air.requester"]::string      AS requester_selfreported,
  resource.attributes:["air.mlflow_run_id"]::string  AS mlflow_run_id,
  severity_text,
  body::string                                       AS message
FROM <catalog>.<schema>.air_otel_logs
ORDER BY time DESC;

-- Authoritative view: join telemetry -> MLflow run -> job run creator.
-- system.lakeflow.job_run_timeline records who ran the job (identity_metadata / creator),
-- independent of anything the container claims. MLflow run tags carry the job link
-- (mlflow.databricks.jobRunId / jobID tags on runs created by AIR submissions).
WITH telemetry AS (
  SELECT
    resource.attributes:["air.mlflow_run_id"]::string AS mlflow_run_id,
    COUNT(*)  AS log_rows,
    MIN(time) AS first_event,
    MAX(time) AS last_event
  FROM <catalog>.<schema>.air_otel_logs
  GROUP BY 1
)
SELECT
  t.*,
  j.run_id                    AS job_run_id,
  j.identity_metadata.run_as  AS authoritative_identity,
  j.period_start_time,
  j.result_state
FROM telemetry t
JOIN system.lakeflow.job_run_timeline j
  -- correlation path: AIR stamps the MLflow run on the job; adjust join key to your
  -- workspace's tag layout (verify with: SELECT tags FROM mlflow run via API)
  ON array_contains(map_values(j.job_parameters), t.mlflow_run_id)  -- placeholder join; see note
ORDER BY t.last_event DESC;

-- NOTE: the exact MLflow-run -> job-run correlation column varies by rollout; the launcher log's
-- SGC_DEBUG_INFO block shows the full ID set (job_id, job_task_run_id, mlflow_run_id, run_id).
-- If needed, log AIR's job run id as a resource attr too (available to the workload) and join
-- directly on j.run_id. TODO: verify which env var carries it (open-q list).
