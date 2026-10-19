-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Reserved GPU attribution: DBUs per usage policy
-- MAGIC
-- MAGIC Attributes reserved serverless GPU (AIR guaranteed capacity) DBUs to the usage policy each job run was
-- MAGIC launched with. Teams map to policies through the policy's custom tags. Run the cells top to bottom in one
-- MAGIC session, in the reservation's workspace. Each block is a temp view, followed by a select over it.
-- MAGIC
-- MAGIC Part A is the load-bearing chain (jobs only): job task -> reserved accelerator -> run -> policy -> tags -> DBUs.
-- MAGIC The "Minimum viable check" right after A2's `v_task_runs` runs the whole chain in two queries.
-- MAGIC Part B is context (the reserved bill) and can wait. Notebooks are not covered yet.
-- MAGIC
-- MAGIC **No workspace filter.** System tables aren't scoped to the workspace you query them from: `system.billing.usage`
-- MAGIC is account-wide, and `system.access.audit` covers the account's workspaces in the region. If the account has
-- MAGIC other workspaces, add `workspace_id = <id>` to the `system.*` reads.
-- MAGIC
-- MAGIC How the chain works:
-- MAGIC - Reserved billing rows are per node with no job or policy, so DBUs are GPU-hold-hours x a DBU rate (param),
-- MAGIC   not a join to the bill.
-- MAGIC - The compute spec is per task, so runs are matched per (job, task_key). Non-AIR tasks in the same job aren't charged.
-- MAGIC - Reserved jobs are identified from audit events whose tasks name the accelerator. Policy history is read from all
-- MAGIC   jobs events for those jobs, so a policy-only `update` (no ai_runtime_task in the request) isn't missed.
-- MAGIC - Policy is as of run start. `create` / `reset` / `submitRun` set the policy, including setting none. An `update`
-- MAGIC   with no policy field leaves it unchanged. One-time runs (submitRun) match on run ID with no time condition,
-- MAGIC   because the audit event is logged after the run is created.
-- MAGIC - NULL policy = no explicit policy in audit. The job's effective policy was then set by an admin or inferred from
-- MAGIC   the run_as identity's accessible policies (jobs.proto, effective_usage_policy_id). It is not in any system
-- MAGIC   table; `databricks jobs get <job_id>` returns the current value.
-- MAGIC - GPU-hold time: per a code reading (not verified hands-on), a workload holds its GPUs from allocation to release,
-- MAGIC   and held GPUs, busy or idle, count against the reservation so other workloads queue (ai-scheduler
-- MAGIC   GuaranteedCapacityReconciler). Timeline periods start at task-run creation, so the period span includes
-- MAGIC   queue/admission wait. setup / execution / cleanup_duration_seconds are the run's totals repeated on every
-- MAGIC   hourly row, so they're taken once per run, not summed. `hold_basis` picks the measure; A2b shows all three.
-- MAGIC - Don't use job_run_timeline.queue_duration_seconds: the public schema documents it as queue time, but the
-- MAGIC   system-table pipeline fills it from cleanupSec, and run_duration_seconds counts cleanup twice.
-- MAGIC
-- MAGIC **Scan cost.** Audit and timeline filters are literals in the `WHERE` clause (not taken from `v_params`) so they
-- MAGIC prune files. Keep them in sync with `v_params`. Temp views aren't cached: every reference re-reads audit. On
-- MAGIC classic compute, `CACHE TABLE v_jobs_audit AS <same select>` reads it once; on serverless, write it to a
-- MAGIC scratch table instead.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Part A: jobs -> policy -> DBUs
-- MAGIC ## A0. Parameters

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_params AS
SELECT
  TIMESTAMP '2026-08-03 19:00:00 UTC' AS since,                -- runs starting at or after (UTC); reservation start
  current_timestamp()                 AS until,                -- runs starting before
  'GPU_8xH100'                        AS accelerator,          -- reserved accelerator type
  8.844                               AS dbus_per_gpu_hour,    -- DBU rate per GPU-hour
  'execution'                         AS hold_basis,           -- 'execution' | 'setup_execution_cleanup' | 'span'
  90                                  AS policy_lookback_days, -- days of policy billing rows read for tags
  160                                 AS reserved_gpus         -- nodes x GPUs per node (Part B only)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A1. AIR task specs from audit
-- MAGIC `v_jobs_audit`: the one audit read. Every jobs create / reset / update / submitRun event since the spec
-- MAGIC lookback date. `job_key` is the job ID (`key_type = 'job'`) or, for submitRun, the run ID (`key_type = 'run'`).
-- MAGIC
-- MAGIC **Spec lookback:** a job whose last create / reset / update predates the `event_date` literal has no spec and
-- MAGIC drops out of everything. Widen it until the Minimum viable check's `gpu_hours` stop growing. Audit keeps
-- MAGIC about a year.
-- MAGIC
-- MAGIC `v_job_task_specs`: one row per (event, task) for every job that has ever had an ai_runtime_task, so a later
-- MAGIC change that removes the AIR task is seen. Non-AIR tasks in those requests are kept (ai_runtime_task NULL).
-- MAGIC `accelerator_count` is total GPUs across all nodes. Events with no parseable tasks (e.g. an `update` that only
-- MAGIC changes the policy) are kept with task_key NULL.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_jobs_audit AS
SELECT
  a.event_time,
  a.action_name,
  a.user_identity.email AS actor,
  IF(a.action_name = 'submitRun', 'run', 'job') AS key_type,
  IF(a.action_name = 'submitRun',
     get_json_object(a.response.result, '$.run_id'),
     COALESCE(a.request_params['job_id'], get_json_object(a.response.result, '$.job_id'))) AS job_key,
  COALESCE(a.request_params['tasks'],
           get_json_object(a.request_params['new_settings'], '$.tasks'))               AS tasks_json,
  COALESCE(a.request_params['usage_policy_id'],
           a.request_params['budget_policy_id'],
           get_json_object(a.request_params['new_settings'], '$.usage_policy_id'),
           get_json_object(a.request_params['new_settings'], '$.budget_policy_id'))    AS policy_id,
  to_json(a.request_params) LIKE concat('%', p.accelerator, '%')                       AS mentions_accelerator
FROM system.access.audit a
CROSS JOIN v_params p
WHERE a.event_date >= DATE '2026-02-01'  -- spec lookback (literal)
  AND a.service_name = 'jobs'
  AND a.action_name IN ('create', 'reset', 'update', 'submitRun')

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_job_task_specs AS
WITH ev_all AS (
  SELECT event_time, action_name, actor, key_type, job_key, tasks_json, mentions_accelerator
  FROM v_jobs_audit
),
-- Jobs that ever had an AIR task; all their spec events are kept, so removing the AIR task is seen.
air_keys AS (
  SELECT DISTINCT key_type, job_key FROM ev_all WHERE tasks_json LIKE '%ai_runtime_task%'
),
ev AS (
  SELECT e.* FROM ev_all e JOIN air_keys k USING (key_type, job_key)
)
SELECT
  ev.event_time,
  ev.action_name,
  ev.actor,
  ev.key_type,
  ev.job_key,
  ev.mentions_accelerator,
  t.task_key,
  t.ai_runtime_task,
  COALESCE(t.ai_runtime_task LIKE concat('%', p.accelerator, '%'), false) AS is_reserved_accelerator,
  try_cast(regexp_extract(t.ai_runtime_task, '"accelerator_count"\\s*:\\s*"?(\\d+)', 1) AS INT) AS gpus
FROM ev
CROSS JOIN v_params p
LATERAL VIEW OUTER explode(
  from_json(ev.tasks_json, 'array<struct<task_key:string,ai_runtime_task:string>>')) x AS t
WHERE ev.job_key IS NOT NULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A1a. Specs by accelerator and GPU count

-- COMMAND ----------

SELECT key_type, is_reserved_accelerator, gpus,
       COUNT(DISTINCT job_key)           AS jobs,
       COUNT(DISTINCT job_key, task_key) AS tasks,
       COUNT(*)                          AS events,
       MIN(event_time)                   AS first_event,
       MAX(event_time)                   AS last_event
FROM v_job_task_specs
WHERE ai_runtime_task IS NOT NULL
GROUP BY ALL
ORDER BY is_reserved_accelerator DESC, jobs DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A1b. Events that name the accelerator but have no parsed tasks (truncated or differently shaped request_params)

-- COMMAND ----------

SELECT action_name, key_type,
       COUNT(*)                AS events,
       COUNT(DISTINCT job_key) AS jobs,
       MIN(event_time)         AS first_event,
       MAX(event_time)         AS last_event
FROM v_job_task_specs
WHERE mentions_accelerator AND task_key IS NULL
GROUP BY ALL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A2. Task runs
-- MAGIC `v_task_runs`: every task run of a job in A1, one row per task run (`run_id`). The timeline's lower bound is
-- MAGIC `since` minus 7 days (literal), so runs that started just before an hourly slice aren't cut off; the window
-- MAGIC itself is applied on `run_start` later.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_task_runs AS
SELECT
  t.job_id,
  t.job_run_id,
  t.run_id,
  t.task_key,
  MIN(t.period_start_time)          AS run_start,
  MAX(t.period_end_time)            AS run_end,
  MAX(t.result_state)               AS result_state,
  MAX(t.setup_duration_seconds)     AS setup_s,     -- run totals, not split per slice
  MAX(t.execution_duration_seconds) AS execution_s,
  MAX(t.cleanup_duration_seconds)   AS cleanup_s
FROM system.lakeflow.job_task_run_timeline t
LEFT JOIN (SELECT DISTINCT job_key FROM v_job_task_specs WHERE key_type = 'job') kj ON kj.job_key = t.job_id
LEFT JOIN (SELECT DISTINCT job_key FROM v_job_task_specs WHERE key_type = 'run') kr ON kr.job_key = t.job_run_id
WHERE t.period_start_time >= TIMESTAMP '2026-07-27 19:00:00 UTC'  -- since minus 7 days (literal)
  AND (kj.job_key IS NOT NULL OR kr.job_key IS NOT NULL)
GROUP BY 1, 2, 3, 4

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Minimum viable check
-- MAGIC The whole chain in two queries, on `v_job_task_specs` and `v_task_runs` only. Cruder than the full chain:
-- MAGIC the spec isn't matched as of run start, a `reset` that clears the policy isn't seen, and hold time is
-- MAGIC execution only.
-- MAGIC
-- MAGIC MVP-1. Reserved GPU-hours and DBUs per policy (NULL row = no policy found in audit)

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_mvp AS
WITH runs AS (
  SELECT r.run_id, r.run_start, r.execution_s, s.key_type, s.job_key, COALESCE(s.gpus, 8) AS gpus
  FROM v_task_runs r
  CROSS JOIN v_params p
  JOIN v_job_task_specs s
    ON s.task_key = r.task_key AND s.is_reserved_accelerator
   AND ((s.key_type = 'job' AND s.job_key = r.job_id) OR (s.key_type = 'run' AND s.job_key = r.job_run_id))
  WHERE r.run_start >= p.since
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.run_id ORDER BY s.event_time DESC) = 1
),
pol AS (
  SELECT event_time, key_type, job_key, policy_id FROM v_jobs_audit
)
SELECT r.*, e.policy_id
FROM runs r
LEFT JOIN pol e
  ON e.key_type = r.key_type AND e.job_key = r.job_key AND e.policy_id IS NOT NULL
 AND (e.key_type = 'run' OR e.event_time <= r.run_start)
QUALIFY ROW_NUMBER() OVER (PARTITION BY r.run_id ORDER BY e.event_time DESC) = 1

-- COMMAND ----------

SELECT policy_id,
       COUNT(DISTINCT job_key)                                               AS jobs,
       COUNT(*)                                                              AS task_runs,
       ROUND(SUM(execution_s * gpus) / 3600.0, 1)                            AS gpu_hours,
       ROUND(SUM(execution_s * gpus) / 3600.0 * MAX(p.dbus_per_gpu_hour), 1) AS dbus
FROM v_mvp
CROSS JOIN v_params p
GROUP BY policy_id
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC MVP-2. Custom tags on those policies' own billing rows (last 90 days)

-- COMMAND ----------

SELECT m.policy_id,
       COUNT(u.usage_quantity)                   AS billing_rows_90d,
       max_by(u.custom_tags, u.usage_start_time) AS sample_tags
FROM (SELECT DISTINCT policy_id FROM v_mvp WHERE policy_id IS NOT NULL) m
LEFT JOIN system.billing.usage u
  ON u.usage_metadata.usage_policy_id = m.policy_id
 AND u.usage_date >= date_sub(current_date(), 90)
GROUP BY m.policy_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A2 (cont.). Task runs matched to their spec as of run start
-- MAGIC `spec_match` says what the matched spec was. Only 'reserved accelerator' rows flow on. `gpus` falls back to
-- MAGIC 8 (one node) when the count wasn't found (`gpus_assumed`). `hold_s` is the GPU-hold seconds per `hold_basis`.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_reserved_runs AS
WITH matched AS (
  SELECT r.*,
         s.key_type, s.job_key, s.ai_runtime_task, s.is_reserved_accelerator, s.gpus AS spec_gpus,
         s.event_time AS spec_event_time
  FROM v_task_runs r
  LEFT JOIN v_job_task_specs s
    ON s.task_key = r.task_key
   AND ((s.key_type = 'job' AND s.job_key = r.job_id AND s.event_time <= r.run_start)
     OR (s.key_type = 'run' AND s.job_key = r.job_run_id))
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.run_id ORDER BY s.event_time DESC, s.action_name) = 1
)
SELECT
  m.*,
  CASE WHEN m.spec_event_time IS NULL  THEN 'no spec event before run'
       WHEN m.ai_runtime_task IS NULL  THEN 'non-AIR task'
       WHEN m.is_reserved_accelerator  THEN 'reserved accelerator'
       ELSE                                 'other accelerator' END       AS spec_match,
  COALESCE(m.spec_gpus, 8)                                               AS gpus,
  m.spec_gpus IS NULL                                                    AS gpus_assumed,
  TIMESTAMPDIFF(SECOND, m.run_start, m.run_end)                          AS span_s,
  CASE p.hold_basis
    WHEN 'execution'               THEN m.execution_s
    WHEN 'setup_execution_cleanup' THEN COALESCE(m.setup_s, 0) + COALESCE(m.execution_s, 0) + COALESCE(m.cleanup_s, 0)
    WHEN 'span'                    THEN TIMESTAMPDIFF(SECOND, m.run_start, m.run_end)
  END                                                                    AS hold_s
FROM matched m
CROSS JOIN v_params p
WHERE m.run_start >= p.since AND m.run_start < p.until

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A2a. Task runs by spec match

-- COMMAND ----------

SELECT spec_match, key_type,
       COUNT(DISTINCT job_key)        AS jobs,
       COUNT(DISTINCT task_key)       AS task_keys,
       COUNT(*)                       AS task_runs,
       SUM(IF(gpus_assumed, 1, 0))    AS runs_gpus_assumed,
       ROUND(SUM(span_s) / 3600.0, 1) AS span_hours
FROM v_reserved_runs
GROUP BY ALL
ORDER BY task_runs DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A2b. GPU-hold measures side by side, by result state

-- COMMAND ----------

SELECT result_state,
       COUNT(*)                                   AS task_runs,
       SUM(IF(execution_s IS NULL, 1, 0))         AS runs_no_durations,
       ROUND(SUM(setup_s     * gpus) / 3600.0, 1) AS setup_gpu_hours,
       ROUND(SUM(execution_s * gpus) / 3600.0, 1) AS execution_gpu_hours,
       ROUND(SUM(cleanup_s   * gpus) / 3600.0, 1) AS cleanup_gpu_hours,
       ROUND(SUM(span_s      * gpus) / 3600.0, 1) AS span_gpu_hours,
       ROUND(percentile(setup_s, 0.5), 0)         AS p50_setup_s,
       ROUND(percentile(setup_s, 0.95), 0)        AS p95_setup_s
FROM v_reserved_runs
WHERE spec_match = 'reserved accelerator'
GROUP BY ALL
ORDER BY task_runs DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A3. Policy per run
-- MAGIC Each reserved run takes the latest policy-setting jobs event at or before its start (one-time runs: their
-- MAGIC submitRun). Events come from `v_jobs_audit`, with no ai_runtime_task filter.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_job_policy_events AS
SELECT event_time, action_name, actor, key_type, job_key, policy_id
FROM v_jobs_audit

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_run_policy AS
SELECT r.*,
       e.policy_id,
       e.action_name AS policy_action,
       e.event_time  AS policy_event_time,
       e.actor       AS policy_actor
FROM v_reserved_runs r
LEFT JOIN v_job_policy_events e
  ON e.key_type = r.key_type
 AND e.job_key  = r.job_key
 AND (e.key_type = 'run' OR e.event_time <= r.run_start)
 AND (e.action_name IN ('create', 'reset', 'submitRun') OR e.policy_id IS NOT NULL)
WHERE r.spec_match = 'reserved accelerator'
QUALIFY ROW_NUMBER() OVER (PARTITION BY r.run_id ORDER BY e.event_time DESC, e.action_name) = 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A3a. Which jobs actions log a policy, for the jobs in A1 (the audit docs list create / submitRun params
-- MAGIC without a policy field)

-- COMMAND ----------

SELECT e.action_name,
       COUNT(*)                  AS events,
       COUNT(e.policy_id)        AS events_with_policy,
       COUNT(DISTINCT e.job_key) AS jobs
FROM v_job_policy_events e
JOIN (SELECT DISTINCT key_type, job_key FROM v_job_task_specs) k USING (key_type, job_key)
GROUP BY 1
ORDER BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A3b. Runs per policy, and the event that set it

-- COMMAND ----------

SELECT policy_id, policy_action,
       COUNT(DISTINCT job_key)               AS jobs,
       COUNT(*)                              AS task_runs,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1) AS gpu_hours
FROM v_run_policy
GROUP BY ALL
ORDER BY policy_id NULLS FIRST, gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A3c. Jobs with no explicit policy in audit (look up `effective_usage_policy_id` with `databricks jobs get <job_id>`)

-- COMMAND ----------

SELECT key_type, job_key,
       COUNT(*)                              AS task_runs,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1) AS gpu_hours,
       MIN(run_start)                        AS first_run,
       MAX(run_start)                        AS last_run,
       collect_set(policy_actor)             AS policy_event_actors
FROM v_run_policy
WHERE policy_id IS NULL
GROUP BY ALL
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A4. Policy tags
-- MAGIC Tags for each policy come from its own billing rows, which are non-reserved by definition, across all
-- MAGIC workspaces. custom_tags mixes policy, job and workspace tags (per a code reading, policy tags win on key
-- MAGIC conflicts with job tags). Only key=value pairs present on every row for the policy are kept, so job tags drop
-- MAGIC out unless the policy has a single job (`billing_jobs` = 1). A policy with no billing rows in the lookback
-- MAGIC gets no tags.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_policy_rows AS
SELECT
  u.usage_metadata.usage_policy_id AS policy_id,
  u.usage_metadata.job_id          AS job_id,
  u.workspace_id,
  u.custom_tags
FROM system.billing.usage u
CROSS JOIN v_params p
WHERE u.usage_metadata.usage_policy_id IN (SELECT DISTINCT policy_id FROM v_run_policy)
  AND u.usage_date >= date_sub(current_date(), p.policy_lookback_days)

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_policy_tags AS
WITH ev AS (
  SELECT policy_id,
         COUNT(*)                     AS billing_rows,
         COUNT(DISTINCT job_id)       AS billing_jobs,
         COUNT(DISTINCT workspace_id) AS billing_workspaces
  FROM v_policy_rows
  GROUP BY 1
),
kv AS (
  SELECT policy_id, key, value, COUNT(*) AS n
  FROM v_policy_rows
  LATERAL VIEW explode(custom_tags) e AS key, value
  GROUP BY 1, 2, 3
)
SELECT ev.*,
       map_from_entries(collect_list(IF(kv.n = ev.billing_rows, struct(kv.key, kv.value), NULL))) AS policy_tags
FROM ev
LEFT JOIN kv USING (policy_id)
GROUP BY ev.policy_id, ev.billing_rows, ev.billing_jobs, ev.billing_workspaces

-- COMMAND ----------

-- MAGIC %md
-- MAGIC A4a. Tags per reserved policy (NULL billing_rows = no billing rows in the lookback)

-- COMMAND ----------

SELECT p.policy_id,
       t.billing_rows, t.billing_jobs, t.billing_workspaces,
       IF(t.policy_tags IS NULL, NULL, size(t.policy_tags)) AS n_tags,
       t.policy_tags
FROM (SELECT DISTINCT policy_id FROM v_run_policy WHERE policy_id IS NOT NULL) p
LEFT JOIN v_policy_tags t USING (policy_id)
ORDER BY n_tags NULLS FIRST, p.policy_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A5. DBUs per policy per month
-- MAGIC Month of run start. `dbus` = GPU-hold-hours x `dbus_per_gpu_hour`. Runs with no policy are the NULL row.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_policy_dbus AS
WITH agg AS (
  SELECT date_trunc('MONTH', r.run_start) AS month,
         r.policy_id,
         COUNT(DISTINCT r.job_key)        AS jobs,
         COUNT(*)                         AS task_runs,
         SUM(IF(r.gpus_assumed, 1, 0))    AS runs_gpus_assumed,
         SUM(IF(r.hold_s IS NULL, 1, 0))  AS runs_no_hold,
         SUM(r.hold_s * r.gpus) / 3600.0  AS gpu_hours
  FROM v_run_policy r
  GROUP BY 1, 2
)
SELECT a.month, a.policy_id, a.jobs, a.task_runs, a.runs_gpus_assumed, a.runs_no_hold,
       ROUND(a.gpu_hours, 1)                       AS gpu_hours,
       ROUND(a.gpu_hours * p.dbus_per_gpu_hour, 2) AS dbus,
       t.policy_tags
FROM agg a
CROSS JOIN v_params p
LEFT JOIN v_policy_tags t USING (policy_id)

-- COMMAND ----------

SELECT * FROM v_policy_dbus
ORDER BY month DESC, dbus DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Part B: context
-- MAGIC ## B1. Reserved bill per day
-- MAGIC `implied_gpu_hours` = billed DBUs / rate. `reservation_gpu_hours` = reserved_gpus x 24.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW v_reserved_daily AS
SELECT
  u.usage_date,
  SUM(u.usage_quantity)                            AS reserved_dbus,
  SUM(u.usage_quantity) / MAX(p.dbus_per_gpu_hour) AS implied_gpu_hours,
  MAX(p.reserved_gpus) * 24                        AS reservation_gpu_hours
FROM system.billing.usage u
CROSS JOIN v_params p
WHERE u.product_features.ai_runtime.compute_type = 'RESERVED_COMPUTE'
  AND u.usage_start_time >= p.since
  AND u.usage_start_time < p.until
GROUP BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC B1a. Bill vs job GPU-hold per day (job hours by run-start date, so long runs land on their first day)

-- COMMAND ----------

SELECT d.usage_date,
       ROUND(d.reserved_dbus, 2)                       AS reserved_dbus,
       ROUND(d.implied_gpu_hours, 1)                   AS implied_gpu_hours,
       d.reservation_gpu_hours,
       ROUND(j.job_gpu_hours, 1)                       AS job_gpu_hours,
       ROUND(j.job_gpu_hours * p.dbus_per_gpu_hour, 2) AS job_dbus
FROM v_reserved_daily d
CROSS JOIN v_params p
LEFT JOIN (
  SELECT to_date(run_start) AS usage_date, SUM(hold_s * gpus) / 3600.0 AS job_gpu_hours
  FROM v_run_policy GROUP BY 1
) j USING (usage_date)
ORDER BY d.usage_date DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B2. 8xH100 billed per workload
-- MAGIC Non-reserved serverless GPU / AIR rows whose per-workload hourly burn rate is at least 60 DBU/hour (one
-- MAGIC 8xH100 node is about 70.8; A10 and 1xH100 are far below). Covers both billing schemas: serverless_gpu notebook
-- MAGIC rows have compute_type NULL and no pool ID. No lower date bound; rows before `since` are labeled.

-- COMMAND ----------

WITH gpu_rows AS (
  SELECT u.usage_date, u.usage_start_time, u.usage_end_time, u.usage_quantity, u.sku_name,
         u.billing_origin_product,
         COALESCE(u.product_features.ai_runtime.compute_type, 'NULL (serverless_gpu schema)') AS compute_type,
         u.product_features.serverless_gpu.workload_type                                      AS sgc_workload_type,
         COALESCE(u.usage_metadata.ai_runtime_workload_id, u.usage_metadata.job_run_id,
                  u.usage_metadata.notebook_id)                                               AS workload_key,
         u.usage_metadata.usage_policy_id
  FROM system.billing.usage u
  WHERE (u.product_features.ai_runtime IS NOT NULL OR u.product_features.serverless_gpu IS NOT NULL)
    AND COALESCE(u.product_features.ai_runtime.compute_type, '') <> 'RESERVED_COMPUTE'
),
per_workload_hour AS (
  SELECT workload_key, date_trunc('HOUR', usage_start_time) AS hour,
         SUM(usage_quantity)
           / GREATEST(TIMESTAMPDIFF(SECOND, MIN(usage_start_time), MAX(usage_end_time)) / 3600.0, 1.0 / 60) AS dbu_per_hour
  FROM gpu_rows
  GROUP BY 1, 2
)
SELECT g.usage_date,
       IF(g.usage_start_time < p.since, 'before since', 'in window') AS period,
       g.billing_origin_product, g.compute_type, g.sgc_workload_type, g.sku_name,
       COUNT(DISTINCT g.workload_key)    AS workloads,
       COUNT(DISTINCT g.usage_policy_id) AS policies,
       ROUND(SUM(g.usage_quantity), 2)   AS dbus
FROM gpu_rows g
JOIN per_workload_hour w
  ON w.workload_key = g.workload_key AND w.hour = date_trunc('HOUR', g.usage_start_time)
CROSS JOIN v_params p
WHERE w.dbu_per_hour >= 60
GROUP BY ALL
ORDER BY g.usage_date

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B3. Billed DBUs per usage policy, all products
-- MAGIC Everything billed to each policy for the window, plus the reserved node uptime (no policy) for comparison.
-- MAGIC These are the policies' non-reserved costs, separate from A5.

-- COMMAND ----------

SELECT
  CASE WHEN u.product_features.ai_runtime.compute_type = 'RESERVED_COMPUTE' THEN 'reserved node uptime (no policy)'
       WHEN u.usage_metadata.usage_policy_id IS NOT NULL                  THEN 'policy'
       ELSE                                                                     'no policy (other)' END AS bucket,
  u.usage_metadata.usage_policy_id AS policy_id,
  u.billing_origin_product,
  ROUND(SUM(u.usage_quantity), 2)  AS dbus
FROM system.billing.usage u
CROSS JOIN v_params p
WHERE u.usage_start_time >= p.since
  AND u.usage_start_time < p.until
  AND u.usage_unit = 'DBU'
GROUP BY ALL
ORDER BY bucket, policy_id, dbus DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Not covered yet
-- MAGIC - Notebooks on the reservation. Notebook attach events carry neither the policy nor the accelerator. Per a code
-- MAGIC   reading, an attached idle session holds its GPUs.
-- MAGIC - Tasks that get a GPU through `compute.hardware_accelerator` (e.g. a notebook task on GPU_8xH100) instead of
-- MAGIC   ai_runtime_task aren't in A1.
-- MAGIC - `update` requests that remove the policy via fields_to_remove are treated as "unchanged".
-- MAGIC - Policy changes made by an admin in the console don't go through the jobs API fields read here.
