-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Reserved GPU attribution: DBUs per usage policy
-- MAGIC
-- MAGIC Attributes reserved serverless GPU (AIR guaranteed capacity) DBUs to the usage policy each job run or
-- MAGIC interactive notebook session was launched with. Teams map to policies through the policy's custom tags. Run the
-- MAGIC cells top to bottom in one session, in the reservation's workspace. Each block is a `v_*` table, followed by a
-- MAGIC select over it.
-- MAGIC
-- MAGIC Part A is the load-bearing chain for jobs: job task -> reserved accelerator -> run -> policy (A1-A3).
-- MAGIC The "Minimum viable check" right after A2's `v_task_runs` runs the jobs chain in two queries.
-- MAGIC Part N does the same for notebooks: attach session -> reserved (by elimination) -> policy (N1-N3); its
-- MAGIC sources and inferences are in its own header. A4-A6 then take tags and DBUs for jobs and notebooks together,
-- MAGIC and A7 the peak concurrent nodes per day (pool, use case, user, team).
-- MAGIC Part B is context (the reserved bill) and can wait.
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
-- MAGIC prune files. Keep them in sync with `v_params`. Each stage is a table, so each system-table read happens once per
-- MAGIC run, not once per reference.
-- MAGIC
-- MAGIC **Resuming.** To resume, run `A0 · scratch schema`, then continue from any cell whose input tables exist (`A0a`
-- MAGIC lists them). Tables don't refresh: after changing `v_params` or any stage, rerun every cell below it. Run in a
-- MAGIC fresh session: a session that ran the old temp-view version still has temp views with the same names, and they
-- MAGIC take precedence over the tables.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Part A: jobs -> policy -> DBUs
-- MAGIC ## A0. Parameters
-- MAGIC Stage tables are created in the scratch schema set by `A0 · scratch schema`. `A0a` lists the `v_*` tables
-- MAGIC already built.
-- MAGIC `until = current_timestamp()` is frozen when `v_params` is built, not at each run. For a new `until`, rebuild
-- MAGIC `v_params` and every cell below it.
-- MAGIC
-- MAGIC `dbus_per_gpu_hour` is a placeholder: fill in your DBU rate per GPU-hour before running (pricing isn't committed).
-- MAGIC GPU-hours don't depend on it; only the `dbus` columns and B2's burn-rate filter do.

-- COMMAND ----------

-- DBTITLE 1,A0 · scratch schema
-- Dedicated schema: the v_* stage tables are dropped and rebuilt by CREATE OR REPLACE.
USE CATALOG <catalog>;
USE SCHEMA <schema>;

-- COMMAND ----------

-- DBTITLE 1,A0a. Stage tables present
SHOW TABLES LIKE 'v_*'

-- COMMAND ----------

-- DBTITLE 1,A0 · v_params
CREATE OR REPLACE TABLE v_params AS
SELECT
  TIMESTAMP '2026-08-03 19:00:00 UTC' AS since,                -- runs starting at or after (UTC); reservation start
  current_timestamp()                 AS until,                -- runs starting before
  'GPU_8xH100'                        AS accelerator,          -- reserved accelerator type
  <dbus_per_gpu_hour>                 AS dbus_per_gpu_hour,    -- REQUIRED: your DBU rate per GPU-hour (not committed)
  'execution'                         AS hold_basis,           -- 'execution' | 'setup_execution_cleanup' | 'span'
  90                                  AS policy_lookback_days, -- days of policy billing rows read for tags
  160                                 AS reserved_gpus,        -- nodes x GPUs per node (Part B only)
  'idle_timeout'                      AS nb_hold_basis,        -- notebooks: 'idle_timeout' | 'activity' (N1)
  60                                  AS nb_idle_timeout_min,  -- notebooks: AIR session idle timeout, minutes
  8                                   AS nb_gpus               -- notebooks: GPUs per session (one node)

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

-- DBTITLE 1,A1 · v_jobs_audit
CREATE OR REPLACE TABLE v_jobs_audit AS
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

-- DBTITLE 1,A1 · v_job_task_specs
CREATE OR REPLACE TABLE v_job_task_specs AS
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
-- MAGIC ### A1a. Specs by accelerator and GPU count

-- COMMAND ----------

-- DBTITLE 1,A1a. Specs by accelerator and GPU count
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
-- MAGIC ### A1b. Events that name the accelerator without parsed tasks
-- MAGIC Truncated or differently shaped request_params.

-- COMMAND ----------

-- DBTITLE 1,A1b. Events that name the accelerator without parsed tasks
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

-- DBTITLE 1,A2 · v_task_runs
CREATE OR REPLACE TABLE v_task_runs AS
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
-- MAGIC ### MVP-1. Reserved GPU-hours and DBUs per policy
-- MAGIC NULL row = no policy found in audit.

-- COMMAND ----------

-- DBTITLE 1,MVP-1 · v_mvp
CREATE OR REPLACE TABLE v_mvp AS
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

-- DBTITLE 1,MVP-1. Reserved GPU-hours and DBUs per policy
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
-- MAGIC ### MVP-2. Custom tags on policy billing rows
-- MAGIC On those policies' own billing rows, last 90 days.

-- COMMAND ----------

-- DBTITLE 1,MVP-2. Custom tags on policy billing rows
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

-- DBTITLE 1,A2 · v_reserved_runs
CREATE OR REPLACE TABLE v_reserved_runs AS
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
-- MAGIC ### A2a. Task runs by spec match

-- COMMAND ----------

-- DBTITLE 1,A2a. Task runs by spec match
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
-- MAGIC ### A2b. GPU-hold measures by result state
-- MAGIC The three hold measures side by side.

-- COMMAND ----------

-- DBTITLE 1,A2b. GPU-hold measures by result state
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

-- DBTITLE 1,A3 · v_job_policy_events
CREATE OR REPLACE TABLE v_job_policy_events AS
SELECT event_time, action_name, actor, key_type, job_key, policy_id
FROM v_jobs_audit

-- COMMAND ----------

-- DBTITLE 1,A3 · v_run_policy
CREATE OR REPLACE TABLE v_run_policy AS
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
-- MAGIC ### A3a. Which jobs actions log a policy
-- MAGIC For the jobs in A1. The audit docs list create / submitRun params without a policy field.

-- COMMAND ----------

-- DBTITLE 1,A3a. Which jobs actions log a policy
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
-- MAGIC ### A3b. Runs per policy
-- MAGIC Which audit actions carry the policy is A3a.

-- COMMAND ----------

-- DBTITLE 1,A3b. Runs per policy
SELECT policy_id,
       COUNT(DISTINCT job_key)               AS jobs,
       COUNT(*)                              AS task_runs,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1) AS gpu_hours
FROM v_run_policy
GROUP BY ALL
ORDER BY policy_id NULLS FIRST, gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A3c. Jobs with no explicit policy in audit
-- MAGIC Look up `effective_usage_policy_id` with `databricks jobs get <job_id>`.

-- COMMAND ----------

-- DBTITLE 1,A3c. Jobs with no explicit policy in audit
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
-- MAGIC ### A3d. Who ran it: GPU-hours per run-as identity and policy
-- MAGIC Jobs: `run_as` from `system.lakeflow.jobs` as
-- MAGIC of run start (the identity the run executes as). One-time runs: the submitRun caller. `policy_actor` is not
-- MAGIC this: it's whoever made the audit event that set the policy (e.g. the last person to edit the job).

-- COMMAND ----------

-- DBTITLE 1,A3d. Who ran it: GPU-hours per run-as identity and policy
WITH job_run_as AS (
  SELECT r.run_id, j.run_as
  FROM v_run_policy r
  JOIN system.lakeflow.jobs j ON j.job_id = r.job_key
  WHERE r.key_type = 'job'
  -- latest job row at or before run start; else the earliest row after (jobs history starts later than audit)
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.run_id
                             ORDER BY IF(j.change_time <= r.run_start, j.change_time, NULL) DESC NULLS LAST,
                                      j.change_time) = 1
)
SELECT COALESCE(ja.run_as, IF(r.key_type = 'run', r.policy_actor, NULL)) AS run_as,
       CASE WHEN ja.run_as IS NOT NULL THEN 'job run_as'
            WHEN r.key_type = 'run'    THEN 'submitRun caller' END    AS run_as_source,
       r.policy_id,
       COUNT(DISTINCT r.job_key)                                      AS jobs,
       COUNT(*)                                                       AS task_runs,
       ROUND(SUM(r.hold_s * r.gpus) / 3600.0, 1)                      AS gpu_hours
FROM v_run_policy r
LEFT JOIN job_run_as ja USING (run_id)
GROUP BY ALL
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Part N: notebooks -> policy
-- MAGIC Interactive notebooks attached to serverless GPU. Unlike jobs, no audit event names the accelerator or the
-- MAGIC policy, so two links in this chain are inferred, and the checks after each block test them on first run.
-- MAGIC Sources, per a code reading of the notebook webapp (not verified hands-on), and the docs where noted:
-- MAGIC - `notebook.attachNotebook` request_params are only `notebookId`, `clusterId`, `path`. For an AIR attach,
-- MAGIC   `clusterId` is the AIR session ID, shaped `MMDD-HHMMSS-xxxxxxxx-cnx`.
-- MAGIC - The accelerator and usage policy are chosen in the notebook's Environment panel and stored on the notebook.
-- MAGIC   Changing them logs `modifyNotebook` with no values, and a policy auto-selected at attach isn't audited.
-- MAGIC - `notebook.submitCommand` (standard audit, per the docs: `notebookId`, `commandId`, `clusterId`,
-- MAGIC   `commandLanguage`) is logged when a cell is submitted. It has no duration, so a long cell looks like a gap.
-- MAGIC - `notebook.runCommand` (verbose audit logs only) is logged when a command ends, with `executionTime`, so the
-- MAGIC   command's start is recoverable. On a first customer workspace it didn't appear for AIR sessions at all.
-- MAGIC - The GPUs are released on idle timeout (ai-scheduler default 60 minutes since last activity) or when the same
-- MAGIC   user attaches the notebook to other compute. Neither is audited.
-- MAGIC
-- MAGIC A first customer run showed two things the original model got wrong: one session ID can carry several
-- MAGIC notebooks or users, and a session ID can stay in use for days with long idle gaps. So N1 now cuts each session
-- MAGIC into **holds**: activity with no gap longer than the idle timeout. A gap longer than that means the GPUs were
-- MAGIC released, and the next command re-acquired them. Each hold counts once, split across the notebooks in it.
-- MAGIC
-- MAGIC So: **holds** come from attach and command events on AIR session IDs (N1). **Reserved** is by elimination (N2):
-- MAGIC reserved billing rows carry no notebook ID, while on-demand GPU notebook usage bills per notebook, so a hold
-- MAGIC with no per-notebook GPU billing during it ran on the reservation, on the reserved accelerator (`nb_gpus`).
-- MAGIC **Policy** (N3) is the notebook's stored policy as seen on the notebook's own billed usage (serverless CPU or
-- MAGIC on-demand GPU rows carry `usage_policy_id` and `notebook_id`), nearest to the hold start.
-- MAGIC
-- MAGIC ## N1. AIR notebook holds from audit
-- MAGIC `v_nb_audit`: the one notebook audit read. `compute_id` is `clusterId`, or else the first request param that
-- MAGIC is an AIR session ID. `submitCommand` rows can be numerous, so this is one of the slower tables to build.

-- COMMAND ----------

-- DBTITLE 1,N1 · v_nb_audit
CREATE OR REPLACE TABLE v_nb_audit AS
WITH a AS (
  SELECT a.*,
         COALESCE(a.request_params['clusterId'],
                  try_element_at(filter(map_values(a.request_params),
                                        v -> v RLIKE '^[0-9]{4}-[0-9]{6}-[a-zA-Z0-9]{1,8}-cnx$'), 1)) AS compute_id
  FROM system.access.audit a
  WHERE a.event_date >= DATE '2026-07-27'  -- since minus 7 days (literal)
    AND a.service_name = 'notebook'
    AND a.action_name IN ('attachNotebook', 'detachNotebook', 'submitCommand', 'runCommand')
)
SELECT
  workspace_id,
  event_time,
  action_name,
  user_identity.email                                   AS actor,
  request_params['notebookId']                          AS notebook_id,
  compute_id,
  request_params['path']                                AS path,
  try_cast(request_params['executionTime'] AS DOUBLE)   AS exec_s,
  COALESCE(response.status_code, 200) < 400             AS ok,
  COALESCE(compute_id RLIKE '^[0-9]{4}-[0-9]{6}-[a-zA-Z0-9]{1,8}-cnx$', false) AS is_air_session
FROM a

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N1a. Notebook events by compute ID shape
-- MAGIC Expect AIR-session attaches if anyone used GPU notebooks, and `submitCommand` rows on AIR sessions back to
-- MAGIC `since` (check `first_event`). Without commands, holds are attach + idle timeout only. `runCommand` rows on AIR
-- MAGIC sessions mean verbose audit is on and N1 can use command durations.

-- COMMAND ----------

-- DBTITLE 1,N1a. Notebook events by compute ID shape
SELECT action_name,
       CASE WHEN is_air_session         THEN 'AIR session (...-cnx)'
            WHEN compute_id IS NULL     THEN 'no clusterId'
            ELSE                             'other compute' END AS compute_kind,
       ok,
       COUNT(*)                    AS events,
       COUNT(DISTINCT notebook_id) AS notebooks,
       COUNT(DISTINCT compute_id)  AS computes,
       MIN(event_time)             AS first_event,
       MAX(event_time)             AS last_event
FROM v_nb_audit
GROUP BY ALL
ORDER BY action_name, compute_kind, ok

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N1b. Command execution time on AIR sessions
-- MAGIC Verbose audit only. N1 takes a `runCommand`'s start as its end minus `executionTime` (seconds). All 0 means no
-- MAGIC verbose commands on AIR sessions, and N1 relies on `submitCommand` (no durations; see N2d). Check the units: if `max_exec_hours` is in the thousands, the field is
-- MAGIC milliseconds and N1 needs `/ 1000`.

-- COMMAND ----------

-- DBTITLE 1,N1b. Command execution time on AIR sessions
SELECT COUNT(*)                                     AS commands,
       COUNT(exec_s)                                AS with_exec_time,
       ROUND(percentile(exec_s, 0.5), 1)            AS p50_exec_s,
       ROUND(percentile(exec_s, 0.95), 1)           AS p95_exec_s,
       ROUND(percentile(exec_s, 0.99) / 3600.0, 2)  AS p99_exec_hours,
       ROUND(MAX(exec_s) / 3600.0, 2)               AS max_exec_hours
FROM v_nb_audit
WHERE action_name = 'runCommand' AND is_air_session

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N1c. Command and detach events on AIR notebooks, by compute ID shape
-- MAGIC For notebooks that had at least one AIR attach: their command and `detachNotebook` events, grouped by the
-- MAGIC shape of `clusterId` (digits shown as 9, letters as x). If N1b is all 0, this shows whether commands on these
-- MAGIC notebooks are logged with a different compute ID, with none, or not at all. No `runCommand` rows anywhere in
-- MAGIC N1a means verbose audit logging is off, and N1 has only attaches to go on.

-- COMMAND ----------

-- DBTITLE 1,N1c. Command and detach events on AIR notebooks, by compute ID shape
WITH air_nb AS (
  SELECT DISTINCT workspace_id, notebook_id
  FROM v_nb_audit
  WHERE action_name = 'attachNotebook' AND is_air_session
)
SELECT e.action_name,
       CASE WHEN e.compute_id IS NULL THEN '<no clusterId>'
            ELSE regexp_replace(regexp_replace(e.compute_id, '[0-9]', '9'), '[a-zA-Z]', 'x') END AS compute_shape,
       e.is_air_session,
       COUNT(*)                      AS events,
       COUNT(e.exec_s)               AS with_exec_time,
       COUNT(DISTINCT e.notebook_id) AS notebooks,
       MIN(e.event_time)             AS first_event,
       MAX(e.event_time)             AS last_event
FROM v_nb_audit e
JOIN air_nb USING (workspace_id, notebook_id)
WHERE e.action_name IN ('submitCommand', 'runCommand', 'detachNotebook')
GROUP BY ALL
ORDER BY events DESC
LIMIT 30

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N1d. Audit events that name an AIR session ID (last 3 days)
-- MAGIC Every audit event, any service, with an AIR session ID (`...-cnx`) in any request param. Finds where session
-- MAGIC activity is logged if not in `notebook.runCommand`: another action, another service, or another param name.
-- MAGIC `param_keys` lists the params those events carry. Reads 3 days of `system.access.audit` without a service
-- MAGIC filter, so it's slower than the other checks; widen the window only if it comes back empty.

-- COMMAND ----------

-- DBTITLE 1,N1d. Audit events that name an AIR session ID (last 3 days)
SELECT a.service_name,
       a.action_name,
       COUNT(*)                                                                       AS events,
       COUNT(DISTINCT a.user_identity.email)                                          AS users,
       array_sort(array_distinct(flatten(collect_list(map_keys(a.request_params))))) AS param_keys,
       MIN(a.event_time)                                                              AS first_event,
       MAX(a.event_time)                                                              AS last_event
FROM system.access.audit a
WHERE a.event_date >= date_sub(current_date(), 3)
  AND exists(map_values(a.request_params), v -> v RLIKE '[0-9]{4}-[0-9]{6}-[a-zA-Z0-9]{1,8}-cnx')
GROUP BY ALL
ORDER BY events DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N1 · v_nb_sessions
-- MAGIC `v_nb_sessions`: one row per (hold, notebook, user). A **hold** is a stretch of one AIR session with no idle gap
-- MAGIC longer than `nb_idle_timeout_min`, built from successful attaches and `submitCommand`s (points) and verbose
-- MAGIC `runCommand`s (end minus `executionTime` to end). Only sessions first attached in the window count.
-- MAGIC Without `runCommand`, a cell that runs longer than the timeout looks like an idle gap and is cut short: this is
-- MAGIC a lower bound for long cells. N2d shows how much a longer gap allowance changes the total. Detach isn't activity: it doesn't
-- MAGIC release the GPUs or reset the idle timer. Columns:
-- MAGIC - `island`: the hold's number within the session. `hold_start`: its first activity.
-- MAGIC - `share`: this notebook and user's part of the hold, by command execution time, else by command count, else
-- MAGIC   an equal split. N2 charges `nb_gpus x share`, so a hold shared by several notebooks is counted once.
-- MAGIC - `ended_by_attach_at`: the first attach of any of the hold's (notebook, user) pairs to other compute after the
-- MAGIC   hold's last activity, which ends the session.
-- MAGIC - `hold_end`: per `nb_hold_basis`. 'idle_timeout' = last activity + `nb_idle_timeout_min`; 'activity' = last
-- MAGIC   activity (lower bound). Both are capped at `ended_by_attach_at` and `until`. `idle_end` is always the
-- MAGIC   idle-timeout end; N2 uses it as the hold's widest extent.
-- MAGIC
-- MAGIC A cell still running at `until` is cut short there.

-- COMMAND ----------

-- DBTITLE 1,N1 · v_nb_sessions
CREATE OR REPLACE TABLE v_nb_sessions AS
WITH sess AS (
  SELECT workspace_id, compute_id AS session_id, MIN(event_time) AS first_attach
  FROM v_nb_audit
  WHERE action_name = 'attachNotebook' AND ok AND is_air_session AND notebook_id IS NOT NULL
  GROUP BY 1, 2
),
ev AS (
  SELECT e.workspace_id, e.compute_id AS session_id, e.notebook_id, e.actor, e.action_name,
         IF(e.action_name = 'runCommand',
            timestamp_millis(unix_millis(e.event_time) - CAST(COALESCE(e.exec_s, 0) * 1000 AS BIGINT)),
            e.event_time)                                             AS start_t,
         e.event_time                                                 AS end_t,
         IF(e.action_name = 'runCommand', COALESCE(e.exec_s, 0), 0.0) AS exec_s
  FROM v_nb_audit e
  JOIN sess s ON s.workspace_id = e.workspace_id AND s.session_id = e.compute_id
  CROSS JOIN v_params p
  WHERE e.notebook_id IS NOT NULL
    AND ((e.action_name = 'attachNotebook' AND e.ok) OR e.action_name IN ('submitCommand', 'runCommand'))
    AND s.first_attach >= p.since
    AND e.event_time < p.until
),
ordered AS (
  SELECT ev.*,
         MAX(end_t) OVER (PARTITION BY workspace_id, session_id ORDER BY start_t, end_t, notebook_id, actor
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
  FROM ev
),
numbered AS (
  -- a new hold starts when the event begins more than the idle timeout after everything before it ended
  SELECT o.*,
         SUM(IF(o.prev_end IS NULL OR o.start_t > timestampadd(MINUTE, p.nb_idle_timeout_min, o.prev_end), 1, 0))
           OVER (PARTITION BY o.workspace_id, o.session_id ORDER BY o.start_t, o.end_t, o.notebook_id, o.actor
                 ROWS UNBOUNDED PRECEDING) AS island
  FROM ordered o
  CROSS JOIN v_params p
),
isl AS (
  SELECT workspace_id, session_id, island,
         MIN(start_t)                              AS hold_start,
         MAX(end_t)                                AS last_activity,
         SUM(IF(action_name IN ('submitCommand', 'runCommand'), 1, 0)) AS hold_commands,
         SUM(exec_s)                               AS hold_exec_s,
         COUNT(DISTINCT notebook_id, actor)        AS contributors
  FROM numbered
  GROUP BY 1, 2, 3
),
contrib AS (
  SELECT workspace_id, session_id, island, notebook_id, actor,
         SUM(IF(action_name IN ('submitCommand', 'runCommand'), 1, 0)) AS commands,
         SUM(exec_s)                               AS exec_s
  FROM numbered
  GROUP BY ALL
),
ended AS (
  SELECT c.workspace_id, c.session_id, c.island, MIN(a.event_time) AS ended_by_attach_at
  FROM contrib c
  JOIN isl i USING (workspace_id, session_id, island)
  JOIN v_nb_audit a
    ON a.workspace_id = c.workspace_id
   AND a.notebook_id  = c.notebook_id
   AND a.actor        = c.actor
   AND a.action_name  = 'attachNotebook' AND a.ok
   AND a.compute_id  <> c.session_id
   AND a.event_time  >= i.last_activity
  GROUP BY 1, 2, 3
),
paths AS (
  SELECT workspace_id, notebook_id, max_by(path, event_time) AS path
  FROM v_nb_audit
  WHERE path IS NOT NULL
  GROUP BY 1, 2
)
SELECT
  c.workspace_id, c.notebook_id, pa.path, c.actor, c.session_id, c.island,
  i.hold_start, i.contributors, c.commands, i.hold_commands, c.exec_s, i.hold_exec_s,
  COALESCE(c.exec_s / NULLIF(i.hold_exec_s, 0), c.commands / NULLIF(i.hold_commands, 0),
           1.0 / i.contributors)                                                       AS share,
  i.last_activity, d.ended_by_attach_at,
  LEAST(p.until, COALESCE(d.ended_by_attach_at, p.until), i.last_activity)              AS activity_end,
  LEAST(p.until, COALESCE(d.ended_by_attach_at, p.until),
        timestampadd(MINUTE, p.nb_idle_timeout_min, i.last_activity))                   AS idle_end,
  CASE p.nb_hold_basis
    WHEN 'activity'     THEN LEAST(p.until, COALESCE(d.ended_by_attach_at, p.until), i.last_activity)
    WHEN 'idle_timeout' THEN LEAST(p.until, COALESCE(d.ended_by_attach_at, p.until),
                                   timestampadd(MINUTE, p.nb_idle_timeout_min, i.last_activity))
  END                                                                                   AS hold_end
FROM contrib c
JOIN isl i USING (workspace_id, session_id, island)
LEFT JOIN ended d USING (workspace_id, session_id, island)
LEFT JOIN paths pa USING (workspace_id, notebook_id)
CROSS JOIN v_params p

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## N2. Reserved holds, by elimination
-- MAGIC `v_nb_gpu_billing`: non-reserved serverless GPU / AIR billing rows (both schemas, as in B2). On-demand GPU
-- MAGIC notebook usage on the `serverless_gpu` schema carries `notebook_id`.
-- MAGIC
-- MAGIC `v_nb_reserved_sessions`: each N1 row with its hold's match, from the on-demand GPU rows of any of the hold's
-- MAGIC notebooks that overlap it (hold start to `idle_end`). `nb_match`, the same for every row of a hold:
-- MAGIC - 'on-demand GPU': billed per notebook, so not the reservation (another accelerator, or bursting).
-- MAGIC - 'recent': no rows, but the hold ended in the last 24 hours and billing may not have landed yet.
-- MAGIC - 'reserved': no rows. Only these flow on.
-- MAGIC
-- MAGIC `gpus` = `nb_gpus x share`, so summing `hold_s x gpus` over a hold's rows gives the hold's GPU-seconds once.
-- MAGIC
-- MAGIC Elimination breaks if on-demand GPU notebook usage bills without `notebook_id` (the older `ai_runtime` schema):
-- MAGIC those holds would read as reserved. N2a's `od_rows_no_workload_key` must be 0 for the workspace.

-- COMMAND ----------

-- DBTITLE 1,N2 · v_nb_gpu_billing
CREATE OR REPLACE TABLE v_nb_gpu_billing AS
SELECT
  u.workspace_id,
  u.usage_metadata.notebook_id                    AS notebook_id,
  u.usage_metadata.job_id                         AS job_id,
  u.usage_metadata.ai_runtime_workload_id         AS workload_id,
  u.usage_start_time,
  u.usage_end_time,
  u.usage_quantity,
  u.product_features.ai_runtime.compute_type      AS compute_type,
  u.product_features.serverless_gpu.workload_type AS sgc_workload_type
FROM system.billing.usage u
WHERE u.usage_date >= DATE '2026-07-27'  -- since minus 7 days (literal)
  AND (u.product_features.ai_runtime IS NOT NULL OR u.product_features.serverless_gpu IS NOT NULL)
  AND COALESCE(u.product_features.ai_runtime.compute_type, '') <> 'RESERVED_COMPUTE'

-- COMMAND ----------

-- DBTITLE 1,N2 · v_nb_reserved_sessions
CREATE OR REPLACE TABLE v_nb_reserved_sessions AS
WITH od AS (
  SELECT h.workspace_id, h.session_id, h.island,
         COUNT(b.usage_quantity) AS od_rows,
         SUM(b.usage_quantity)   AS od_dbus
  FROM (SELECT DISTINCT workspace_id, session_id, island, notebook_id, hold_start, idle_end FROM v_nb_sessions) h
  LEFT JOIN v_nb_gpu_billing b
    ON b.workspace_id      = h.workspace_id
   AND b.notebook_id       = h.notebook_id
   AND b.usage_end_time    > h.hold_start
   AND b.usage_start_time  < h.idle_end
  GROUP BY 1, 2, 3
)
SELECT
  s.*,
  od.od_rows,
  od.od_dbus,
  CASE WHEN od.od_rows > 0                                      THEN 'on-demand GPU'
       WHEN s.idle_end > current_timestamp() - INTERVAL 24 HOURS THEN 'recent'
       ELSE                                                          'reserved' END AS nb_match,
  p.nb_gpus * s.share                                                               AS gpus,
  TIMESTAMPDIFF(SECOND, s.hold_start, s.activity_end)                               AS activity_s,
  TIMESTAMPDIFF(SECOND, s.hold_start, s.idle_end)                                   AS idle_timeout_s,
  TIMESTAMPDIFF(SECOND, s.hold_start, s.hold_end)                                   AS hold_s
FROM v_nb_sessions s
JOIN od USING (workspace_id, session_id, island)
CROSS JOIN v_params p

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N2a. Holds by match
-- MAGIC `sessions` = distinct session IDs, `holds` = holds across them. `holds_with_commands` = 0 everywhere means
-- MAGIC verbose audit is off (see N1a). Hold hours are per hold (not per notebook). `od_rows_no_workload_key` counts the
-- MAGIC workspace's on-demand GPU rows in the window with no notebook, job or workload ID: if it's above 0, some
-- MAGIC on-demand notebook usage can't be seen, and 'reserved' may include on-demand holds.

-- COMMAND ----------

-- DBTITLE 1,N2a. Holds by match
WITH h AS (
  SELECT workspace_id, nb_match, session_id, island,
         MAX(hold_commands)                                AS commands,
         MAX(IF(ended_by_attach_at IS NOT NULL, 1, 0))     AS ended_by_attach,
         MAX(idle_timeout_s)                               AS idle_timeout_s,
         SUM(activity_s * gpus)                            AS activity_gpu_s,
         SUM(idle_timeout_s * gpus)                        AS idle_timeout_gpu_s
  FROM v_nb_reserved_sessions
  GROUP BY ALL
),
who AS (
  SELECT workspace_id, nb_match, COUNT(DISTINCT notebook_id) AS notebooks, COUNT(DISTINCT actor) AS users
  FROM v_nb_reserved_sessions
  GROUP BY 1, 2
),
k AS (
  SELECT b.workspace_id, COUNT(*) AS od_rows_no_workload_key
  FROM v_nb_gpu_billing b
  CROSS JOIN v_params p
  WHERE b.notebook_id IS NULL AND b.job_id IS NULL AND b.workload_id IS NULL
    AND b.usage_start_time >= p.since AND b.usage_start_time < p.until
  GROUP BY 1
)
SELECT h.workspace_id, h.nb_match, w.notebooks, w.users,
       COUNT(DISTINCT h.session_id)                          AS sessions,
       COUNT(*)                                              AS holds,
       SUM(IF(h.commands > 0, 1, 0))                         AS holds_with_commands,
       SUM(h.ended_by_attach)                                AS holds_ended_by_attach,
       ROUND(SUM(h.activity_gpu_s) / 3600.0, 1)              AS activity_gpu_hours,
       ROUND(SUM(h.idle_timeout_gpu_s) / 3600.0, 1)          AS idle_timeout_gpu_hours,
       ROUND(percentile(h.idle_timeout_s, 0.5) / 3600.0, 2)  AS p50_hold_hours,
       ROUND(percentile(h.idle_timeout_s, 0.95) / 3600.0, 2) AS p95_hold_hours,
       ROUND(MAX(h.idle_timeout_s) / 3600.0, 2)              AS max_hold_hours,
       COALESCE(MAX(k.od_rows_no_workload_key), 0)           AS od_rows_no_workload_key
FROM h
JOIN who w USING (workspace_id, nb_match)
LEFT JOIN k USING (workspace_id)
GROUP BY h.workspace_id, h.nb_match, w.notebooks, w.users
ORDER BY h.workspace_id, holds DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N2b. Holds shared across notebooks or users
-- MAGIC Holds with more than one (notebook, user). These are split by `share`, not double counted, so `gpu_hours` is
-- MAGIC the hold's total once. Informational: shows how common shared sessions are.

-- COMMAND ----------

-- DBTITLE 1,N2b. Holds shared across notebooks or users
SELECT session_id, island,
       COUNT(*)                              AS rows,
       COUNT(DISTINCT notebook_id)           AS notebooks,
       COUNT(DISTINCT actor)                 AS users,
       any_value(nb_match)                   AS nb_match,
       MIN(hold_start)                       AS hold_start,
       ROUND(MAX(idle_timeout_s) / 3600.0, 2) AS hold_hours,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1) AS gpu_hours
FROM v_nb_reserved_sessions
GROUP BY 1, 2
HAVING COUNT(*) > 1
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N2c. Peak GPUs held by reserved notebook holds
-- MAGIC Running total of `gpus` from hold start to `hold_end` over the 'reserved' rows, per workspace: the busiest days
-- MAGIC first. Notebooks alone can't hold more than `reserved_gpus`, so `peak_nb_gpus` above it means N1/N2 still
-- MAGIC overcount. Jobs share the same GPUs, so a peak below the cap doesn't prove the hold is right.

-- COMMAND ----------

-- DBTITLE 1,N2c. Peak GPUs held by reserved notebook holds
WITH ev AS (
  SELECT workspace_id, hold_start AS t, gpus AS d FROM v_nb_reserved_sessions WHERE nb_match = 'reserved'
  UNION ALL
  SELECT workspace_id, hold_end, -gpus FROM v_nb_reserved_sessions WHERE nb_match = 'reserved'
),
run AS (
  -- releases (negative d) sort before acquisitions at the same instant
  SELECT workspace_id, t,
         SUM(d) OVER (PARTITION BY workspace_id ORDER BY t, d ROWS UNBOUNDED PRECEDING) AS gpus_held
  FROM ev
)
SELECT r.workspace_id,
       to_date(r.t)                            AS day,
       ROUND(MAX(r.gpus_held), 1)              AS peak_nb_gpus,
       max_by(r.t, r.gpus_held)                AS peak_at,
       MAX(p.reserved_gpus)                    AS reserved_gpus,
       MAX(r.gpus_held) > MAX(p.reserved_gpus) AS over_cap
FROM run r
CROSS JOIN v_params p
GROUP BY 1, 2
ORDER BY peak_nb_gpus DESC
LIMIT 30

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N2d. Reserved notebook GPU-hours by assumed longest cell
-- MAGIC Without command durations, a gap between commands is either idle (GPUs released after the idle timeout) or a cell
-- MAGIC still running. This rebuilds the holds of the sessions N2 found 'reserved', treating any gap up to `gap_min`
-- MAGIC as one hold (a running cell), then adds the idle timeout after the last activity. `gap_min` = 60 is N1's
-- MAGIC answer; the never-split row is attach to last activity, about the old model. Per session (one node), so shared
-- MAGIC sessions count once; holds aren't cut at an attach elsewhere here.
-- MAGIC
-- MAGIC Read it with the cap: `days_over_cap` > 0 means notebooks alone held more than `reserved_gpus` at some point,
-- MAGIC so that assumption is too generous. The largest `gap_min` with no days over the cap bounds the notebook total.

-- COMMAND ----------

-- DBTITLE 1,N2d. Reserved notebook GPU-hours by assumed longest cell
WITH g AS (
  SELECT explode(array(60, 120, 240, 480, 1440, 525600)) AS gap_min
),
res AS (
  SELECT workspace_id, session_id
  FROM v_nb_reserved_sessions
  GROUP BY 1, 2
  HAVING MIN(nb_match) = 'reserved' AND MAX(nb_match) = 'reserved'
),
ev AS (
  SELECT e.workspace_id, e.compute_id AS session_id,
         IF(e.action_name = 'runCommand',
            timestamp_millis(unix_millis(e.event_time) - CAST(COALESCE(e.exec_s, 0) * 1000 AS BIGINT)),
            e.event_time)          AS start_t,
         e.event_time              AS end_t,
         p.until,
         p.nb_idle_timeout_min     AS idle_min
  FROM v_nb_audit e
  JOIN res r ON r.workspace_id = e.workspace_id AND r.session_id = e.compute_id
  CROSS JOIN v_params p
  WHERE e.notebook_id IS NOT NULL
    AND ((e.action_name = 'attachNotebook' AND e.ok) OR e.action_name IN ('submitCommand', 'runCommand'))
    AND e.event_time < p.until
),
ordered AS (
  SELECT g.gap_min, ev.*,
         MAX(ev.end_t) OVER (PARTITION BY g.gap_min, ev.workspace_id, ev.session_id ORDER BY ev.start_t, ev.end_t
                             ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
  FROM ev
  CROSS JOIN g
),
numbered AS (
  SELECT o.*,
         SUM(IF(o.prev_end IS NULL OR o.start_t > timestampadd(MINUTE, o.gap_min, o.prev_end), 1, 0))
           OVER (PARTITION BY o.gap_min, o.workspace_id, o.session_id ORDER BY o.start_t, o.end_t
                 ROWS UNBOUNDED PRECEDING) AS island
  FROM ordered o
),
holds AS (
  SELECT gap_min, workspace_id, session_id, island,
         MIN(start_t)                                                       AS hold_start,
         LEAST(MAX(until), timestampadd(MINUTE, MAX(idle_min), MAX(end_t))) AS hold_end
  FROM numbered
  GROUP BY 1, 2, 3, 4
),
deltas AS (
  SELECT gap_min, workspace_id, hold_start AS t, 1 AS d FROM holds
  UNION ALL
  SELECT gap_min, workspace_id, hold_end, -1 FROM holds
),
run AS (
  SELECT gap_min, workspace_id, t,
         SUM(d) OVER (PARTITION BY gap_min, workspace_id ORDER BY t, d ROWS UNBOUNDED PRECEDING) AS nodes_held
  FROM deltas
),
peak AS (
  SELECT r.gap_min, r.workspace_id,
         MAX(r.nodes_held) * MAX(p.nb_gpus)                                                    AS peak_gpus,
         COUNT(DISTINCT IF(r.nodes_held * p.nb_gpus > p.reserved_gpus, to_date(r.t), NULL))    AS days_over_cap
  FROM run r
  CROSS JOIN v_params p
  GROUP BY 1, 2
)
SELECT h.workspace_id,
       IF(h.gap_min = 525600, 'never split', CAST(h.gap_min AS STRING))                      AS gap_min,
       COUNT(*)                                                                               AS holds,
       ROUND(SUM(TIMESTAMPDIFF(SECOND, h.hold_start, h.hold_end)) * MAX(p.nb_gpus) / 3600.0, 1) AS gpu_hours,
       MAX(k.peak_gpus)                                                                       AS peak_gpus,
       MAX(k.days_over_cap)                                                                   AS days_over_cap,
       MAX(p.reserved_gpus)                                                                   AS reserved_gpus
FROM holds h
JOIN peak k USING (gap_min, workspace_id)
CROSS JOIN v_params p
GROUP BY h.workspace_id, h.gap_min
ORDER BY h.workspace_id, h.gap_min

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N2e. Long gaps: re-attached or not?
-- MAGIC Per the scheduler code (not verified hands-on), the idle clock runs from command *completion*, so a running
-- MAGIC cell keeps the session alive. After 60 idle minutes the session is terminated and its GPUs released, so running
-- MAGIC another cell needs a new attach. That attach can return the same session ID: the attach key is notebook, policy and
-- MAGIC accelerator, and it is kept for weeks. So for each gap between activity on a 'reserved' session, look at what
-- MAGIC came next:
-- MAGIC - `attachNotebook`: the session had ended and was re-attached. Splitting the hold here (N1) is right.
-- MAGIC - `submitCommand` with no attach before it: the session was still alive, which over a gap longer than the
-- MAGIC   timeout means a cell was running. N1 undercounts these by `extra_gpu_hours_if_running`.
-- MAGIC
-- MAGIC That the webapp logs an attach before a cell submitted to an ended session is an inference from the code, not
-- MAGIC verified. Sample a few of the long-gap sessions to confirm.

-- COMMAND ----------

-- DBTITLE 1,N2e. Long gaps: re-attached or not?
WITH res AS (
  SELECT workspace_id, session_id
  FROM v_nb_reserved_sessions
  GROUP BY 1, 2
  HAVING MIN(nb_match) = 'reserved' AND MAX(nb_match) = 'reserved'
),
ev AS (
  SELECT e.workspace_id, e.compute_id AS session_id, e.action_name, e.event_time
  FROM v_nb_audit e
  JOIN res r ON r.workspace_id = e.workspace_id AND r.session_id = e.compute_id
  CROSS JOIN v_params p
  WHERE e.notebook_id IS NOT NULL
    AND ((e.action_name = 'attachNotebook' AND e.ok) OR e.action_name IN ('submitCommand', 'runCommand'))
    AND e.event_time < p.until
),
seq AS (
  -- attach sorts before submit at the same instant
  SELECT ev.*,
         LAG(event_time) OVER (PARTITION BY workspace_id, session_id ORDER BY event_time, action_name) AS prev_t
  FROM ev
),
gaps AS (
  SELECT s.*, TIMESTAMPDIFF(SECOND, s.prev_t, s.event_time) / 60.0 AS gap_min
  FROM seq s
  WHERE s.prev_t IS NOT NULL
)
SELECT CASE WHEN g.gap_min <= p.nb_idle_timeout_min THEN '1. within timeout'
            WHEN g.gap_min <= 240                   THEN '2. timeout to 4 h'
            WHEN g.gap_min <= 1440                  THEN '3. 4 to 24 h'
            ELSE                                         '4. over 24 h' END                 AS gap_bucket,
       g.action_name                                                                        AS next_event,
       COUNT(*)                                                                             AS gaps,
       COUNT(DISTINCT g.session_id)                                                         AS sessions,
       ROUND(SUM(GREATEST(g.gap_min - p.nb_idle_timeout_min, 0)) / 60.0 * MAX(p.nb_gpus), 1) AS extra_gpu_hours_if_running
FROM gaps g
CROSS JOIN v_params p
GROUP BY ALL
ORDER BY gap_bucket, next_event

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## N3. Policy per reserved hold
-- MAGIC `v_nb_policy_rows`: every billing row of those notebooks that carries a usage policy (any product), back to
-- MAGIC `since` minus 90 days (literal). `principal` is the row's identity (AIR rows carry only created_by).
-- MAGIC
-- MAGIC `v_nb_session_policy`: each 'reserved' row (hold, notebook, user) takes one billing row's policy: the same
-- MAGIC user's rows first, then the latest row at or before the hold start, else the earliest after. `policy_source` says which. NULL policy = the notebook
-- MAGIC has no policy-bearing billing in the lookback: look the notebook up in the Environment panel instead.

-- COMMAND ----------

-- DBTITLE 1,N3 · v_nb_policy_rows
CREATE OR REPLACE TABLE v_nb_policy_rows AS
SELECT
  u.workspace_id,
  u.usage_metadata.notebook_id     AS notebook_id,
  u.usage_metadata.usage_policy_id AS policy_id,
  COALESCE(u.identity_metadata.run_as, u.identity_metadata.owned_by, u.identity_metadata.created_by) AS principal,
  u.usage_start_time,
  u.billing_origin_product
FROM system.billing.usage u
WHERE u.usage_date >= DATE '2026-05-05'  -- since minus 90 days (literal)
  AND u.usage_metadata.usage_policy_id IS NOT NULL
  AND u.usage_metadata.notebook_id IN (SELECT DISTINCT notebook_id FROM v_nb_reserved_sessions WHERE nb_match = 'reserved')

-- COMMAND ----------

-- DBTITLE 1,N3 · v_nb_session_policy
CREATE OR REPLACE TABLE v_nb_session_policy AS
SELECT
  s.*,
  r.policy_id,
  r.usage_start_time       AS policy_row_time,
  r.billing_origin_product AS policy_row_product,
  CASE WHEN r.policy_id IS NOT NULL THEN
    concat(IF(r.principal = s.actor, 'same user', 'other user'), ', ',
           IF(r.usage_start_time <= s.hold_start, 'before', 'after'), ' start') END AS policy_source
FROM v_nb_reserved_sessions s
LEFT JOIN v_nb_policy_rows r
  ON r.workspace_id = s.workspace_id
 AND r.notebook_id  = s.notebook_id
WHERE s.nb_match = 'reserved'
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY s.workspace_id, s.session_id, s.island, s.notebook_id, s.actor
  ORDER BY (r.principal = s.actor) DESC NULLS LAST,
           IF(r.usage_start_time <= s.hold_start, r.usage_start_time, NULL) DESC NULLS LAST,
           r.usage_start_time) = 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N3a. Reserved notebook GPU-hours by policy source
-- MAGIC The inferred link: 'same user, before start' is strongest. `holds` counts (hold, notebook, user) rows.

-- COMMAND ----------

-- DBTITLE 1,N3a. Reserved notebook GPU-hours by policy source
SELECT policy_source, policy_row_product,
       COUNT(DISTINCT notebook_id)           AS notebooks,
       COUNT(*)                              AS holds,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1) AS gpu_hours
FROM v_nb_session_policy
GROUP BY ALL
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N3b. Holds with a policy change around the start
-- MAGIC Holds whose notebook billed under a different policy before and after the hold start (same user). The
-- MAGIC policy may have been changed when the user switched to GPU, so N3 can pick the wrong one for these.

-- COMMAND ----------

-- DBTITLE 1,N3b. Holds with a policy change around the start
WITH x AS (
  SELECT s.session_id, s.island, s.notebook_id, s.path, s.actor, s.hold_start, s.hold_s, s.gpus, s.policy_id,
         max_by(r.policy_id, IF(r.usage_start_time <= s.hold_start, r.usage_start_time, NULL)) AS policy_before,
         min_by(r.policy_id, IF(r.usage_start_time >  s.hold_start, r.usage_start_time, NULL)) AS policy_after
  FROM v_nb_session_policy s
  JOIN v_nb_policy_rows r
    ON r.workspace_id = s.workspace_id AND r.notebook_id = s.notebook_id AND r.principal = s.actor
  GROUP BY ALL
)
SELECT notebook_id, path, actor, hold_start, policy_id AS chosen_policy, policy_before, policy_after,
       ROUND(hold_s * gpus / 3600.0, 1) AS gpu_hours
FROM x
WHERE policy_before <> policy_after
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N3c. Who ran it: reserved notebook GPU-hours per user and policy
-- MAGIC `actor` = the attaching user.

-- COMMAND ----------

-- DBTITLE 1,N3c. Who ran it: reserved notebook GPU-hours per user and policy
SELECT actor, policy_id,
       COUNT(DISTINCT notebook_id)           AS notebooks,
       COUNT(*)                              AS holds,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1) AS gpu_hours
FROM v_nb_session_policy
GROUP BY ALL
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N3d. Reserved holds with no policy: why
-- MAGIC For notebooks whose holds got a NULL policy in N3, all their billing rows of any product in the N3 lookback.
-- MAGIC - 'billed, never with a policy': the notebook ran without an explicit policy. NULL may be the right answer
-- MAGIC   (check whether the workspace uses a default policy).
-- MAGIC - 'no billing rows at all': no evidence either way, typically a notebook that only ever ran on the reservation.
-- MAGIC   N3e shows whether the user's policy could fill these.
-- MAGIC Reads `system.billing.usage`; the date literal matches N3's (`since` minus 90 days).

-- COMMAND ----------

-- DBTITLE 1,N3d. Reserved holds with no policy: why
WITH nulls AS (
  SELECT workspace_id, notebook_id, COUNT(*) AS holds, SUM(hold_s * gpus) / 3600.0 AS gpu_hours
  FROM v_nb_session_policy
  WHERE policy_id IS NULL
  GROUP BY 1, 2
),
billed AS (
  SELECT u.workspace_id, u.usage_metadata.notebook_id AS notebook_id,
         COUNT(*)                              AS billing_rows,
         collect_set(u.billing_origin_product) AS products
  FROM system.billing.usage u
  WHERE u.usage_date >= DATE '2026-05-05'  -- since minus 90 days (literal, as in N3)
    AND u.usage_metadata.notebook_id IN (SELECT notebook_id FROM nulls)
  GROUP BY 1, 2
)
SELECT IF(b.notebook_id IS NULL, 'no billing rows at all', 'billed, never with a policy') AS bucket,
       COUNT(*)                                         AS notebooks,
       SUM(n.holds)                                     AS holds,
       ROUND(SUM(n.gpu_hours), 1)                       AS gpu_hours,
       SUM(b.billing_rows)                              AS billing_rows,
       array_distinct(flatten(collect_list(b.products))) AS products
FROM nulls n
LEFT JOIN billed b USING (workspace_id, notebook_id)
GROUP BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N3e. Could the user's policy fill the NULLs?
-- MAGIC For each user with NULL-policy holds, the policies on their own billing rows (any product, N3 lookback),
-- MAGIC bucketed by how many policies they have. If most `null_gpu_hours` belong to users with exactly one policy, that
-- MAGIC policy is a reasonable fallback, labeled as such. `avg_top_share` = share of the user's policy DBUs on their
-- MAGIC most-used policy. Counts only, no names.

-- COMMAND ----------

-- DBTITLE 1,N3e. Could the user's policy fill the NULLs?
WITH nu AS (
  SELECT actor, SUM(hold_s * gpus) / 3600.0 AS gpu_hours
  FROM v_nb_session_policy
  WHERE policy_id IS NULL
  GROUP BY 1
),
up AS (
  SELECT COALESCE(u.identity_metadata.run_as, u.identity_metadata.owned_by, u.identity_metadata.created_by) AS actor,
         u.usage_metadata.usage_policy_id AS policy_id,
         SUM(u.usage_quantity)            AS dbus
  FROM system.billing.usage u
  WHERE u.usage_date >= DATE '2026-05-05'  -- since minus 90 days (literal, as in N3)
    AND u.usage_metadata.usage_policy_id IS NOT NULL
    AND COALESCE(u.identity_metadata.run_as, u.identity_metadata.owned_by, u.identity_metadata.created_by)
        IN (SELECT actor FROM nu)
  GROUP BY 1, 2
),
per_user AS (
  SELECT nu.actor,
         MAX(nu.gpu_hours)             AS gpu_hours,
         COUNT(up.policy_id)           AS policies,
         MAX(up.dbus) / SUM(up.dbus)   AS top_share
  FROM nu
  LEFT JOIN up ON up.actor = nu.actor
  GROUP BY 1
)
SELECT CASE WHEN policies = 0 THEN '0' WHEN policies = 1 THEN '1' ELSE '2+' END AS user_policies,
       COUNT(*)                   AS users,
       ROUND(SUM(gpu_hours), 1)   AS null_gpu_hours,
       ROUND(AVG(top_share), 2)   AS avg_top_share
FROM per_user
GROUP BY 1
ORDER BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## N4. Policy from each user's use cases
-- MAGIC Notebook policy selection, per the serverless usage policy docs: a user with one permitted policy gets it
-- MAGIC automatically; a user with several picks one, and "if a user doesn't select a policy, the setting defaults to
-- MAGIC whichever policy comes first alphabetically". Per the attach code, a notebook's stored policy is checked against
-- MAGIC the user's permissions at attach and re-resolved if it isn't permitted.
-- MAGIC
-- MAGIC The permitted policies come from a **use-case sheet** (input): one row per use case, `use_case_name` plus
-- MAGIC `team_member_standard_identifier_list` (ARRAY<STRING>: the members' user IDs, as they appear in `actor`). A use case is tied to
-- MAGIC its policy by a custom tag on the policy, so N4 finds each use case's policy from the tags on that policy's
-- MAGIC billed usage, then treats a user's use cases' policies as their permitted policies.
-- MAGIC
-- MAGIC Load the sheet: save it as CSV, upload it to a volume (or create a table from it in Catalog Explorer), and fill
-- MAGIC in the `N4 · v_use_case_members_raw` cell. If it's left empty, every hold is 'user not in list' and N4 keeps N3's
-- MAGIC answer, so downstream results don't change.

-- COMMAND ----------

-- DBTITLE 1,N4 · v_use_case_members_raw (input)
-- Load the use-case sheet, replacing the table. team_member_standard_identifier_list is ARRAY<STRING>. From a table
-- that already has the array column:
--   CREATE OR REPLACE TABLE v_use_case_members_raw AS
--   SELECT use_case_name, team_member_standard_identifier_list FROM <uploaded table>;
-- From a CSV in a volume, where the list arrives as text, convert it: from_json for a JSON array ("["a","b"]"),
-- split for a delimited list ("a, b"):
--   CREATE OR REPLACE TABLE v_use_case_members_raw AS
--   SELECT use_case_name,
--          split(team_member_standard_identifier_list, '[,;\\s]+') AS team_member_standard_identifier_list
--   FROM read_files('/Volumes/<catalog>/<schema>/<volume>/use_cases.csv', format => 'csv', header => true);
-- This cell only creates an empty table if none exists, so it never overwrites a loaded sheet. An existing table with
-- the old STRING column must be reloaded (DROP TABLE v_use_case_members_raw first).
CREATE TABLE IF NOT EXISTS v_use_case_members_raw (
  use_case_name                        STRING,
  team_member_standard_identifier_list ARRAY<STRING>
)

-- COMMAND ----------

-- DBTITLE 1,N4 · v_use_case_members
-- One row per (use case, member ID). Array elements are trimmed and lowercased; an element holding several IDs
-- ("a, b") is split on anything other than letters, digits and @ . _ -.
CREATE OR REPLACE TABLE v_use_case_members AS
SELECT DISTINCT trim(r.use_case_name) AS use_case_name, lower(trim(f.member_id)) AS member_id
FROM v_use_case_members_raw r
LATERAL VIEW explode(COALESCE(r.team_member_standard_identifier_list, CAST(array() AS ARRAY<STRING>))) e AS raw_id
LATERAL VIEW explode(split(COALESCE(e.raw_id, ''), '[^A-Za-z0-9@._-]+')) f AS member_id
WHERE trim(f.member_id) <> '' AND trim(r.use_case_name) <> ''

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4 · v_policy_use_cases
-- MAGIC Use case -> policy, from **AIR usage only**. A policy's tags are the key=value pairs on every one of its AIR
-- MAGIC billing rows in the lookback (`product_features.ai_runtime` or `serverless_gpu`); its usage on other products
-- MAGIC is left out, so a use case that never ran on AIR doesn't count. A use case matches a policy whose tag *value*
-- MAGIC equals `use_case_name`, or failing that a tag *key* (case-insensitive). Reserved rows carry no policy, so in
-- MAGIC practice these are on-demand AIR rows: a policy with no on-demand AIR usage in the lookback can't be matched.
-- MAGIC N4a counts those use cases. Reads `system.billing.usage` (the date literal matches N3's).

-- COMMAND ----------

-- DBTITLE 1,N4 · v_policy_use_cases
CREATE OR REPLACE TABLE v_policy_use_cases AS
WITH pr AS (
  SELECT u.usage_metadata.usage_policy_id AS policy_id, u.custom_tags
  FROM system.billing.usage u
  WHERE u.usage_date >= DATE '2026-05-05'  -- since minus 90 days (literal, as in N3)
    AND u.usage_metadata.usage_policy_id IS NOT NULL
    -- AIR usage only: a policy's other products (SQL, jobs, ...) don't make its use case an AIR use case
    AND (u.product_features.ai_runtime IS NOT NULL OR u.product_features.serverless_gpu IS NOT NULL)
),
n AS (
  SELECT policy_id, COUNT(*) AS billing_rows FROM pr GROUP BY 1
),
kv AS (
  SELECT policy_id, key, value, COUNT(*) AS c
  FROM pr
  LATERAL VIEW explode(custom_tags) e AS key, value
  GROUP BY 1, 2, 3
),
tags AS (
  SELECT kv.policy_id, kv.key, kv.value
  FROM kv
  JOIN n USING (policy_id)
  WHERE kv.c = n.billing_rows
),
uc AS (
  SELECT DISTINCT trim(use_case_name) AS use_case_name
  FROM v_use_case_members_raw
  WHERE trim(use_case_name) <> ''
),
m AS (
  SELECT uc.use_case_name, t.policy_id, t.key AS tag_key, t.value AS tag_value,
         IF(lower(trim(t.value)) = lower(uc.use_case_name), 'tag value', 'tag key') AS matched_on
  FROM uc
  JOIN tags t
    ON lower(trim(t.value)) = lower(uc.use_case_name)
    OR lower(trim(t.key))   = lower(uc.use_case_name)
)
-- prefer value matches: drop a use case's key matches when it has any value match
SELECT m.*
FROM m
QUALIFY matched_on = 'tag value' OR MAX(IF(matched_on = 'tag value', 1, 0)) OVER (PARTITION BY use_case_name) = 0

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4 · v_user_policies
-- MAGIC User -> permitted policy, from the sheet: each member ID (`v_use_case_members`) is matched to a
-- MAGIC reserved-notebook `actor` that equals it or whose part before `@` equals it (case-insensitive). `use_case_unmapped` marks a membership whose use case matched no policy. N4 ignores
-- MAGIC those memberships: the use cases that count are the ones a policy's tags carry, not every name in the sheet.
-- MAGIC `policy_name` is NULL (the sheet has use cases, not policy names), so the alphabetical default isn't computed.

-- COMMAND ----------

-- DBTITLE 1,N4 · v_user_policies
CREATE OR REPLACE TABLE v_user_policies AS
WITH members AS (
  SELECT use_case_name, member_id FROM v_use_case_members
),
actors AS (
  SELECT DISTINCT actor FROM v_nb_session_policy WHERE actor IS NOT NULL
),
matched AS (
  SELECT DISTINCT a.actor, m.use_case_name
  FROM members m
  JOIN actors a
    ON m.member_id = lower(trim(a.actor))
    OR m.member_id = lower(split_part(trim(a.actor), '@', 1))
)
SELECT DISTINCT
  mt.actor                      AS user_email,
  p.policy_id,
  CAST(NULL AS STRING)          AS policy_name,
  mt.use_case_name,
  p.policy_id IS NULL           AS use_case_unmapped
FROM matched mt
LEFT JOIN (SELECT DISTINCT use_case_name, policy_id FROM v_policy_use_cases) p
  ON lower(p.use_case_name) = lower(mt.use_case_name)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4a. Use-case sheet coverage
-- MAGIC One row per measure. The ones that decide how far N4 gets: `use_cases_unmapped` (no policy carries that use case
-- MAGIC as a tag), `use_cases_on_tag_key` (matched a tag key rather than a value: check these are right), and the
-- MAGIC reserved-notebook GPU-hours of users not found in the sheet (usually an ID-format mismatch).

-- COMMAND ----------

-- DBTITLE 1,N4a. Use-case sheet coverage
WITH uc AS (
  SELECT DISTINCT lower(trim(use_case_name)) AS uc FROM v_use_case_members_raw WHERE trim(use_case_name) <> ''
),
mapped AS (
  SELECT lower(use_case_name) AS uc, COUNT(DISTINCT policy_id) AS policies, MAX(IF(matched_on = 'tag value', 1, 0)) AS by_value
  FROM v_policy_use_cases
  GROUP BY 1
),
act AS (
  SELECT actor,
         SUM(hold_s * gpus) / 3600.0                              AS gpu_hours,
         SUM(IF(policy_id IS NULL, hold_s * gpus, 0)) / 3600.0    AS null_gpu_hours
  FROM v_nb_session_policy
  GROUP BY 1
),
listed AS (
  SELECT user_email, MAX(IF(use_case_unmapped, 1, 0)) AS incomplete, COUNT(DISTINCT policy_id) AS policies
  FROM v_user_policies
  GROUP BY 1
)
SELECT '1. use cases in sheet' AS measure, CAST(COUNT(*) AS DOUBLE) AS value FROM uc
UNION ALL SELECT '2. use cases mapped to one policy',      CAST(COUNT(*) AS DOUBLE) FROM mapped WHERE policies = 1
UNION ALL SELECT '3. use cases mapped to several policies', CAST(COUNT(*) AS DOUBLE) FROM mapped WHERE policies > 1
UNION ALL SELECT '4. use cases unmapped',                   CAST(COUNT(*) AS DOUBLE) FROM uc LEFT ANTI JOIN mapped USING (uc)
UNION ALL SELECT '5. use cases on tag key',                 CAST(COUNT(*) AS DOUBLE) FROM mapped WHERE by_value = 0
UNION ALL SELECT '6. reserved-notebook users',              CAST(COUNT(*) AS DOUBLE) FROM act
UNION ALL SELECT '7. ... found in sheet',                   CAST(COUNT(*) AS DOUBLE) FROM act JOIN listed ON listed.user_email = act.actor
UNION ALL SELECT '8. ... found, permitted set complete',    CAST(COUNT(*) AS DOUBLE) FROM act JOIN listed ON listed.user_email = act.actor WHERE incomplete = 0
UNION ALL SELECT '9. ... found, exactly one policy',        CAST(COUNT(*) AS DOUBLE) FROM act JOIN listed ON listed.user_email = act.actor WHERE incomplete = 0 AND policies = 1
UNION ALL SELECT '10. GPU-hours, users not in sheet',       ROUND(SUM(gpu_hours), 1) FROM act LEFT ANTI JOIN listed ON listed.user_email = act.actor
UNION ALL SELECT '11. NULL-policy GPU-hours, users in sheet', ROUND(SUM(null_gpu_hours), 1) FROM act JOIN listed ON listed.user_email = act.actor

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4c. Why use cases don't map to a policy
-- MAGIC Each sheet use case by its best match against policy tags on billed usage (any row in the N4 lookback):
-- MAGIC - 1: exact, on every billing row of the policy (what `v_policy_use_cases` uses today).
-- MAGIC - 2: exact, but only on some rows: the tag was probably added or changed during the lookback.
-- MAGIC - 3: equal only after removing case, spaces and punctuation (e.g. `Fraud Detection` vs `fraud-detection`).
-- MAGIC - 4: matches a tag key, not a value.
-- MAGIC - 5: no match anywhere: the tag isn't on any billed usage, e.g. the policy didn't bill in the lookback.
-- MAGIC `tag_keys` lists the tag keys the matches were found on (keys only). If one key dominates, it's the use-case
-- MAGIC key, and N4 can match on that key only. Reads `system.billing.usage` (date literal as in N3).

-- COMMAND ----------

-- DBTITLE 1,N4c. Why use cases don't map to a policy
WITH pr AS (
  SELECT u.usage_metadata.usage_policy_id AS policy_id, u.custom_tags
  FROM system.billing.usage u
  WHERE u.usage_date >= DATE '2026-05-05'  -- since minus 90 days (literal, as in N3)
    AND u.usage_metadata.usage_policy_id IS NOT NULL
    -- AIR usage only: a policy's other products (SQL, jobs, ...) don't make its use case an AIR use case
    AND (u.product_features.ai_runtime IS NOT NULL OR u.product_features.serverless_gpu IS NOT NULL)
),
n AS (
  SELECT policy_id, COUNT(*) AS billing_rows FROM pr GROUP BY 1
),
t AS (
  SELECT kv.policy_id, kv.key, kv.value, kv.c / n.billing_rows AS share,
         regexp_replace(lower(kv.value), '[^a-z0-9]', '') AS nv,
         regexp_replace(lower(kv.key), '[^a-z0-9]', '')   AS nk
  FROM (
    SELECT policy_id, key, value, COUNT(*) AS c
    FROM pr
    LATERAL VIEW explode(custom_tags) e AS key, value
    GROUP BY 1, 2, 3
  ) kv
  JOIN n USING (policy_id)
),
uc AS (
  SELECT DISTINCT trim(use_case_name) AS use_case_name,
         regexp_replace(lower(trim(use_case_name)), '[^a-z0-9]', '') AS nu
  FROM v_use_case_members_raw
  WHERE trim(use_case_name) <> ''
),
m AS (
  SELECT uc.use_case_name, t.policy_id, t.key,
         CASE WHEN lower(trim(t.value)) = lower(uc.use_case_name) AND t.share = 1 THEN 1
              WHEN lower(trim(t.value)) = lower(uc.use_case_name)                 THEN 2
              WHEN t.nv = uc.nu                                                   THEN 3
              ELSE                                                                     4 END AS lvl
  FROM uc
  JOIN t ON t.nv = uc.nu OR t.nk = uc.nu
),
best AS (
  SELECT use_case_name, MIN(lvl) AS lvl FROM m GROUP BY 1
),
per AS (
  SELECT uc.use_case_name,
         COALESCE(b.lvl, 5)                                     AS lvl,
         COUNT(DISTINCT IF(m.lvl = b.lvl, m.policy_id, NULL))   AS policies,
         collect_set(IF(m.lvl = b.lvl, m.key, NULL))            AS keys
  FROM uc
  LEFT JOIN best b USING (use_case_name)
  LEFT JOIN m ON m.use_case_name = uc.use_case_name
  GROUP BY 1, 2
)
SELECT CASE lvl WHEN 1 THEN '1. exact, every billing row (N4 today)'
                WHEN 2 THEN '2. exact, some rows only'
                WHEN 3 THEN '3. same ignoring case/punctuation'
                WHEN 4 THEN '4. tag key only'
                ELSE        '5. no match' END                  AS best_match,
       COUNT(*)                                                AS use_cases,
       SUM(IF(policies = 1, 1, 0))                             AS one_policy,
       SUM(IF(policies > 1, 1, 0))                             AS several_policies,
       array_sort(array_distinct(flatten(collect_list(keys)))) AS tag_keys
FROM per
GROUP BY lvl
ORDER BY lvl

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4d. Users found in the sheet, by number of use cases
-- MAGIC Even with every use case mapped, a user in several use cases has several permitted policies, so 'only permitted
-- MAGIC policy' can't decide their holds. `null_gpu_hours` in the '1' row is the most that rule could resolve.

-- COMMAND ----------

-- DBTITLE 1,N4d. Users found in the sheet, by number of use cases
WITH m AS (
  SELECT user_email,
         COUNT(DISTINCT use_case_name)          AS use_cases,
         COUNT(DISTINCT policy_id)              AS policies,
         MAX(IF(use_case_unmapped, 1, 0))       AS incomplete
  FROM v_user_policies
  GROUP BY 1
),
a AS (
  SELECT actor,
         SUM(hold_s * gpus) / 3600.0                            AS gpu_hours,
         SUM(IF(policy_id IS NULL, hold_s * gpus, 0)) / 3600.0  AS null_gpu_hours
  FROM v_nb_session_policy
  GROUP BY 1
)
SELECT CASE WHEN m.use_cases = 1 THEN '1'
            WHEN m.use_cases = 2 THEN '2'
            WHEN m.use_cases <= 5 THEN '3-5'
            ELSE '6+' END                  AS use_cases_per_user,
       COUNT(*)                            AS users,
       SUM(m.incomplete)                   AS users_with_unmapped_use_case,
       ROUND(SUM(a.gpu_hours), 1)          AS gpu_hours,
       ROUND(SUM(a.null_gpu_hours), 1)     AS null_gpu_hours
FROM a
JOIN m ON m.user_email = a.actor
GROUP BY 1
ORDER BY 1

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4 · v_nb_session_policy_resolved
-- MAGIC N3's rows with `resolution`, the rule that decided `resolved_policy_id`:
-- MAGIC - 'billing evidence, permitted': N3's policy, which one of the user's use cases carries. Strongest.
-- MAGIC - 'billing evidence, untagged use cases': N3's policy isn't among the user's, but the user is also in use cases
-- MAGIC   that matched no policy, so it may belong to one of those. N3's answer kept.
-- MAGIC - 'only permitted policy': the user's one policy, from their use cases that a policy's tags carry. Use cases
-- MAGIC   that matched no policy are ignored (`permitted_incomplete` marks those users).
-- MAGIC - 'no tagged use case': none of the user's use cases matched a policy, and N3 found none. Unresolved (NULL).
-- MAGIC - 'several permitted, none evidenced': unresolved (NULL). `default_guess_policy_id` is the alphabetically first
-- MAGIC   policy by name when names are known (not from the use-case sheet).
-- MAGIC - 'billing evidence, not permitted': N3's policy isn't among the user's, and they have several. Unresolved (NULL).
-- MAGIC - 'user not in list': N3's answer kept.
-- MAGIC
-- MAGIC Ignoring the unmatched use cases has one blind spot: reserved usage carries no policy on its billing rows, so a
-- MAGIC policy used only on the reservation never shows its tags, and its use case counts as unmatched. N4e lists the
-- MAGIC users this could affect.
-- MAGIC
-- MAGIC Membership is as of the sheet, not history: a change during the window can make a rule wrong for the earlier
-- MAGIC period. A4 and A5 use `resolved_policy_id`.

-- COMMAND ----------

-- DBTITLE 1,N4 · v_nb_session_policy_resolved
CREATE OR REPLACE TABLE v_nb_session_policy_resolved AS
WITH up AS (
  SELECT DISTINCT lower(trim(user_email)) AS actor_key, NULLIF(trim(policy_id), '') AS policy_id, policy_name,
         COALESCE(use_case_unmapped, false) AS use_case_unmapped
  FROM v_user_policies
  WHERE user_email IS NOT NULL
),
per_user AS (
  SELECT actor_key,
         COUNT(DISTINCT policy_id)                                         AS permitted_policies,
         MIN(policy_id)                                                    AS any_policy,
         min_by(policy_id, lower(policy_name))                             AS first_by_name,
         MAX(use_case_unmapped)                                            AS incomplete
  FROM up
  GROUP BY 1
),
r AS (
  SELECT s.*,
         u.permitted_policies,
         u.any_policy,
         u.first_by_name,
         COALESCE(u.incomplete, false)                 AS permitted_incomplete,
         u.actor_key IS NOT NULL                       AS in_list,
         ok.policy_id IS NOT NULL                      AS n3_policy_permitted
  FROM v_nb_session_policy s
  LEFT JOIN per_user u ON u.actor_key = lower(trim(s.actor))
  LEFT JOIN (SELECT DISTINCT actor_key, policy_id FROM up WHERE policy_id IS NOT NULL) ok
    ON ok.actor_key = lower(trim(s.actor)) AND ok.policy_id = s.policy_id
)
SELECT r.* EXCEPT (any_policy, first_by_name, in_list),
       CASE WHEN NOT r.in_list                                THEN 'user not in list'
            WHEN r.n3_policy_permitted                         THEN 'billing evidence, permitted'
            WHEN r.permitted_incomplete AND r.policy_id IS NOT NULL THEN 'billing evidence, untagged use cases'
            WHEN r.permitted_policies = 1                      THEN 'only permitted policy'
            WHEN r.permitted_policies = 0                      THEN 'no tagged use case'
            WHEN r.policy_id IS NOT NULL                       THEN 'billing evidence, not permitted'
            ELSE                                                    'several permitted, none evidenced' END AS resolution,
       CASE WHEN NOT r.in_list                                THEN r.policy_id
            WHEN r.n3_policy_permitted                         THEN r.policy_id
            WHEN r.permitted_incomplete AND r.policy_id IS NOT NULL THEN r.policy_id
            WHEN r.permitted_policies = 1                      THEN r.any_policy
            ELSE NULL END                                                                                  AS resolved_policy_id,
       IF(r.in_list AND r.permitted_policies > 1 AND r.policy_id IS NULL,
          r.first_by_name, NULL)                                                                           AS default_guess_policy_id
FROM r

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4b. Reserved notebook GPU-hours by resolution
-- MAGIC How much each rule decides. `n3_was_null_gpu_hours` = holds N3 left without a policy; `n3_overridden_gpu_hours`
-- MAGIC = N3 had a policy the user isn't permitted to use; `guess_gpu_hours` = what the alphabetical default would add.

-- COMMAND ----------

-- DBTITLE 1,N4b. Reserved notebook GPU-hours by resolution
SELECT resolution,
       COUNT(DISTINCT actor)                                                   AS users,
       COUNT(DISTINCT notebook_id)                                             AS notebooks,
       COUNT(*)                                                                AS holds,
       ROUND(SUM(hold_s * gpus) / 3600.0, 1)                                   AS gpu_hours,
       ROUND(SUM(IF(policy_id IS NULL, hold_s * gpus, 0)) / 3600.0, 1)         AS n3_was_null_gpu_hours,
       ROUND(SUM(IF(policy_id IS NOT NULL AND NOT n3_policy_permitted
                    AND resolved_policy_id IS DISTINCT FROM policy_id, hold_s * gpus, 0)) / 3600.0, 1) AS n3_overridden_gpu_hours,
       ROUND(SUM(IF(default_guess_policy_id IS NOT NULL, hold_s * gpus, 0)) / 3600.0, 1) AS guess_gpu_hours
FROM v_nb_session_policy_resolved
GROUP BY 1
ORDER BY gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N4e. Users with use cases that match no policy
-- MAGIC One row per user N4 ignored a use case for, biggest first: their use cases that map to a policy, and the ones
-- MAGIC that don't (`unmapped_use_cases`). `null_gpu_hours` had no policy from N3; `resolved_gpu_hours` has one after
-- MAGIC N4. If an ignored use case is really the one the user ran under, that row's answer is wrong.

-- COMMAND ----------

-- DBTITLE 1,N4e. Users with use cases that match no policy
WITH h AS (
  SELECT actor,
         COUNT(*)                                                         AS holds,
         ROUND(SUM(hold_s * gpus) / 3600.0, 1)                            AS gpu_hours,
         ROUND(SUM(IF(policy_id IS NULL, hold_s * gpus, 0)) / 3600.0, 1)  AS null_gpu_hours,
         ROUND(SUM(IF(resolved_policy_id IS NOT NULL, hold_s * gpus, 0)) / 3600.0, 1) AS resolved_gpu_hours
  FROM v_nb_session_policy_resolved
  WHERE permitted_incomplete
  GROUP BY 1
),
u AS (
  SELECT user_email,
         COUNT(DISTINCT policy_id)                                                 AS mapped_policies,
         sort_array(collect_set(IF(NOT use_case_unmapped, use_case_name, NULL)))   AS mapped_use_cases,
         sort_array(collect_set(IF(use_case_unmapped, use_case_name, NULL)))       AS unmapped_use_cases
  FROM v_user_policies
  GROUP BY 1
)
SELECT h.actor, h.holds, h.gpu_hours, h.null_gpu_hours, h.resolved_gpu_hours,
       u.mapped_policies, u.mapped_use_cases, u.unmapped_use_cases
FROM h
LEFT JOIN u ON lower(trim(u.user_email)) = lower(trim(h.actor))
ORDER BY h.gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## N5. Teams
-- MAGIC When the policy (or use case) can't be pinned down, the team is the next best unit. A use case's team is the
-- MAGIC prefix its `use_case_name` starts with (case-insensitive, longest prefix first), from `v_team_prefixes`; a use
-- MAGIC case matching no prefix is team 'other'. A user's teams are their use cases' teams. A4 then gives every job run
-- MAGIC and notebook hold a team: from its policy's use cases if they're all one team, else from its user if all of
-- MAGIC that user's use cases are one team.
-- MAGIC
-- MAGIC `v_team_prefixes` is an **input**, kept out of the committed notebook: insert the team prefixes once, e.g.
-- MAGIC `INSERT INTO v_team_prefixes VALUES ('<prefix-a>'), ('<prefix-b>')`. Left empty, every use case is 'other'.

-- COMMAND ----------

-- DBTITLE 1,N5 · v_team_prefixes (input)
-- Insert the team prefixes once (they're customer-specific, so they aren't committed here):
--   INSERT INTO v_team_prefixes VALUES ('<prefix-a>'), ('<prefix-b>');
-- This cell only creates an empty table if none exists, so it never overwrites the list.
CREATE TABLE IF NOT EXISTS v_team_prefixes (prefix STRING)

-- COMMAND ----------

-- DBTITLE 1,N5 · v_use_case_teams
CREATE OR REPLACE TABLE v_use_case_teams AS
SELECT uc.use_case_name,
       COALESCE(max_by(lower(trim(p.prefix)), length(trim(p.prefix))), 'other') AS team
FROM (
  SELECT DISTINCT trim(use_case_name) AS use_case_name
  FROM v_use_case_members_raw
  WHERE trim(use_case_name) <> ''
) uc
LEFT JOIN v_team_prefixes p
  ON trim(p.prefix) <> ''
 AND startswith(lower(uc.use_case_name), lower(trim(p.prefix)))
GROUP BY 1

-- COMMAND ----------

-- DBTITLE 1,N5 · v_policy_teams
-- Policy -> team(s), through the use cases N4 matched to the policy.
CREATE OR REPLACE TABLE v_policy_teams AS
SELECT DISTINCT p.policy_id, t.team
FROM v_policy_use_cases p
JOIN v_use_case_teams t USING (use_case_name)

-- COMMAND ----------

-- DBTITLE 1,N5 · v_run_as
-- Each reserved job run's run-as identity, as in A3d: the job's run_as as of run start (system.lakeflow.jobs), or the
-- submitRun caller for one-time runs.
CREATE OR REPLACE TABLE v_run_as AS
WITH job_run_as AS (
  SELECT r.run_id, j.run_as
  FROM v_run_policy r
  JOIN system.lakeflow.jobs j ON j.job_id = r.job_key
  WHERE r.key_type = 'job'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.run_id
                             ORDER BY IF(j.change_time <= r.run_start, j.change_time, NULL) DESC NULLS LAST,
                                      j.change_time) = 1
)
SELECT r.run_id,
       COALESCE(ja.run_as, IF(r.key_type = 'run', r.policy_actor, NULL)) AS run_as,
       CASE WHEN ja.run_as IS NOT NULL THEN 'job run_as'
            WHEN r.key_type = 'run'    THEN 'submitRun caller' END    AS run_as_source
FROM v_run_policy r
LEFT JOIN job_run_as ja USING (run_id)

-- COMMAND ----------

-- DBTITLE 1,N5 · v_user_teams
-- Identity -> team(s): notebook users and job run-as identities matched to sheet members, whole or by the part
-- before @ (as in v_user_policies).
CREATE OR REPLACE TABLE v_user_teams AS
WITH ids AS (
  SELECT DISTINCT actor AS identity FROM v_nb_session_policy WHERE actor IS NOT NULL
  UNION
  SELECT DISTINCT run_as FROM v_run_as WHERE run_as IS NOT NULL
)
SELECT DISTINCT lower(trim(i.identity)) AS identity_key, i.identity, m.use_case_name, t.team
FROM v_use_case_members m
JOIN ids i
  ON m.member_id = lower(trim(i.identity))
  OR m.member_id = lower(split_part(trim(i.identity), '@', 1))
JOIN v_use_case_teams t USING (use_case_name)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### N5a. Teams per use case and per user
-- MAGIC Use cases and users per team, and users whose use cases span several teams (A4 can't give those a team
-- MAGIC from the user alone). Counts only.

-- COMMAND ----------

-- DBTITLE 1,N5a. Teams per use case and per user
WITH uc AS (
  SELECT t.team, t.use_case_name, MAX(IF(p.policy_id IS NOT NULL, 1, 0)) AS has_policy
  FROM v_use_case_teams t
  LEFT JOIN v_policy_use_cases p USING (use_case_name)
  GROUP BY 1, 2
),
ut AS (
  SELECT identity_key, COUNT(DISTINCT team) AS teams, MIN(team) AS team FROM v_user_teams GROUP BY 1
)
SELECT team, use_cases, use_cases_with_policy, users_only_this_team
FROM (SELECT team, COUNT(*) AS use_cases, SUM(has_policy) AS use_cases_with_policy FROM uc GROUP BY 1) a
FULL JOIN (SELECT team, COUNT(*) AS users_only_this_team FROM ut WHERE teams = 1 GROUP BY 1) b USING (team)
UNION ALL
SELECT '(users in several teams)', NULL, NULL, COUNT(*) FROM ut WHERE teams > 1
ORDER BY team

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Part A (cont.): tags and DBUs, jobs and notebooks
-- MAGIC ## A4 · v_attributed
-- MAGIC `v_attributed`: one row per charged unit: a reserved task run (A3) or a reserved notebook hold row (N4's
-- MAGIC `resolved_policy_id`; unit_id = session#hold). A4 and A5 read from it.
-- MAGIC
-- MAGIC `identity` is the job's run-as (N5 · v_run_as) or the notebook's attaching user. `team`, in order:
-- MAGIC - 'policy use case': the unit's policy carries use cases of exactly one team (N5 · v_policy_teams).
-- MAGIC - 'only team of user': the identity's use cases in the sheet are all one team (N5 · v_user_teams).
-- MAGIC - 'user in several teams' / 'user not in sheet' / 'no identity': no team (NULL).

-- COMMAND ----------

-- DBTITLE 1,A4 · v_attributed
CREATE OR REPLACE TABLE v_attributed AS
WITH units AS (
  SELECT 'job' AS workload, r.key_type, r.job_key AS workload_key, CAST(r.run_id AS STRING) AS unit_id,
         r.run_start AS start_time, r.policy_id, r.gpus, r.gpus_assumed, r.hold_s, ra.run_as AS identity
  FROM v_run_policy r
  LEFT JOIN v_run_as ra USING (run_id)
  UNION ALL
  SELECT 'notebook', 'notebook', notebook_id, concat(session_id, '#', island),
         hold_start, resolved_policy_id, gpus, true, hold_s, actor
  FROM v_nb_session_policy_resolved
),
pt AS (
  SELECT policy_id, MIN(team) AS team, COUNT(DISTINCT team) AS teams FROM v_policy_teams GROUP BY 1
),
ut AS (
  SELECT identity_key, MIN(team) AS team, COUNT(DISTINCT team) AS teams FROM v_user_teams GROUP BY 1
)
SELECT u.*,
       CASE WHEN pt.teams = 1 THEN pt.team
            WHEN ut.teams = 1 THEN ut.team END                AS team,
       CASE WHEN pt.teams = 1          THEN 'policy use case'
            WHEN ut.teams = 1          THEN 'only team of user'
            WHEN ut.teams > 1          THEN 'user in several teams'
            WHEN u.identity IS NULL    THEN 'no identity'
            ELSE                            'user not in sheet' END AS team_basis
FROM units u
LEFT JOIN pt ON pt.policy_id = u.policy_id
LEFT JOIN ut ON ut.identity_key = lower(trim(u.identity))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A4. Policy tags
-- MAGIC Tags for each policy come from its own billing rows, which are non-reserved by definition, across all
-- MAGIC workspaces. custom_tags mixes policy, job and workspace tags (per a code reading, policy tags win on key
-- MAGIC conflicts with job tags). Only key=value pairs present on every row for the policy are kept, so job tags drop
-- MAGIC out unless the policy has a single job (`billing_jobs` = 1). A policy with no billing rows in the lookback
-- MAGIC gets no tags.

-- COMMAND ----------

-- DBTITLE 1,A4 · v_policy_rows
CREATE OR REPLACE TABLE v_policy_rows AS
SELECT
  u.usage_metadata.usage_policy_id AS policy_id,
  u.usage_metadata.job_id          AS job_id,
  u.workspace_id,
  u.custom_tags
FROM system.billing.usage u
CROSS JOIN v_params p
WHERE u.usage_metadata.usage_policy_id IN (SELECT DISTINCT policy_id FROM v_attributed)
  AND u.usage_date >= date_sub(current_date(), p.policy_lookback_days)

-- COMMAND ----------

-- DBTITLE 1,A4 · v_policy_tags
CREATE OR REPLACE TABLE v_policy_tags AS
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
-- MAGIC ### A4a. Tags per reserved policy
-- MAGIC NULL billing_rows = no billing rows in the lookback.

-- COMMAND ----------

-- DBTITLE 1,A4a. Tags per reserved policy
SELECT p.policy_id,
       t.billing_rows, t.billing_jobs, t.billing_workspaces,
       IF(t.policy_tags IS NULL, NULL, size(t.policy_tags)) AS n_tags,
       t.policy_tags
FROM (SELECT DISTINCT policy_id FROM v_attributed WHERE policy_id IS NOT NULL) p
LEFT JOIN v_policy_tags t USING (policy_id)
ORDER BY n_tags NULLS FIRST, p.policy_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A5. DBUs per policy per month
-- MAGIC Month of run start / hold start. `dbus` = GPU-hold-hours x `dbus_per_gpu_hour`. Runs and holds with no policy
-- MAGIC are the NULL row. `runs_gpus_assumed` counts job runs only (every notebook hold uses `nb_gpus x share`).
-- MAGIC ### A5a. DBUs per policy per month
-- MAGIC The select below prints `v_policy_dbus` by month and policy.

-- COMMAND ----------

-- DBTITLE 1,A5 · v_policy_dbus
CREATE OR REPLACE TABLE v_policy_dbus AS
SELECT
  date_trunc('MONTH', a.start_time)                                         AS month,
  a.policy_id,
  COUNT(DISTINCT IF(a.workload = 'job', a.workload_key, NULL))              AS jobs,
  SUM(IF(a.workload = 'job', 1, 0))                                         AS task_runs,
  COUNT(DISTINCT IF(a.workload = 'notebook', a.workload_key, NULL))         AS notebooks,
  SUM(IF(a.workload = 'notebook', 1, 0))                                    AS nb_holds,
  SUM(IF(a.workload = 'job' AND a.gpus_assumed, 1, 0))                      AS runs_gpus_assumed,
  SUM(IF(a.hold_s IS NULL, 1, 0))                                           AS runs_no_hold,
  ROUND(SUM(IF(a.workload = 'job', a.hold_s * a.gpus, 0)) / 3600.0, 1)      AS job_gpu_hours,
  ROUND(SUM(IF(a.workload = 'notebook', a.hold_s * a.gpus, 0)) / 3600.0, 1) AS nb_gpu_hours,
  ROUND(SUM(a.hold_s * a.gpus) / 3600.0, 1)                                 AS gpu_hours,
  ROUND(SUM(a.hold_s * a.gpus) / 3600.0 * MAX(p.dbus_per_gpu_hour), 2)      AS dbus,
  any_value(t.policy_tags)                                                  AS policy_tags  -- one tags row per policy; maps can't be grouped on
FROM v_attributed a
CROSS JOIN v_params p
LEFT JOIN v_policy_tags t USING (policy_id)
GROUP BY 1, 2

-- COMMAND ----------

-- DBTITLE 1,A5a. DBUs per policy per month
SELECT month, policy_id, jobs, task_runs, notebooks, nb_holds, runs_gpus_assumed, runs_no_hold,
       job_gpu_hours, nb_gpu_hours, gpu_hours, dbus, policy_tags
FROM v_policy_dbus
ORDER BY month DESC, dbus DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A6. GPU-hours and DBUs per team per month
-- MAGIC A5 by team instead of policy. `team_basis` says how the team was decided (A4); a NULL team is a unit whose
-- MAGIC policy and user don't pin one team down.

-- COMMAND ----------

-- DBTITLE 1,A6. GPU-hours and DBUs per team per month
SELECT date_trunc('MONTH', a.start_time)                                         AS month,
       a.team,
       a.team_basis,
       SUM(IF(a.workload = 'job', 1, 0))                                         AS task_runs,
       SUM(IF(a.workload = 'notebook', 1, 0))                                    AS nb_holds,
       COUNT(DISTINCT a.identity)                                                AS identities,
       ROUND(SUM(IF(a.workload = 'job', a.hold_s * a.gpus, 0)) / 3600.0, 1)      AS job_gpu_hours,
       ROUND(SUM(IF(a.workload = 'notebook', a.hold_s * a.gpus, 0)) / 3600.0, 1) AS nb_gpu_hours,
       ROUND(SUM(a.hold_s * a.gpus) / 3600.0, 1)                                 AS gpu_hours,
       ROUND(SUM(a.hold_s * a.gpus) / 3600.0 * MAX(p.dbus_per_gpu_hour), 2)      AS dbus
FROM v_attributed a
CROSS JOIN v_params p
GROUP BY 1, 2, 3
ORDER BY month DESC, gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## A7. Peak concurrent nodes per day
-- MAGIC How many reserved nodes were held at the same time: for the whole pool, and per use case, user and team. This
-- MAGIC needs each unit's start and end, not just its hours, so `v_unit_intervals` gives every `v_attributed` row an
-- MAGIC interval:
-- MAGIC - Job runs: `hold_s` ending at `run_end` (minus cleanup when `hold_basis` = 'execution'), so queue and setup
-- MAGIC   time at the front aren't counted as held. Runs with no `hold_s` yet are left out.
-- MAGIC - Notebook holds: `hold_start` to `hold_end` (N1).
-- MAGIC - `nodes` = `gpus` / `nb_gpus` (GPUs per node). A notebook row's `nodes` is its share of the hold.
-- MAGIC
-- MAGIC `use_case`, in order (the same rule as `team` in A4):
-- MAGIC - 'policy use case': the unit's policy maps to exactly one use case (N4 · v_policy_use_cases).
-- MAGIC - 'only use case of user': the identity is in exactly one use case that a policy's tags carry. Use cases that
-- MAGIC   matched no policy are ignored, as in N4; an identity with none that match falls back to its only sheet use case.
-- MAGIC - 'user in several use cases' / 'user not in sheet' / 'no identity': no use case (NULL).

-- COMMAND ----------

-- DBTITLE 1,A7 · v_unit_intervals
CREATE OR REPLACE TABLE v_unit_intervals AS
WITH puc AS (
  SELECT policy_id, MIN(use_case_name) AS use_case, COUNT(DISTINCT use_case_name) AS use_cases
  FROM v_policy_use_cases GROUP BY 1
),
uuc AS (
  -- the identity's use cases that a policy carries; if none do, all of its sheet use cases
  SELECT identity_key,
         COALESCE(MIN(IF(tagged, use_case_name, NULL)), MIN(use_case_name))                         AS use_case,
         COALESCE(NULLIF(COUNT(DISTINCT IF(tagged, use_case_name, NULL)), 0), COUNT(DISTINCT use_case_name)) AS use_cases
  FROM (
    SELECT t.identity_key, t.use_case_name,
           t.use_case_name IN (SELECT use_case_name FROM v_policy_use_cases) AS tagged
    FROM v_user_teams t
  )
  GROUP BY 1
),
t AS (
  SELECT a.*,
         IF(a.workload = 'job',
            timestampadd(SECOND, -CAST(IF(p.hold_basis = 'execution', COALESCE(r.cleanup_s, 0), 0) AS INT), r.run_end),
            timestampadd(SECOND, CAST(a.hold_s AS INT), a.start_time)) AS end_time,
         a.gpus / p.nb_gpus                                            AS nodes
  FROM v_attributed a
  CROSS JOIN v_params p
  LEFT JOIN v_run_policy r ON a.workload = 'job' AND CAST(r.run_id AS STRING) = a.unit_id
)
SELECT t.workload, t.unit_id, t.identity, t.policy_id, t.team, t.team_basis,
       CASE WHEN puc.use_cases = 1 THEN puc.use_case
            WHEN uuc.use_cases = 1 THEN uuc.use_case END                  AS use_case,
       CASE WHEN puc.use_cases = 1     THEN 'policy use case'
            WHEN uuc.use_cases = 1     THEN 'only use case of user'
            WHEN uuc.use_cases > 1     THEN 'user in several use cases'
            WHEN t.identity IS NULL    THEN 'no identity'
            ELSE                            'user not in sheet' END       AS use_case_basis,
       timestampadd(SECOND, -CAST(t.hold_s AS INT), t.end_time)           AS start_time,
       t.end_time,
       t.nodes
FROM t
LEFT JOIN puc ON puc.policy_id = t.policy_id
LEFT JOIN uuc ON uuc.identity_key = lower(trim(t.identity))
WHERE t.hold_s > 0 AND t.end_time IS NOT NULL

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7a. Node-hours by how the use case was decided
-- MAGIC How much of the held time has a use case at all. The NULL rows show up as '(unknown)' in A7c.

-- COMMAND ----------

-- DBTITLE 1,A7a. Node-hours by how the use case was decided
SELECT use_case_basis,
       COUNT(DISTINCT identity)                                                             AS identities,
       COUNT(DISTINCT use_case)                                                             AS use_cases,
       ROUND(SUM(IF(workload = 'job', nodes * TIMESTAMPDIFF(SECOND, start_time, end_time), 0)) / 3600.0, 1)      AS job_node_hours,
       ROUND(SUM(IF(workload = 'notebook', nodes * TIMESTAMPDIFF(SECOND, start_time, end_time), 0)) / 3600.0, 1) AS nb_node_hours,
       ROUND(SUM(nodes * TIMESTAMPDIFF(SECOND, start_time, end_time)) / 3600.0, 1)          AS node_hours
FROM v_unit_intervals
GROUP BY 1
ORDER BY node_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7 · v_daily_peaks
-- MAGIC `v_daily_peaks`: one row per (`dim`, `key`, `day`) with `peak_nodes`, the most nodes held at one moment that
-- MAGIC day, and `peak_at`, a moment it was reached. `dim` is 'pool' (key 'all'), 'workload' ('job' / 'notebook'),
-- MAGIC 'use_case', 'identity' or 'team'; a NULL key is '(unknown)'.
-- MAGIC - Nodes are whole: a notebook hold shared by several users counts as one node for each use case, user or team
-- MAGIC   in it, and once for the pool. So per-key peaks can add up to more than the pool's peak; they would anyway,
-- MAGIC   since each key peaks at its own time.
-- MAGIC - A unit still held at midnight carries into the next day.
-- MAGIC - Days are in the session time zone, from `since` to `until`. The last day or two run low: N2 leaves out
-- MAGIC   notebook holds from the last 24 hours, and audit and timeline rows arrive late.

-- COMMAND ----------

-- DBTITLE 1,A7 · v_daily_peaks
CREATE OR REPLACE TABLE v_daily_peaks AS
WITH dims AS (
  SELECT d.dim, d.key, u.unit_id, u.start_time, u.end_time, u.nodes
  FROM v_unit_intervals u
  LATERAL VIEW stack(5, 'pool',     'all',
                        'workload', u.workload,
                        'use_case', COALESCE(u.use_case, '(unknown)'),
                        'identity', COALESCE(lower(trim(u.identity)), '(unknown)'),
                        'team',     COALESCE(u.team, '(unknown)')) d AS dim, key
),
iv AS (
  -- one interval per unit within a key: a shared notebook hold's shares add up to at most one node
  SELECT dim, key, unit_id, MIN(start_time) AS start_time, MAX(end_time) AS end_time,
         CEIL(ROUND(SUM(nodes), 6)) AS nodes
  FROM dims
  GROUP BY 1, 2, 3
),
days AS (
  SELECT explode(sequence(to_date(since), to_date(until), INTERVAL 1 DAY)) AS day FROM v_params
),
ev AS (
  SELECT dim, key, start_time AS t, nodes AS d FROM iv
  UNION ALL
  SELECT dim, key, end_time, -nodes FROM iv
  UNION ALL
  -- a zero marker at each midnight, so a day with no start or end still reports what it carried in
  SELECT k.dim, k.key, CAST(d.day AS TIMESTAMP), 0
  FROM (SELECT DISTINCT dim, key FROM iv) k
  CROSS JOIN days d
),
run AS (
  -- at the same instant: releases, then the midnight marker, then acquisitions
  SELECT dim, key, t,
         SUM(d) OVER (PARTITION BY dim, key ORDER BY t, d ROWS UNBOUNDED PRECEDING) AS nodes_held
  FROM ev
)
SELECT r.dim, r.key, to_date(r.t) AS day,
       CAST(MAX(r.nodes_held) AS INT)  AS peak_nodes,
       max_by(r.t, r.nodes_held)       AS peak_at
FROM run r
JOIN days d ON d.day = to_date(r.t)
GROUP BY 1, 2, 3

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7b. Pool: peak nodes per day
-- MAGIC `peak_nodes` of `reserved_nodes` (= `reserved_gpus` / `nb_gpus`). `peak_job_nodes` and `peak_nb_nodes` are each
-- MAGIC workload's own peak, not its part of the pool's peak. `over_cap` should never be true: jobs and notebooks
-- MAGIC together can't hold more than the reservation, so a true row means the hold model overcounts that day.

-- COMMAND ----------

-- DBTITLE 1,A7b. Pool: peak nodes per day
SELECT d.day,
       d.peak_nodes,
       CAST(p.reserved_gpus / p.nb_gpus AS INT)        AS reserved_nodes,
       d.peak_at,
       COALESCE(j.peak_nodes, 0)                       AS peak_job_nodes,
       COALESCE(n.peak_nodes, 0)                       AS peak_nb_nodes,
       d.peak_nodes > p.reserved_gpus / p.nb_gpus      AS over_cap
FROM v_daily_peaks d
CROSS JOIN v_params p
LEFT JOIN v_daily_peaks j ON j.dim = 'workload' AND j.key = 'job'      AND j.day = d.day
LEFT JOIN v_daily_peaks n ON n.dim = 'workload' AND n.key = 'notebook' AND n.day = d.day
WHERE d.dim = 'pool'
ORDER BY d.day DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7c. Use case: peak nodes per day
-- MAGIC Days a use case held nothing are left out. '(unknown)' is every unit with no use case (A7a).

-- COMMAND ----------

-- DBTITLE 1,A7c. Use case: peak nodes per day
SELECT day, key AS use_case, peak_nodes, peak_at
FROM v_daily_peaks
WHERE dim = 'use_case' AND peak_nodes > 0
ORDER BY day DESC, peak_nodes DESC, use_case

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7d. User: peak nodes per day
-- MAGIC `identity` is the job's run-as or the notebook's user (A4), lowercased so one person's jobs and notebooks add up.

-- COMMAND ----------

-- DBTITLE 1,A7d. User: peak nodes per day
SELECT day, key AS identity, peak_nodes, peak_at
FROM v_daily_peaks
WHERE dim = 'identity' AND peak_nodes > 0
ORDER BY day DESC, peak_nodes DESC, identity

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7e. Team: peak nodes per day

-- COMMAND ----------

-- DBTITLE 1,A7e. Team: peak nodes per day
SELECT day, key AS team, peak_nodes, peak_at
FROM v_daily_peaks
WHERE dim = 'team' AND peak_nodes > 0
ORDER BY day DESC, peak_nodes DESC, team

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### A7f. GPU-hours per use case per month
-- MAGIC A6 by use case instead of team: reserved GPU-hours for jobs and notebooks, by month of the hold's start.
-- MAGIC `use_case_basis` says how the use case was decided (A7 · v_unit_intervals); a NULL use case is a unit whose
-- MAGIC policy and user don't pin one use case down. `nb_gpu_hours` is the notebook column.

-- COMMAND ----------

-- DBTITLE 1,A7f. GPU-hours per use case per month
SELECT date_trunc('MONTH', u.start_time)                                                AS month,
       u.use_case,
       u.use_case_basis,
       COUNT(DISTINCT lower(trim(u.identity)))                                          AS identities,
       ROUND(SUM(IF(u.workload = 'job', u.nodes * p.nb_gpus * TIMESTAMPDIFF(SECOND, u.start_time, u.end_time), 0)) / 3600.0, 1)      AS job_gpu_hours,
       ROUND(SUM(IF(u.workload = 'notebook', u.nodes * p.nb_gpus * TIMESTAMPDIFF(SECOND, u.start_time, u.end_time), 0)) / 3600.0, 1) AS nb_gpu_hours,
       ROUND(SUM(u.nodes * p.nb_gpus * TIMESTAMPDIFF(SECOND, u.start_time, u.end_time)) / 3600.0, 1)                                 AS gpu_hours
FROM v_unit_intervals u
CROSS JOIN v_params p
GROUP BY 1, 2, 3
ORDER BY month DESC, gpu_hours DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Part B: context
-- MAGIC ## B1. Reserved bill per day
-- MAGIC `implied_gpu_hours` = billed DBUs / rate. `reservation_gpu_hours` = reserved_gpus x 24.

-- COMMAND ----------

-- DBTITLE 1,B1 · v_reserved_daily
CREATE OR REPLACE TABLE v_reserved_daily AS
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
-- MAGIC ### B1a. Bill vs attributed GPU-hold per day
-- MAGIC Hours by run-start / attach date, so long runs land on their first day.

-- COMMAND ----------

-- DBTITLE 1,B1a. Bill vs attributed GPU-hold per day
SELECT d.usage_date,
       ROUND(d.reserved_dbus, 2)                                          AS reserved_dbus,
       ROUND(d.implied_gpu_hours, 1)                                      AS implied_gpu_hours,
       d.reservation_gpu_hours,
       ROUND(j.job_gpu_hours, 1)                                          AS job_gpu_hours,
       ROUND(j.nb_gpu_hours, 1)                                           AS nb_gpu_hours,
       ROUND(COALESCE(j.job_gpu_hours, 0) + COALESCE(j.nb_gpu_hours, 0), 1) AS attributed_gpu_hours,
       ROUND((j.job_gpu_hours + j.nb_gpu_hours) * p.dbus_per_gpu_hour, 2) AS attributed_dbus
FROM v_reserved_daily d
CROSS JOIN v_params p
LEFT JOIN (
  SELECT to_date(start_time) AS usage_date,
         SUM(IF(workload = 'job',      hold_s * gpus, 0)) / 3600.0 AS job_gpu_hours,
         SUM(IF(workload = 'notebook', hold_s * gpus, 0)) / 3600.0 AS nb_gpu_hours
  FROM v_attributed GROUP BY 1
) j USING (usage_date)
ORDER BY d.usage_date DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### B1b. Bill vs attributed GPU-hold per week
-- MAGIC B1a summed by week, which smooths out holds that start on one day and run into the next. Compare
-- MAGIC `attributed_gpu_hours` with `implied_gpu_hours` (what the reserved bill paid for): `attributed_share` above 1
-- MAGIC means hold time is overcounted. Below 1 is expected; the rest is reserved capacity nobody held.

-- COMMAND ----------

-- DBTITLE 1,B1b. Bill vs attributed GPU-hold per week
SELECT date_trunc('WEEK', d.usage_date)                                              AS week,
       COUNT(*)                                                                      AS days,
       ROUND(SUM(d.implied_gpu_hours), 1)                                            AS implied_gpu_hours,
       SUM(d.reservation_gpu_hours)                                                  AS reservation_gpu_hours,
       ROUND(SUM(COALESCE(j.job_gpu_hours, 0)), 1)                                   AS job_gpu_hours,
       ROUND(SUM(COALESCE(j.nb_gpu_hours, 0)), 1)                                    AS nb_gpu_hours,
       ROUND(SUM(COALESCE(j.job_gpu_hours, 0) + COALESCE(j.nb_gpu_hours, 0)), 1)     AS attributed_gpu_hours,
       ROUND(SUM(COALESCE(j.job_gpu_hours, 0) + COALESCE(j.nb_gpu_hours, 0))
             / NULLIF(SUM(d.implied_gpu_hours), 0), 2)                               AS attributed_share
FROM v_reserved_daily d
LEFT JOIN (
  SELECT to_date(start_time) AS usage_date,
         SUM(IF(workload = 'job',      hold_s * gpus, 0)) / 3600.0 AS job_gpu_hours,
         SUM(IF(workload = 'notebook', hold_s * gpus, 0)) / 3600.0 AS nb_gpu_hours
  FROM v_attributed GROUP BY 1
) j USING (usage_date)
GROUP BY 1
ORDER BY 1 DESC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B2. 8xH100 billed per workload
-- MAGIC Non-reserved serverless GPU / AIR rows whose per-workload hourly burn rate is at least 85% of one 8-GPU node
-- MAGIC at `dbus_per_gpu_hour` (A10 and 1xH100 are far below). Covers both billing schemas: serverless_gpu notebook
-- MAGIC rows have compute_type NULL and no pool ID. No lower date bound; rows before `since` are labeled.

-- COMMAND ----------

-- DBTITLE 1,B2. 8xH100 billed per workload
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
WHERE w.dbu_per_hour >= 0.85 * 8 * p.dbus_per_gpu_hour
GROUP BY ALL
ORDER BY g.usage_date

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## B3. Billed DBUs per usage policy, all products
-- MAGIC Everything billed to each policy for the window, plus the reserved node uptime (no policy) for comparison.
-- MAGIC These are the policies' non-reserved costs, separate from A5.

-- COMMAND ----------

-- DBTITLE 1,B3. Billed DBUs per usage policy, all products
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
-- MAGIC - Notebooks are attributed by inference (Part N header): reserved by elimination, policy from the notebook's
-- MAGIC   other billed usage, GPUs = `nb_gpus`. The direct record (AICM session-create activity log: accelerator,
-- MAGIC   guaranteed capacity ID, notebook ID, usage policy) is internal, not in any system table.
-- MAGIC - Notebook GPU-pool attaches (pool ID set in the Environment panel) and `@distributed` launches from a notebook
-- MAGIC   aren't checked to produce an AIR session ID in `attachNotebook`.
-- MAGIC - Tasks that get a GPU through `compute.hardware_accelerator` (e.g. a notebook task on GPU_8xH100) instead of
-- MAGIC   ai_runtime_task aren't in A1.
-- MAGIC - `update` requests that remove the policy via fields_to_remove are treated as "unchanged".
-- MAGIC - Policy changes made by an admin in the console don't go through the jobs API fields read here.
