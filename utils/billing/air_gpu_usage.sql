-- AIR GPU-hours per day, by capacity (reserved / on-demand), accelerator, workload shape and usage policy.
--
-- The dataset behind the AIR GPU usage dashboard: bundles/workspace-notebooks/build_dashboard.py copies this file
-- into the dashboard JSON (edit here, then rebuild). It also runs as is in the SQL editor, which prompts for the
-- named parameters.
--
-- One row per (day, workspace, workload, capacity, accelerator, shape, policy, and how each was found).
-- It is a single-statement version of reserved_gpu_attribution.sql (Parts A and N), widened from reserved only to
-- every AIR workload. Read that notebook's header for the sources and inferences. In short:
--   * GPU-hours are GPU-hold time x GPUs, not billed DBUs / rate: reserved billing rows are per node, with no job,
--     notebook or policy, so the bill can't be split. No DBU rate is needed.
--   * Jobs: the accelerator and GPU count come from the ai_runtime_task spec in the jobs audit events, matched
--     to each task run (system.lakeflow.job_task_run_timeline) as of the run's start. Hold time follows the
--     hold_basis parameter: 'execution' (default), 'setup_execution_cleanup' or 'span'.
--   * Notebooks: AIR sessions from notebook audit events, cut into holds at idle gaps longer than the idle timeout
--     (the hold ends at last activity + timeout). The accelerator isn't logged anywhere for notebooks: reserved
--     holds are taken to be on the first reserved accelerator with nb_gpus GPUs, and on-demand holds are
--     'unknown (notebook)' with nb_gpus GPUs assumed. gpus_basis marks both.
--   * Capacity: a workload with on-demand AIR billing rows (same job run, or same notebook, overlapping in time) is
--     on-demand. One on an accelerator that isn't reserved is on-demand. Otherwise it's reserved, by
--     elimination, unless it ended in the last 24 hours, when billing may not have landed: 'pending billing'.
--   * Policy: from the on-demand billing rows when there are any (what was actually billed). Otherwise, for jobs,
--     the latest policy-setting jobs audit event before the run start, and, for notebooks, the notebook's
--     nearest policy-bearing billing row of any product. '(none found)' = neither. policy_source says which.
--   * A workload's GPU-hours are spread over the days it spans, by wall-clock overlap, and clipped to since.
--
-- Scan cost: reads system.access.audit (jobs and notebook events), system.billing.usage and the job task run
-- timeline, from since minus a lookback. Each dashboard refresh runs it, so keep since modest.
--
-- Parameters:
--   since                  DATE    first day shown
--   reserved_accelerators  STRING  comma-separated accelerator types with a reservation, e.g. GPU_8xH100
--   hold_basis             STRING  jobs: 'execution' | 'setup_execution_cleanup' | 'span'
--   nb_idle_timeout_min    INT     notebooks: AIR session idle timeout, minutes (60 by default)
--   nb_gpus                INT     notebooks: GPUs per session (one node)
--   spec_lookback_days     INT     days before since to read job specs from; jobs last edited earlier drop out
WITH
-- ---------------------------------------------------------------------------------------------------------------
-- Jobs
-- ---------------------------------------------------------------------------------------------------------------
jobs_audit AS (
  SELECT
    a.workspace_id,
    a.event_time,
    a.action_name,
    IF(a.action_name = 'submitRun', 'run', 'job') AS key_type,
    IF(a.action_name = 'submitRun',
       get_json_object(a.response.result, '$.run_id'),
       COALESCE(a.request_params['job_id'], get_json_object(a.response.result, '$.job_id'))) AS job_key,
    COALESCE(a.request_params['tasks'],
             get_json_object(a.request_params['new_settings'], '$.tasks'))               AS tasks_json,
    COALESCE(a.request_params['usage_policy_id'],
             a.request_params['budget_policy_id'],
             get_json_object(a.request_params['new_settings'], '$.usage_policy_id'),
             get_json_object(a.request_params['new_settings'], '$.budget_policy_id'))    AS policy_id
  FROM system.access.audit a
  WHERE a.event_date >= date_sub(:since, :spec_lookback_days)
    AND a.service_name = 'jobs'
    AND a.action_name IN ('create', 'reset', 'update', 'submitRun')
),
-- Jobs (or one-time runs) that ever had an AIR task; all their spec events are kept, so removing it is seen.
air_keys AS (
  SELECT DISTINCT workspace_id, key_type, job_key
  FROM jobs_audit
  WHERE tasks_json LIKE '%ai_runtime_task%' AND job_key IS NOT NULL
),
task_specs AS (
  SELECT
    e.workspace_id, e.event_time, e.action_name, e.key_type, e.job_key,
    t.task_key,
    t.ai_runtime_task,
    NULLIF(regexp_extract(t.ai_runtime_task, '"accelerator_type"\\s*:\\s*"([A-Za-z0-9_]+)"', 1), '') AS accelerator,
    try_cast(regexp_extract(t.ai_runtime_task, '"accelerator_count"\\s*:\\s*"?(\\d+)', 1) AS INT)    AS spec_gpus
  FROM jobs_audit e
  JOIN air_keys k USING (workspace_id, key_type, job_key)
  LATERAL VIEW OUTER explode(
    from_json(e.tasks_json, 'array<struct<task_key:string,ai_runtime_task:string>>')) x AS t
),
task_runs AS (
  SELECT
    t.workspace_id, t.job_id, t.job_run_id, t.run_id, t.task_key,
    MIN(t.period_start_time)          AS run_start,
    MAX(t.period_end_time)            AS run_end,
    MAX(t.setup_duration_seconds)     AS setup_s,     -- run totals, repeated on every hourly slice
    MAX(t.execution_duration_seconds) AS execution_s,
    MAX(t.cleanup_duration_seconds)   AS cleanup_s
  FROM system.lakeflow.job_task_run_timeline t
  LEFT JOIN (SELECT DISTINCT workspace_id, job_key FROM air_keys WHERE key_type = 'job') kj
    ON kj.workspace_id = t.workspace_id AND kj.job_key = t.job_id
  LEFT JOIN (SELECT DISTINCT workspace_id, job_key FROM air_keys WHERE key_type = 'run') kr
    ON kr.workspace_id = t.workspace_id AND kr.job_key = t.job_run_id
  WHERE t.period_start_time >= CAST(date_sub(:since, 7) AS TIMESTAMP)  -- runs that started up to 7 days before since
    AND (kj.job_key IS NOT NULL OR kr.job_key IS NOT NULL)
  GROUP BY 1, 2, 3, 4, 5
),
-- Each task run takes its task's spec as of the run start (one-time runs: their submitRun).
task_runs_spec AS (
  SELECT r.*, s.key_type, s.job_key, s.ai_runtime_task, s.accelerator, s.spec_gpus
  FROM task_runs r
  JOIN task_specs s
    ON s.workspace_id = r.workspace_id
   AND s.task_key = r.task_key
   AND ((s.key_type = 'job' AND s.job_key = r.job_id AND s.event_time <= r.run_start)
     OR (s.key_type = 'run' AND s.job_key = r.job_run_id))
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.workspace_id, r.run_id ORDER BY s.event_time DESC, s.action_name) = 1
),
job_runs AS (
  SELECT * FROM task_runs_spec WHERE ai_runtime_task IS NOT NULL
),
-- Non-reserved AIR / serverless GPU billing rows (both billing schemas). Reserved rows are per node and carry no
-- workload, so they can't be matched to anything.
od_billing AS (
  SELECT
    u.workspace_id,
    u.usage_metadata.job_id          AS job_id,
    u.usage_metadata.job_run_id      AS job_run_id,
    u.usage_metadata.notebook_id     AS notebook_id,
    u.usage_metadata.usage_policy_id AS policy_id,
    u.usage_start_time,
    u.usage_end_time,
    u.usage_quantity
  FROM system.billing.usage u
  WHERE u.usage_date >= date_sub(:since, 7)
    AND (u.product_features.ai_runtime IS NOT NULL OR u.product_features.serverless_gpu IS NOT NULL)
    AND COALESCE(u.product_features.ai_runtime.compute_type, '') <> 'RESERVED_COMPUTE'
    AND u.sku_name NOT LIKE '%PROVISIONED_CAPACITY%'
),
job_od AS (
  SELECT r.workspace_id, r.run_id,
         COUNT(b.usage_quantity)          AS od_rows,
         max_by(b.policy_id, IF(b.policy_id IS NULL, NULL, b.usage_quantity)) AS od_policy_id
  FROM job_runs r
  JOIN od_billing b
    ON b.workspace_id = r.workspace_id
   AND b.job_id = r.job_id
   AND (b.job_run_id IS NULL OR b.job_run_id = r.job_run_id)
   AND b.usage_end_time > r.run_start
   AND b.usage_start_time < COALESCE(r.run_end, current_timestamp())
  GROUP BY 1, 2
),
-- Latest policy-setting audit event at or before the run start. create / reset / submitRun set the policy, including
-- setting none; an update with no policy field leaves it unchanged.
job_audit_policy AS (
  SELECT r.workspace_id, r.run_id, e.policy_id
  FROM job_runs r
  JOIN jobs_audit e
    ON e.workspace_id = r.workspace_id
   AND e.key_type = r.key_type
   AND e.job_key  = r.job_key
   AND (e.key_type = 'run' OR e.event_time <= r.run_start)
   AND (e.action_name IN ('create', 'reset', 'submitRun') OR e.policy_id IS NOT NULL)
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.workspace_id, r.run_id ORDER BY e.event_time DESC, e.action_name) = 1
),
job_units AS (
  SELECT
    'job'                                                              AS workload,
    r.workspace_id,
    CAST(r.run_id AS STRING)                                           AS unit_id,
    r.run_start                                                        AS start_t,
    COALESCE(r.run_end, current_timestamp())                           AS end_t,
    CASE :hold_basis
      WHEN 'execution'               THEN r.execution_s
      WHEN 'setup_execution_cleanup' THEN COALESCE(r.setup_s, 0) + COALESCE(r.execution_s, 0) + COALESCE(r.cleanup_s, 0)
      WHEN 'span'                    THEN TIMESTAMPDIFF(SECOND, r.run_start, COALESCE(r.run_end, current_timestamp()))
    END                                                                AS hold_s,
    COALESCE(r.accelerator, 'unknown')                                 AS accelerator,
    try_cast(regexp_extract(r.accelerator, '^GPU_(\\d+)x', 1) AS INT)  AS gpus_per_node,
    COALESCE(r.spec_gpus, try_cast(regexp_extract(r.accelerator, '^GPU_(\\d+)x', 1) AS INT), 8) AS gpus,
    IF(r.spec_gpus IS NOT NULL, 'job spec', 'assumed: one node')       AS gpus_basis,
    CASE WHEN COALESCE(o.od_rows, 0) > 0
           THEN 'on-demand'
         WHEN NOT COALESCE(array_contains(split(regexp_replace(:reserved_accelerators, '\\s', ''), ','), r.accelerator), false)
           THEN 'on-demand'
         WHEN COALESCE(r.run_end, current_timestamp()) > current_timestamp() - INTERVAL 24 HOURS
           THEN 'pending billing'
         ELSE 'reserved' END                                           AS capacity,
    CASE WHEN COALESCE(o.od_rows, 0) > 0
           THEN 'on-demand billing rows'
         WHEN NOT COALESCE(array_contains(split(regexp_replace(:reserved_accelerators, '\\s', ''), ','), r.accelerator), false)
           THEN 'accelerator not reserved'
         WHEN COALESCE(r.run_end, current_timestamp()) > current_timestamp() - INTERVAL 24 HOURS
           THEN 'ended < 24 h ago'
         ELSE 'no on-demand billing (elimination)' END                 AS capacity_basis,
    COALESCE(o.od_policy_id, p.policy_id)                              AS policy_id,
    CASE WHEN o.od_policy_id IS NOT NULL THEN 'billing'
         WHEN p.policy_id IS NOT NULL    THEN 'jobs audit'
         ELSE 'none found' END                                         AS policy_source
  FROM job_runs r
  LEFT JOIN job_od o           USING (workspace_id, run_id)
  LEFT JOIN job_audit_policy p USING (workspace_id, run_id)
),
-- ---------------------------------------------------------------------------------------------------------------
-- Notebooks
-- ---------------------------------------------------------------------------------------------------------------
nb_audit AS (
  SELECT
    a.workspace_id,
    a.event_time,
    a.action_name,
    a.user_identity.email                                 AS actor,
    a.request_params['notebookId']                        AS notebook_id,
    COALESCE(a.request_params['clusterId'],
             try_element_at(filter(map_values(a.request_params),
                                   v -> v RLIKE '^[0-9]{4}-[0-9]{6}-[a-zA-Z0-9]{1,8}-cnx$'), 1)) AS compute_id,
    try_cast(a.request_params['executionTime'] AS DOUBLE) AS exec_s,
    COALESCE(a.response.status_code, 200) < 400           AS ok
  FROM system.access.audit a
  WHERE a.event_date >= date_sub(:since, 7)
    AND a.service_name = 'notebook'
    AND a.action_name IN ('attachNotebook', 'submitCommand', 'runCommand')
),
nb_sessions AS (
  -- AIR session IDs are shaped MMDD-HHMMSS-xxxxxxxx-cnx
  SELECT workspace_id, compute_id AS session_id
  FROM nb_audit
  WHERE action_name = 'attachNotebook' AND ok AND notebook_id IS NOT NULL
    AND compute_id RLIKE '^[0-9]{4}-[0-9]{6}-[a-zA-Z0-9]{1,8}-cnx$'
  GROUP BY 1, 2
),
nb_ev AS (
  -- activity points: attaches and submitted commands; verbose runCommand spans end minus executionTime to end
  SELECT e.workspace_id, e.compute_id AS session_id, e.notebook_id, e.actor, e.action_name,
         IF(e.action_name = 'runCommand',
            timestamp_millis(unix_millis(e.event_time) - CAST(COALESCE(e.exec_s, 0) * 1000 AS BIGINT)),
            e.event_time)                                             AS start_t,
         e.event_time                                                 AS end_t,
         IF(e.action_name = 'runCommand', COALESCE(e.exec_s, 0), 0.0) AS exec_s
  FROM nb_audit e
  JOIN nb_sessions s ON s.workspace_id = e.workspace_id AND s.session_id = e.compute_id
  WHERE e.notebook_id IS NOT NULL
    AND ((e.action_name = 'attachNotebook' AND e.ok) OR e.action_name IN ('submitCommand', 'runCommand'))
),
nb_ordered AS (
  SELECT nb_ev.*,
         MAX(end_t) OVER (PARTITION BY workspace_id, session_id ORDER BY start_t, end_t, notebook_id, actor
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end
  FROM nb_ev
),
nb_numbered AS (
  -- a new hold starts when activity begins more than the idle timeout after everything before it ended
  SELECT o.*,
         SUM(IF(o.prev_end IS NULL OR o.start_t > timestampadd(MINUTE, :nb_idle_timeout_min, o.prev_end), 1, 0))
           OVER (PARTITION BY o.workspace_id, o.session_id ORDER BY o.start_t, o.end_t, o.notebook_id, o.actor
                 ROWS UNBOUNDED PRECEDING) AS island
  FROM nb_ordered o
),
nb_isl AS (
  SELECT workspace_id, session_id, island,
         MIN(start_t)                                                  AS hold_start,
         MAX(end_t)                                                    AS last_activity,
         SUM(IF(action_name IN ('submitCommand', 'runCommand'), 1, 0)) AS hold_commands,
         SUM(exec_s)                                                   AS hold_exec_s,
         COUNT(DISTINCT notebook_id, actor)                            AS contributors
  FROM nb_numbered
  GROUP BY 1, 2, 3
),
nb_contrib AS (
  SELECT workspace_id, session_id, island, notebook_id, actor,
         SUM(IF(action_name IN ('submitCommand', 'runCommand'), 1, 0)) AS commands,
         SUM(exec_s)                                                   AS exec_s
  FROM nb_numbered
  GROUP BY ALL
),
-- attaching one of the hold's notebooks (same user) to other compute ends the session
nb_ended AS (
  SELECT c.workspace_id, c.session_id, c.island, MIN(a.event_time) AS ended_by_attach_at
  FROM nb_contrib c
  JOIN nb_isl i USING (workspace_id, session_id, island)
  JOIN nb_audit a
    ON a.workspace_id = c.workspace_id
   AND a.notebook_id  = c.notebook_id
   AND a.actor        = c.actor
   AND a.action_name  = 'attachNotebook' AND a.ok
   AND a.compute_id  <> c.session_id
   AND a.event_time  >= i.last_activity
  GROUP BY 1, 2, 3
),
-- one row per (hold, notebook, user); share splits a shared hold so it counts once
nb_holds AS (
  SELECT
    c.workspace_id, c.session_id, c.island, c.notebook_id, c.actor,
    i.hold_start,
    LEAST(current_timestamp(), COALESCE(d.ended_by_attach_at, current_timestamp()),
          timestampadd(MINUTE, :nb_idle_timeout_min, i.last_activity))                 AS hold_end,
    COALESCE(c.exec_s / NULLIF(i.hold_exec_s, 0), c.commands / NULLIF(i.hold_commands, 0),
             1.0 / i.contributors)                                                     AS share
  FROM nb_contrib c
  JOIN nb_isl i USING (workspace_id, session_id, island)
  LEFT JOIN nb_ended d USING (workspace_id, session_id, island)
),
nb_od AS (
  SELECT h.workspace_id, h.session_id, h.island, h.notebook_id,
         COUNT(b.usage_quantity)                                               AS od_rows,
         max_by(b.policy_id, IF(b.policy_id IS NULL, NULL, b.usage_quantity)) AS od_policy_id
  FROM nb_holds h
  LEFT JOIN od_billing b
    ON b.workspace_id     = h.workspace_id
   AND b.notebook_id      = h.notebook_id
   AND b.usage_end_time   > h.hold_start
   AND b.usage_start_time < h.hold_end
  GROUP BY 1, 2, 3, 4
),
-- policy-bearing billing rows (any product) of these notebooks: the notebook's stored policy
nb_policy_rows AS (
  SELECT
    u.workspace_id,
    u.usage_metadata.notebook_id     AS notebook_id,
    u.usage_metadata.usage_policy_id AS policy_id,
    COALESCE(u.identity_metadata.run_as, u.identity_metadata.owned_by, u.identity_metadata.created_by) AS principal,
    u.usage_start_time
  FROM system.billing.usage u
  WHERE u.usage_date >= date_sub(:since, 90)
    AND u.usage_metadata.usage_policy_id IS NOT NULL
    AND u.usage_metadata.notebook_id IN (SELECT DISTINCT notebook_id FROM nb_holds)
),
-- same user's rows first, then the latest at or before the hold start, else the earliest after
nb_fallback_policy AS (
  SELECT h.workspace_id, h.session_id, h.island, h.notebook_id, h.actor, r.policy_id
  FROM nb_holds h
  LEFT JOIN nb_policy_rows r
    ON r.workspace_id = h.workspace_id AND r.notebook_id = h.notebook_id
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY h.workspace_id, h.session_id, h.island, h.notebook_id, h.actor
    ORDER BY (r.principal = h.actor) DESC NULLS LAST,
             IF(r.usage_start_time <= h.hold_start, r.usage_start_time, NULL) DESC NULLS LAST,
             r.usage_start_time) = 1
),
nb_units AS (
  SELECT
    'notebook'                                                         AS workload,
    h.workspace_id,
    concat(h.session_id, '#', h.island)                                AS unit_id,
    h.hold_start                                                       AS start_t,
    h.hold_end                                                         AS end_t,
    TIMESTAMPDIFF(SECOND, h.hold_start, h.hold_end)                    AS hold_s,
    IF(m.capacity = 'reserved',
       element_at(split(regexp_replace(:reserved_accelerators, '\\s', ''), ','), 1),
       'unknown (notebook)')                                           AS accelerator,
    CAST(:nb_gpus AS INT)                                              AS gpus_per_node,
    :nb_gpus * h.share                                                 AS gpus,
    'notebook: assumed nb_gpus'                                        AS gpus_basis,
    m.capacity,
    m.capacity_basis,
    COALESCE(o.od_policy_id, f.policy_id)                              AS policy_id,
    CASE WHEN o.od_policy_id IS NOT NULL THEN 'billing'
         WHEN f.policy_id IS NOT NULL    THEN 'notebook billing (other compute)'
         ELSE 'none found' END                                         AS policy_source
  FROM nb_holds h
  JOIN nb_od o USING (workspace_id, session_id, island, notebook_id)
  -- a hold is on-demand if any of its notebooks billed on-demand GPU during it
  JOIN (
    SELECT o2.workspace_id, o2.session_id, o2.island,
           CASE WHEN SUM(o2.od_rows) > 0 THEN 'on-demand'
                WHEN MAX(h2.hold_end) > current_timestamp() - INTERVAL 24 HOURS THEN 'pending billing'
                ELSE 'reserved' END AS capacity,
           CASE WHEN SUM(o2.od_rows) > 0 THEN 'on-demand billing rows'
                WHEN MAX(h2.hold_end) > current_timestamp() - INTERVAL 24 HOURS THEN 'ended < 24 h ago'
                ELSE 'no on-demand billing (elimination)' END AS capacity_basis
    FROM nb_od o2
    JOIN nb_holds h2 USING (workspace_id, session_id, island, notebook_id)
    GROUP BY 1, 2, 3
  ) m USING (workspace_id, session_id, island)
  LEFT JOIN nb_fallback_policy f USING (workspace_id, session_id, island, notebook_id, actor)
),
-- ---------------------------------------------------------------------------------------------------------------
-- Both, spread over the days each unit spans
-- ---------------------------------------------------------------------------------------------------------------
units AS (
  SELECT * FROM job_units
  UNION ALL
  SELECT * FROM nb_units
),
unit_days AS (
  SELECT
    u.*,
    d.day,
    GREATEST(TIMESTAMPDIFF(SECOND, u.start_t, u.end_t), 0) AS span_s,
    GREATEST(TIMESTAMPDIFF(SECOND,
               GREATEST(u.start_t, CAST(d.day AS TIMESTAMP)),
               LEAST(u.end_t, CAST(date_add(d.day, 1) AS TIMESTAMP))), 0) AS day_s
  FROM units u
  LATERAL VIEW explode(sequence(to_date(u.start_t), to_date(u.end_t))) d AS day
  WHERE u.hold_s > 0
)
SELECT
  day,
  CAST(workspace_id AS STRING)                                         AS workspace_id,
  workload,
  capacity,
  accelerator,
  CASE WHEN gpus_per_node IS NULL OR gpus_per_node = 0 THEN concat(CAST(ROUND(gpus) AS INT), ' GPUs')
       ELSE concat(CAST(CEIL(gpus / gpus_per_node) AS INT),
                   IF(CEIL(gpus / gpus_per_node) = 1, ' node x ', ' nodes x '),
                   gpus_per_node, ' GPU') END                          AS shape,
  IF(gpus_per_node IS NOT NULL AND gpus > gpus_per_node, 'multi-node', 'single node') AS node_class,
  COALESCE(policy_id, '(none found)')                                  AS policy_id,
  policy_source,
  capacity_basis,
  gpus_basis,
  ROUND(SUM(hold_s * gpus / 3600.0 * IF(span_s = 0, 1.0, day_s / span_s)), 3) AS gpu_hours,
  SUM(IF(to_date(start_t) = day, 1, 0))                               AS units_started
FROM unit_days
WHERE day >= :since
  AND (span_s = 0 OR day_s > 0)
GROUP BY ALL
