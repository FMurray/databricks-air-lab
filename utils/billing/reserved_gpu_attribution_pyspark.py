# Databricks notebook source
# MAGIC %md
# MAGIC # Reserved GPU attribution: DBUs per usage policy (PySpark)
# MAGIC
# MAGIC PySpark version of `reserved_gpu_attribution.sql`: same chain, views and caveats (read that notebook's header,
# MAGIC and its Part N header for notebooks, for how the chain works and why). Each `v_*` SQL view is a DataFrame of the
# MAGIC same name without the prefix, and each check (A1a, MVP-1, N1a, ...) is a `display` cell. Run top to bottom in
# MAGIC the reservation's workspace.
# MAGIC
# MAGIC DataFrames are lazy like temp views: every `display` re-reads its sources. Set `SCRATCH_SCHEMA` to write the
# MAGIC audit read to a table once and read it back.

# COMMAND ----------

# MAGIC %md
# MAGIC ## A0. Parameters
# MAGIC Python values become literals in the scans, so filters prune files with nothing to keep in sync.

# COMMAND ----------

from datetime import date, datetime, timedelta, timezone

from pyspark.sql import Window
from pyspark.sql import functions as F

SINCE = datetime(2026, 8, 3, 19, tzinfo=timezone.utc)  # runs starting at or after; reservation start
UNTIL = None                                           # runs starting before; None = now
SPEC_LOOKBACK_START = date(2026, 2, 1)                 # audit read starts here (see A1)
ACCELERATOR = "GPU_8xH100"                             # reserved accelerator type
DBUS_PER_GPU_HOUR = 8.844                              # DBU rate per GPU-hour
HOLD_BASIS = "execution"                               # 'execution' | 'setup_execution_cleanup' | 'span'
DEFAULT_GPUS = 8                                       # GPUs assumed when accelerator_count isn't found (one node)
POLICY_LOOKBACK_DAYS = 90                              # days of policy billing rows read for tags
RESERVED_GPUS = 160                                    # nodes x GPUs per node (Part B only)
NB_HOLD_BASIS = "idle_timeout"                         # notebooks: 'idle_timeout' | 'activity' (N1)
NB_IDLE_TIMEOUT_MIN = 60                               # notebooks: AIR session idle timeout, minutes
NB_GPUS = 8                                            # notebooks: GPUs per session (one node)
NB_POLICY_LOOKBACK_DAYS = 90                           # notebooks: billing read for policy starts SINCE minus this
WORKSPACE_ID = None                                   # system tables span workspaces; set to scope them to one
SCRATCH_SCHEMA = None                                  # e.g. "main.scratch"; materializes the audit read

TABLES = {
    "audit": "system.access.audit",
    "timeline": "system.lakeflow.job_task_run_timeline",
    "usage": "system.billing.usage",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers

# COMMAND ----------

PHASES = ("setup", "execution", "cleanup")
RESERVED = "RESERVED_COMPUTE"
UNTIL_TS = F.lit(UNTIL) if UNTIL else F.current_timestamp()
AIR_SESSION_ID = r"^[0-9]{4}-[0-9]{6}-[a-zA-Z0-9]{1,8}-cnx$"  # attachNotebook clusterId of an AIR session
compute_type = F.col("product_features.ai_runtime.compute_type")


def system(name, ws_scoped=True):
    """A system table, filtered to WORKSPACE_ID when set."""
    df = spark.table(TABLES[name])
    if WORKSPACE_ID is None or not ws_scoped:
        return df
    return df.where(F.col("workspace_id") == F.lit(str(WORKSPACE_ID)).cast(df.schema["workspace_id"].dataType))


def materialize(df, name):
    """Write df to SCRATCH_SCHEMA.name and read it back, so later cells don't re-scan its sources."""
    if not SCRATCH_SCHEMA:
        return df
    df.write.mode("overwrite").saveAsTable(f"{SCRATCH_SCHEMA}.{name}")
    return spark.table(f"{SCRATCH_SCHEMA}.{name}")


def in_window(col):
    return (F.col(col) >= F.lit(SINCE)) & (F.col(col) < UNTIL_TS)


def latest(df, key, *order):
    """One row per key: the first by `order` (pass e.g. F.desc('event_time'))."""
    w = Window.partitionBy(key).orderBy(*order)
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def keyed_to_run(ev, run, as_of=True):
    """Join condition: audit rows (alias `ev`) that belong to a task run (alias `run`).

    A job event belongs to every run of that job (at or before the run's start when as_of); a submitRun event
    belongs to its one-time run, with no time condition because it's logged after the run is created.
    """
    e, r = (lambda c: F.col(f"{ev}.{c}")), (lambda c: F.col(f"{run}.{c}"))
    job = (e("key_type") == "job") & (e("job_key") == r("job_id"))
    if as_of:
        job = job & (e("event_time") <= r("run_start"))
    return job | ((e("key_type") == "run") & (e("job_key") == r("job_run_id")))


def n_if(cond, name):
    return F.sum(F.when(cond, 1).otherwise(0)).alias(name)


def gpu_hours(seconds="hold_s", name="gpu_hours", where=None):
    gpu_s = F.col(seconds) * F.col("gpus")
    if where is not None:
        gpu_s = F.when(where, gpu_s).otherwise(0)
    return F.round(F.sum(gpu_s) / 3600, 1).alias(name)


def dbus(seconds="hold_s", name="dbus"):
    return F.round(F.sum(F.col(seconds) * F.col("gpus")) / 3600 * DBUS_PER_GPU_HOUR, 2).alias(name)


def n_jobs():
    return F.countDistinct("job_key").alias("jobs")


def n_task_runs():
    return F.count("*").alias("task_runs")


def event_span():
    return [F.count("*").alias("events"), n_jobs(),
            F.min("event_time").alias("first_event"), F.max("event_time").alias("last_event")]

# COMMAND ----------

# MAGIC %md
# MAGIC # Part A: jobs -> policy -> DBUs
# MAGIC ## A1. AIR task specs from audit
# MAGIC `jobs_audit` is the one audit read. **Spec lookback:** a job whose last create / reset / update predates
# MAGIC `SPEC_LOOKBACK_START` has no spec and drops out of everything; widen it until MVP-1's `gpu_hours` stop growing.

# COMMAND ----------

rp = lambda k: F.col("request_params")[k]
new_settings = lambda path: F.get_json_object(rp("new_settings"), f"$.{path}")
result = lambda path: F.get_json_object(F.col("response.result"), f"$.{path}")
is_submit = F.col("action_name") == "submitRun"

jobs_audit = materialize(
    system("audit")
    .where((F.col("event_date") >= F.lit(SPEC_LOOKBACK_START))
           & (F.col("service_name") == "jobs")
           & F.col("action_name").isin("create", "reset", "update", "submitRun"))
    .select(
        "event_time",
        "action_name",
        F.col("user_identity.email").alias("actor"),
        F.when(is_submit, "run").otherwise("job").alias("key_type"),
        F.when(is_submit, result("run_id")).otherwise(F.coalesce(rp("job_id"), result("job_id"))).alias("job_key"),
        F.coalesce(rp("tasks"), new_settings("tasks")).alias("tasks_json"),
        F.coalesce(rp("usage_policy_id"), rp("budget_policy_id"),
                   new_settings("usage_policy_id"), new_settings("budget_policy_id")).alias("policy_id"),
        F.to_json("request_params").contains(ACCELERATOR).alias("mentions_accelerator"),
    )
    .where(F.col("job_key").isNotNull()),
    "air_jobs_audit",
)

# Jobs that ever had an AIR task; all their spec events are kept, so removing the AIR task is seen.
air_keys = jobs_audit.where(F.col("tasks_json").contains("ai_runtime_task")).select("key_type", "job_key").distinct()
accelerator_count = F.regexp_extract("t.ai_runtime_task", r'"accelerator_count"\s*:\s*"?(\d+)', 1)

task_specs = (
    jobs_audit.join(air_keys, ["key_type", "job_key"])
    .withColumn("t", F.explode_outer(F.from_json(
        "tasks_json", "array<struct<task_key:string,ai_runtime_task:string>>")))
    .select(
        "event_time", "action_name", "actor", "key_type", "job_key", "mentions_accelerator",
        "t.task_key",
        "t.ai_runtime_task",
        F.coalesce(F.col("t.ai_runtime_task").contains(ACCELERATOR), F.lit(False)).alias("is_reserved_accelerator"),
        F.when(accelerator_count != "", accelerator_count.cast("int")).alias("gpus"),
    )
)

# COMMAND ----------

# A1a. Specs by accelerator and GPU count
display(
    task_specs.where(F.col("ai_runtime_task").isNotNull())
    .groupBy("key_type", "is_reserved_accelerator", "gpus")
    .agg(F.countDistinct("job_key", "task_key").alias("tasks"), *event_span())
    .orderBy(F.desc("is_reserved_accelerator"), F.desc("jobs"))
)

# COMMAND ----------

# A1b. Events that name the accelerator but have no parsed tasks (truncated or differently shaped request_params)
display(
    task_specs.where(F.col("mentions_accelerator") & F.col("task_key").isNull())
    .groupBy("action_name", "key_type")
    .agg(*event_span())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A2. Task runs
# MAGIC Every task run of a job (or one-time run) in A1, one row per `run_id`. Timeline rows are read from `SINCE`
# MAGIC minus 7 days so runs aren't cut off; the window is applied on `run_start`.

# COMMAND ----------

timeline = system("timeline").where(F.col("period_start_time") >= F.lit(SINCE - timedelta(days=7)))
spec_keys = lambda key_type, col: (
    task_specs.where(F.col("key_type") == key_type).select(F.col("job_key").alias(col)).distinct())

# A run belongs to A1 by its job ID or (one-time runs) its run ID. A run matching both appears twice in the
# union; the MIN/MAX aggregates make that harmless.
task_runs = (
    timeline.join(spec_keys("job", "job_id"), "job_id", "left_semi")
    .unionByName(timeline.join(spec_keys("run", "job_run_id"), "job_run_id", "left_semi"))
    .groupBy("job_id", "job_run_id", "run_id", "task_key")
    .agg(
        F.min("period_start_time").alias("run_start"),
        F.max("period_end_time").alias("run_end"),
        F.max("result_state").alias("result_state"),
        *[F.max(f"{p}_duration_seconds").alias(f"{p}_s") for p in PHASES],  # run totals, not split per slice
    )
    .where(in_window("run_start"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Minimum viable check
# MAGIC The whole chain on `task_specs` and `task_runs` only. Cruder than the full chain: the spec isn't matched as of
# MAGIC run start, a `reset` that clears the policy isn't seen, and hold time is execution only.

# COMMAND ----------

mvp_runs = latest(
    task_runs.alias("r")
    .join(task_specs.where("is_reserved_accelerator").alias("s"),
          (F.col("s.task_key") == F.col("r.task_key")) & keyed_to_run("s", "r", as_of=False))
    .select("r.*", F.coalesce("s.gpus", F.lit(DEFAULT_GPUS)).alias("gpus"),
            F.col("s.job_key"), F.col("s.event_time").alias("spec_event_time")),
    "run_id", F.desc("spec_event_time"),
)
mvp = latest(
    mvp_runs.alias("r")
    .join(jobs_audit.where(F.col("policy_id").isNotNull()).alias("e"), keyed_to_run("e", "r"), "left")
    .select("r.*", "e.policy_id", F.col("e.event_time").alias("policy_event_time")),
    "run_id", F.desc("policy_event_time"),
)

# MVP-1. Reserved GPU-hours and DBUs per policy (NULL row = no policy found in audit)
display(
    mvp.groupBy("policy_id")
    .agg(n_jobs(), n_task_runs(), gpu_hours("execution_s"), dbus("execution_s"))
    .orderBy(F.desc("gpu_hours"))
)

# COMMAND ----------

# MVP-2. Custom tags on those policies' own billing rows (last 90 days)
usage_by_policy = (
    system("usage", ws_scoped=False)
    .where(F.col("usage_date") >= F.date_sub(F.current_date(), 90))
    .select(F.col("usage_metadata.usage_policy_id").alias("policy_id"), "usage_start_time", "custom_tags")
)
display(
    mvp.where(F.col("policy_id").isNotNull()).select("policy_id").distinct()
    .join(usage_by_policy, "policy_id", "left")
    .groupBy("policy_id")
    .agg(F.count("usage_start_time").alias("billing_rows_90d"),
         F.max_by("custom_tags", "usage_start_time").alias("sample_tags"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A2 (cont.). Task runs matched to their spec as of run start
# MAGIC Only 'reserved accelerator' rows flow on. `hold_s` is the GPU-hold seconds per `HOLD_BASIS`.

# COMMAND ----------

HOLD = {
    "execution": F.col("execution_s"),
    "setup_execution_cleanup": sum(F.coalesce(F.col(f"{p}_s"), F.lit(0)) for p in PHASES),
    "span": F.col("span_s"),
}

reserved_runs = (
    latest(
        task_runs.alias("r")
        .join(task_specs.alias("s"), (F.col("s.task_key") == F.col("r.task_key")) & keyed_to_run("s", "r"), "left")
        .select("r.*", "s.key_type", "s.job_key", "s.ai_runtime_task", "s.is_reserved_accelerator",
                F.col("s.gpus").alias("spec_gpus"),
                F.col("s.event_time").alias("spec_event_time"),
                F.col("s.action_name").alias("spec_action")),
        "run_id", F.desc("spec_event_time"), "spec_action",
    )
    .withColumn("spec_match",
                F.when(F.col("spec_event_time").isNull(), "no spec event before run")
                .when(F.col("ai_runtime_task").isNull(), "non-AIR task")
                .when(F.col("is_reserved_accelerator"), "reserved accelerator")
                .otherwise("other accelerator"))
    .withColumn("gpus", F.coalesce("spec_gpus", F.lit(DEFAULT_GPUS)))
    .withColumn("gpus_assumed", F.col("spec_gpus").isNull())
    .withColumn("span_s", F.expr("timestampdiff(SECOND, run_start, run_end)"))
    .withColumn("hold_s", HOLD[HOLD_BASIS])
)

# COMMAND ----------

# A2a. Task runs by spec match
display(
    reserved_runs.groupBy("spec_match", "key_type")
    .agg(n_jobs(), F.countDistinct("task_key").alias("task_keys"), n_task_runs(),
         n_if(F.col("gpus_assumed"), "runs_gpus_assumed"),
         F.round(F.sum("span_s") / 3600, 1).alias("span_hours"))
    .orderBy(F.desc("task_runs"))
)

# COMMAND ----------

# A2b. GPU-hold measures side by side, by result state
display(
    reserved_runs.where(F.col("spec_match") == "reserved accelerator")
    .groupBy("result_state")
    .agg(n_task_runs(), n_if(F.col("execution_s").isNull(), "runs_no_durations"),
         *[gpu_hours(f"{p}_s", f"{p}_gpu_hours") for p in (*PHASES, "span")],
         F.expr("round(percentile(setup_s, 0.5), 0)").alias("p50_setup_s"),
         F.expr("round(percentile(setup_s, 0.95), 0)").alias("p95_setup_s"))
    .orderBy(F.desc("task_runs"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A3. Policy per run
# MAGIC Latest policy-setting jobs event at or before run start (one-time runs: their submitRun). `create` / `reset` /
# MAGIC `submitRun` set the policy, including setting none; an `update` with no policy field leaves it unchanged.

# COMMAND ----------

policy_setters = jobs_audit.where(
    F.col("action_name").isin("create", "reset", "submitRun") | F.col("policy_id").isNotNull())

run_policy = latest(
    reserved_runs.where(F.col("spec_match") == "reserved accelerator").alias("r")
    .join(policy_setters.alias("e"), keyed_to_run("e", "r"), "left")
    .select("r.*", "e.policy_id",
            F.col("e.action_name").alias("policy_action"),
            F.col("e.event_time").alias("policy_event_time"),
            F.col("e.actor").alias("policy_actor")),
    "run_id", F.desc("policy_event_time"), "policy_action",
)

# COMMAND ----------

# A3a. Which jobs actions log a policy, for the jobs in A1
display(
    jobs_audit.join(task_specs.select("key_type", "job_key").distinct(), ["key_type", "job_key"])
    .groupBy("action_name")
    .agg(F.count("*").alias("events"), F.count("policy_id").alias("events_with_policy"), n_jobs())
    .orderBy("action_name")
)

# COMMAND ----------

# A3b. Runs per policy, and the event that set it
display(
    run_policy.groupBy("policy_id", "policy_action")
    .agg(n_jobs(), n_task_runs(), gpu_hours())
    .orderBy(F.asc_nulls_first("policy_id"), F.desc("gpu_hours"))
)

# COMMAND ----------

# A3c. Jobs with no explicit policy in audit (look up effective_usage_policy_id with `databricks jobs get <job_id>`)
display(
    run_policy.where(F.col("policy_id").isNull())
    .groupBy("key_type", "job_key")
    .agg(n_task_runs(), gpu_hours(),
         F.min("run_start").alias("first_run"), F.max("run_start").alias("last_run"),
         F.collect_set("policy_actor").alias("policy_event_actors"))
    .orderBy(F.desc("gpu_hours"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part N: notebooks -> policy
# MAGIC Interactive notebooks on serverless GPU. No audit event names their accelerator or policy, so (as in the SQL
# MAGIC notebook's Part N header): sessions come from attach events with an AIR session ID, reserved is by elimination
# MAGIC (no per-notebook GPU billing during the session), and the policy is the notebook's own billed usage policy.
# MAGIC
# MAGIC ## N1. AIR notebook sessions from audit
# MAGIC `nb_audit` is the one notebook audit read (`runCommand` rows need verbose audit logs; N1a shows whether
# MAGIC they're there). Sessions are per (notebook, user, session ID), from the first successful attach. A session ends
# MAGIC at last activity (attach, the user's detach, or its last `runCommand`) plus `NB_IDLE_TIMEOUT_MIN`, or when
# MAGIC the same user attaches the notebook elsewhere (`replaced_at`), or at `UNTIL`.

# COMMAND ----------

nb_audit = materialize(
    system("audit")
    .where((F.col("event_date") >= F.lit((SINCE - timedelta(days=7)).date()))
           & (F.col("service_name") == "notebook")
           & F.col("action_name").isin("attachNotebook", "detachNotebook", "runCommand"))
    .select(
        "workspace_id",
        "event_time",
        "action_name",
        F.col("user_identity.email").alias("actor"),
        rp("notebookId").alias("notebook_id"),
        rp("clusterId").alias("compute_id"),
        rp("path").alias("path"),
        (F.coalesce(F.col("response.status_code"), F.lit(200)) < 400).alias("ok"),
        F.coalesce(rp("clusterId").rlike(AIR_SESSION_ID), F.lit(False)).alias("is_air_session"),
    ),
    "air_nb_audit",
)

# N1a. Notebook events by compute ID shape (runCommand rows on AIR sessions = verbose audit is on)
display(
    nb_audit.groupBy(
        "action_name",
        F.when(F.col("is_air_session"), "AIR session (...-cnx)")
        .when(F.col("compute_id").isNull(), "no clusterId")
        .otherwise("other compute").alias("compute_kind"),
        "ok")
    .agg(F.count("*").alias("events"), F.countDistinct("notebook_id").alias("notebooks"),
         F.countDistinct("compute_id").alias("computes"),
         F.min("event_time").alias("first_event"), F.max("event_time").alias("last_event"))
    .orderBy("action_name", "compute_kind", "ok")
)

# COMMAND ----------

first_attaches = (
    nb_audit.where((F.col("action_name") == "attachNotebook") & F.col("ok")
                   & F.col("notebook_id").isNotNull() & F.col("compute_id").isNotNull())
    .groupBy("workspace_id", "notebook_id", "actor", "compute_id", "is_air_session")
    .agg(F.min("event_time").alias("attach_time"),
         F.max_by("path", "event_time").alias("path"),
         F.count("*").alias("attaches"))
    .withColumn("replaced_at", F.lead("attach_time").over(
        Window.partitionBy("workspace_id", "notebook_id", "actor").orderBy("attach_time")))
)
air_sessions = (
    first_attaches.where(F.col("is_air_session") & in_window("attach_time"))
    .withColumn("bound", F.coalesce("replaced_at", UNTIL_TS))
)

s, e = (lambda c: F.col(f"s.{c}")), (lambda c: F.col(f"e.{c}"))
is_cmd = e("action_name") == "runCommand"
session_activity = (
    air_sessions.alias("s")
    .join(nb_audit.alias("e"),
          (e("workspace_id") == s("workspace_id")) & (e("notebook_id") == s("notebook_id"))
          & (e("event_time") >= s("attach_time")) & (e("event_time") < s("bound"))
          & ((is_cmd & (e("compute_id") == s("compute_id")))
             | ((e("action_name") == "detachNotebook") & (e("actor") == s("actor")))))
    .groupBy("s.compute_id", "s.notebook_id", "s.actor")
    .agg(F.max(F.when(is_cmd, e("event_time"))).alias("last_command"),
         n_if(is_cmd, "commands"),
         F.max(F.when(~is_cmd, e("event_time"))).alias("last_detach"))
)

NB_HOLD = {"idle_timeout": "idle_end", "activity": "activity_end"}
nb_sessions = (
    air_sessions.join(session_activity, ["compute_id", "notebook_id", "actor"], "left")
    .withColumnRenamed("compute_id", "session_id")
    .withColumn("commands", F.coalesce("commands", F.lit(0)))
    .withColumn("last_activity", F.greatest("attach_time", F.coalesce("last_command", "attach_time"),
                                            F.coalesce("last_detach", "attach_time")))
    .withColumn("activity_end", F.least("bound", "last_activity"))
    .withColumn("idle_end", F.least("bound", F.expr(f"timestampadd(MINUTE, {NB_IDLE_TIMEOUT_MIN}, last_activity)")))
    .withColumn("hold_end", F.col(NB_HOLD[NB_HOLD_BASIS]))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## N2. Reserved sessions, by elimination
# MAGIC `nb_match`: 'on-demand GPU' = the notebook billed GPU usage overlapping the session (attach to `idle_end`);
# MAGIC 'recent' = no rows yet but the session ended in the last 24 hours; 'reserved' = no rows (only these flow on).
# MAGIC N2a's `od_rows_no_workload_key` must be 0: on-demand GPU rows with no notebook / job / workload ID would hide
# MAGIC on-demand notebook sessions, which would then read as reserved.

# COMMAND ----------

nb_gpu_billing = (
    system("usage")
    .where((F.col("usage_date") >= F.lit((SINCE - timedelta(days=7)).date()))
           & (F.col("product_features.ai_runtime").isNotNull() | F.col("product_features.serverless_gpu").isNotNull())
           & (F.coalesce(compute_type, F.lit("")) != RESERVED))
    .select("workspace_id",
            F.col("usage_metadata.notebook_id").alias("notebook_id"),
            F.col("usage_metadata.job_id").alias("job_id"),
            F.col("usage_metadata.ai_runtime_workload_id").alias("workload_id"),
            "usage_start_time", "usage_end_time", "usage_quantity")
)

b = lambda c: F.col(f"b.{c}")
on_demand = (
    nb_sessions.alias("s")
    .join(nb_gpu_billing.alias("b"),
          (b("workspace_id") == s("workspace_id")) & (b("notebook_id") == s("notebook_id"))
          & (b("usage_end_time") > s("attach_time")) & (b("usage_start_time") < s("idle_end")), "left")
    .groupBy("s.session_id", "s.notebook_id", "s.actor")
    .agg(F.count(b("usage_quantity")).alias("od_rows"), F.sum(b("usage_quantity")).alias("od_dbus"))
)
seconds_to = lambda end: F.expr(f"timestampdiff(SECOND, attach_time, {end})")
nb_reserved_sessions = (
    nb_sessions.join(on_demand, ["session_id", "notebook_id", "actor"])
    .withColumn("nb_match",
                F.when(F.col("od_rows") > 0, "on-demand GPU")
                .when(F.col("idle_end") > F.current_timestamp() - F.expr("INTERVAL 24 HOURS"), "recent")
                .otherwise("reserved"))
    .withColumn("gpus", F.lit(NB_GPUS))
    .withColumn("activity_s", seconds_to("activity_end"))
    .withColumn("idle_timeout_s", seconds_to("idle_end"))
    .withColumn("hold_s", seconds_to("hold_end"))
)

# COMMAND ----------

# N2a. Sessions by match, with both hold measures
no_workload_key = (
    nb_gpu_billing.where(F.col("notebook_id").isNull() & F.col("job_id").isNull() & F.col("workload_id").isNull()
                         & in_window("usage_start_time"))
    .groupBy("workspace_id").agg(F.count("*").alias("od_rows_no_workload_key"))
)
display(
    nb_reserved_sessions.groupBy("workspace_id", "nb_match")
    .agg(F.countDistinct("notebook_id").alias("notebooks"), F.countDistinct("actor").alias("users"),
         F.count("*").alias("sessions"),
         n_if(F.col("commands") > 0, "sessions_with_commands"),
         n_if(F.col("replaced_at").isNotNull(), "sessions_replaced"),
         gpu_hours("activity_s", "activity_gpu_hours"),
         gpu_hours("idle_timeout_s", "idle_timeout_gpu_hours"),
         F.expr("round(percentile(idle_timeout_s, 0.5) / 3600, 2)").alias("p50_session_hours"),
         F.expr("round(percentile(idle_timeout_s, 0.95) / 3600, 2)").alias("p95_session_hours"))
    .join(no_workload_key, "workspace_id", "left")
    .fillna(0, ["od_rows_no_workload_key"])
    .orderBy("workspace_id", F.desc("sessions"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## N3. Policy per reserved session
# MAGIC Each 'reserved' session takes the policy of one of its notebook's policy-bearing billing rows (any product): the
# MAGIC same user's first, then the latest at or before attach, else the earliest after. NULL policy = no such row in
# MAGIC the lookback: check the notebook's Environment panel.

# COMMAND ----------

reserved_nb = nb_reserved_sessions.where(F.col("nb_match") == "reserved")
nb_policy_rows = (
    system("usage")
    .where((F.col("usage_date") >= F.lit((SINCE - timedelta(days=NB_POLICY_LOOKBACK_DAYS)).date()))
           & F.col("usage_metadata.usage_policy_id").isNotNull())
    .select("workspace_id",
            F.col("usage_metadata.notebook_id").alias("notebook_id"),
            F.col("usage_metadata.usage_policy_id").alias("policy_id"),
            F.coalesce("identity_metadata.run_as", "identity_metadata.owned_by",
                       "identity_metadata.created_by").alias("principal"),
            "usage_start_time", "billing_origin_product")
    .join(reserved_nb.select("notebook_id").distinct(), "notebook_id", "left_semi")
)

r = lambda c: F.col(f"r.{c}")
same_notebook = (r("workspace_id") == s("workspace_id")) & (r("notebook_id") == s("notebook_id"))
nb_session_policy = (
    latest(
        reserved_nb.alias("s").join(nb_policy_rows.alias("r"), same_notebook, "left")
        .select("s.*", "r.policy_id",
                r("usage_start_time").alias("policy_row_time"),
                r("billing_origin_product").alias("policy_row_product"),
                (r("principal") == s("actor")).alias("_same_user"),
                (r("usage_start_time") <= s("attach_time")).alias("_before")),
        ["session_id", "notebook_id", "actor"],
        F.desc_nulls_last("_same_user"),
        F.desc_nulls_last(F.when(F.col("_before"), F.col("policy_row_time"))),
        F.asc("policy_row_time"),
    )
    .withColumn("policy_source", F.when(F.col("policy_id").isNotNull(), F.concat(
        F.when(F.col("_same_user"), "same user").otherwise("other user"), F.lit(", "),
        F.when(F.col("_before"), "before").otherwise("after"), F.lit(" attach"))))
    .drop("_same_user", "_before")
)

# N3a. Reserved notebook GPU-hours by policy source ('same user, before attach' is strongest)
display(
    nb_session_policy.groupBy("policy_source", "policy_row_product")
    .agg(F.countDistinct("notebook_id").alias("notebooks"), F.count("*").alias("sessions"), gpu_hours())
    .orderBy(F.desc("gpu_hours"))
)

# COMMAND ----------

# N3b. Sessions whose notebook billed under a different policy before and after the attach (same user)
display(
    nb_session_policy.alias("s")
    .join(nb_policy_rows.alias("r"), same_notebook & (r("principal") == s("actor")))
    .groupBy("s.session_id", "s.notebook_id", "s.path", "s.actor", "s.attach_time", "s.hold_s", "s.gpus",
             s("policy_id").alias("chosen_policy"))
    .agg(F.max_by(r("policy_id"), F.when(r("usage_start_time") <= s("attach_time"), r("usage_start_time")))
         .alias("policy_before"),
         F.min_by(r("policy_id"), F.when(r("usage_start_time") > s("attach_time"), r("usage_start_time")))
         .alias("policy_after"))
    .where(F.col("policy_before") != F.col("policy_after"))
    .select("notebook_id", "path", "actor", "attach_time", "chosen_policy", "policy_before", "policy_after",
            F.round(F.col("hold_s") * F.col("gpus") / 3600, 1).alias("gpu_hours"))
    .orderBy(F.desc("gpu_hours"))
)

# COMMAND ----------

# N3c. Who ran it: reserved notebook GPU-hours per user and policy (actor = the attaching user)
display(
    nb_session_policy.groupBy("actor", "policy_id")
    .agg(F.countDistinct("notebook_id").alias("notebooks"), F.count("*").alias("sessions"), gpu_hours())
    .orderBy(F.desc("gpu_hours"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part A (cont.): tags and DBUs, jobs and notebooks
# MAGIC `attributed`: one row per charged unit, a reserved task run (A3) or a reserved notebook session (N3).

# COMMAND ----------

attributed = (
    run_policy.select(F.lit("job").alias("workload"), "key_type", F.col("job_key").alias("workload_key"),
                      F.col("run_id").cast("string").alias("unit_id"), F.col("run_start").alias("start_time"),
                      "policy_id", "gpus", "gpus_assumed", "hold_s")
    .unionByName(nb_session_policy.select(
        F.lit("notebook").alias("workload"), F.lit("notebook").alias("key_type"),
        F.col("notebook_id").alias("workload_key"), F.col("session_id").alias("unit_id"),
        F.col("attach_time").alias("start_time"), "policy_id", "gpus",
        F.lit(True).alias("gpus_assumed"), "hold_s"))
)
is_job, is_nb = F.col("workload") == "job", F.col("workload") == "notebook"

# COMMAND ----------

# MAGIC %md
# MAGIC ## A4. Policy tags
# MAGIC From each policy's own billing rows, across all workspaces. Only key=value pairs present on every row for the
# MAGIC policy are kept, so job tags drop out unless the policy has a single job (`billing_jobs` = 1).

# COMMAND ----------

policy_rows = (
    system("usage", ws_scoped=False)
    .where(F.col("usage_date") >= F.date_sub(F.current_date(), POLICY_LOOKBACK_DAYS))
    .select(F.col("usage_metadata.usage_policy_id").alias("policy_id"),
            F.col("usage_metadata.job_id").alias("job_id"), "workspace_id", "custom_tags")
    .join(attributed.select("policy_id").where(F.col("policy_id").isNotNull()).distinct(), "policy_id", "left_semi")
)
evidence = policy_rows.groupBy("policy_id").agg(
    F.count("*").alias("billing_rows"),
    F.countDistinct("job_id").alias("billing_jobs"),
    F.countDistinct("workspace_id").alias("billing_workspaces"),
)
tag_counts = policy_rows.select("policy_id", F.explode("custom_tags")).groupBy("policy_id", "key", "value").count()

policy_tags = (
    evidence.join(tag_counts, "policy_id", "left")
    .groupBy(*evidence.columns)
    .agg(F.map_from_entries(F.collect_list(
        F.when(F.col("count") == F.col("billing_rows"), F.struct("key", "value")))).alias("policy_tags"))
)

# COMMAND ----------

# A4a. Tags per reserved policy (NULL billing_rows = no billing rows in the lookback)
display(
    attributed.where(F.col("policy_id").isNotNull()).select("policy_id").distinct()
    .join(policy_tags, "policy_id", "left")
    .withColumn("n_tags", F.when(F.col("policy_tags").isNotNull(), F.size("policy_tags")))
    .orderBy(F.asc_nulls_first("n_tags"), "policy_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A5. DBUs per policy per month
# MAGIC Month of run start / attach. `dbus` = GPU-hold-hours x `DBUS_PER_GPU_HOUR`. Runs and sessions with no policy
# MAGIC are the NULL row. `runs_gpus_assumed` counts job runs only (every notebook session uses `NB_GPUS`).

# COMMAND ----------

policy_dbus = (
    attributed.groupBy(F.date_trunc("MONTH", "start_time").alias("month"), "policy_id")
    .agg(F.countDistinct(F.when(is_job, F.col("workload_key"))).alias("jobs"), n_if(is_job, "task_runs"),
         F.countDistinct(F.when(is_nb, F.col("workload_key"))).alias("notebooks"), n_if(is_nb, "nb_sessions"),
         n_if(is_job & F.col("gpus_assumed"), "runs_gpus_assumed"),
         n_if(F.col("hold_s").isNull(), "runs_no_hold"),
         gpu_hours(name="job_gpu_hours", where=is_job), gpu_hours(name="nb_gpu_hours", where=is_nb),
         gpu_hours(), dbus())
    .join(policy_tags.select("policy_id", "policy_tags"), "policy_id", "left")
)
display(policy_dbus.orderBy(F.desc("month"), F.desc("dbus")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Part B: context
# MAGIC ## B1. Reserved bill per day
# MAGIC `implied_gpu_hours` = billed DBUs / rate. `reservation_gpu_hours` = RESERVED_GPUS x 24.
# MAGIC B1a puts it next to attributed GPU-hold by run-start / attach date (long runs land on their first day).

# COMMAND ----------

usage = system("usage")

reserved_daily = (
    usage.where((compute_type == RESERVED) & in_window("usage_start_time"))
    .groupBy("usage_date")
    .agg(F.round(F.sum("usage_quantity"), 2).alias("reserved_dbus"),
         F.round(F.sum("usage_quantity") / DBUS_PER_GPU_HOUR, 1).alias("implied_gpu_hours"))
    .withColumn("reservation_gpu_hours", F.lit(RESERVED_GPUS * 24))
)

# B1a. Bill vs attributed GPU-hold per day
display(
    reserved_daily.join(
        attributed.groupBy(F.to_date("start_time").alias("usage_date"))
        .agg(gpu_hours(name="job_gpu_hours", where=is_job), gpu_hours(name="nb_gpu_hours", where=is_nb),
             dbus(name="attributed_dbus")),
        "usage_date", "left")
    .orderBy(F.desc("usage_date"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## B2. 8xH100 billed per workload
# MAGIC Non-reserved serverless GPU / AIR rows whose per-workload hourly burn rate is at least 60 DBU/hour (one 8xH100
# MAGIC node is about 70.8; A10 and 1xH100 are far below). Covers both billing schemas. No lower date bound; rows
# MAGIC before `SINCE` are labeled.

# COMMAND ----------

gpu_rows = (
    usage.where((F.col("product_features.ai_runtime").isNotNull()
                 | F.col("product_features.serverless_gpu").isNotNull())
                & (F.coalesce(compute_type, F.lit("")) != RESERVED))
    .select("usage_date", "usage_start_time", "usage_end_time", "usage_quantity", "sku_name",
            "billing_origin_product",
            F.coalesce(compute_type, F.lit("NULL (serverless_gpu schema)")).alias("compute_type"),
            F.col("product_features.serverless_gpu.workload_type").alias("sgc_workload_type"),
            F.coalesce("usage_metadata.ai_runtime_workload_id", "usage_metadata.job_run_id",
                       "usage_metadata.notebook_id").alias("workload_key"),
            F.col("usage_metadata.usage_policy_id").alias("policy_id"),
            F.date_trunc("HOUR", "usage_start_time").alias("hour"))
)
burn_rate = gpu_rows.groupBy("workload_key", "hour").agg(
    (F.sum("usage_quantity") / F.greatest(
        F.expr("timestampdiff(SECOND, min(usage_start_time), max(usage_end_time))") / 3600, F.lit(1 / 60))
     ).alias("dbu_per_hour"))

display(
    gpu_rows.join(burn_rate.where(F.col("dbu_per_hour") >= 60), ["workload_key", "hour"])
    .withColumn("period", F.when(F.col("usage_start_time") < F.lit(SINCE), "before since").otherwise("in window"))
    .groupBy("usage_date", "period", "billing_origin_product", "compute_type", "sgc_workload_type", "sku_name")
    .agg(F.countDistinct("workload_key").alias("workloads"),
         F.countDistinct("policy_id").alias("policies"),
         F.round(F.sum("usage_quantity"), 2).alias("dbus"))
    .orderBy("usage_date")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## B3. Billed DBUs per usage policy, all products
# MAGIC Everything billed to each policy for the window, plus the reserved node uptime (no policy) for comparison.
# MAGIC These are the policies' non-reserved costs, separate from A5.

# COMMAND ----------

policy_id = F.col("usage_metadata.usage_policy_id")
display(
    usage.where(in_window("usage_start_time") & (F.col("usage_unit") == "DBU"))
    .groupBy(
        F.when(compute_type == RESERVED, "reserved node uptime (no policy)")
        .when(policy_id.isNotNull(), "policy")
        .otherwise("no policy (other)").alias("bucket"),
        policy_id.alias("policy_id"),
        "billing_origin_product")
    .agg(F.round(F.sum("usage_quantity"), 2).alias("dbus"))
    .orderBy("bucket", "policy_id", F.desc("dbus"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Not covered yet
# MAGIC Same as `reserved_gpu_attribution.sql`: notebooks are attributed by inference only (no system table records
# MAGIC their accelerator or policy), notebook GPU-pool attaches and `@distributed` launches are unchecked,
# MAGIC `compute.hardware_accelerator` GPU tasks, policy removal via `fields_to_remove`, and console policy changes.
