# Databricks notebook source
# MAGIC %md
# MAGIC # AIR sessions and notebook API probe
# MAGIC
# MAGIC Companion to `reserved_gpu_attribution.sql`. Tests, from inside the workspace, whether these REST endpoints
# MAGIC answer, and whether they can replace Part N's inferred links (policy per session, hold time):
# MAGIC
# MAGIC | Probe | Endpoint | Visibility (per a code reading) | Would give |
# MAGIC |---|---|---|---|
# MAGIC | P1 | `GET /api/2.0/ai-compute-manager/sessions` | public, undocumented | every AIR session: notebook, policy, accelerator, start / termination |
# MAGIC | P2 | `GET /api/2.0/ai-compute-manager/sessions/{session_id}` | public, undocumented | the same for one session |
# MAGIC | P3 | `GET /api/2.0/notebooks/{notebook_id}/usage-policy` | internal | the notebook's stored policy (Environment panel) |
# MAGIC | P4 | `GET /api/2.0/workspace/get-status` | public | baseline: object info only, no policy expected |
# MAGIC
# MAGIC All calls are read-only GETs as the user running this notebook (notebook auth, no token handling). Session
# MAGIC records are purged about 65 days after the session ends (per the code), so P1 can't cover older sessions.
# MAGIC
# MAGIC **Result from a first customer run: P1/P2 only return the caller's own sessions.** Per the handlers
# MAGIC (`AiComputeGatewayHandlers.handleListSessions` / `handleGetSession`), only internal system users see every
# MAGIC session; a regular user, workspace admins included, sees only sessions they own. P2 on another user's session
# MAGIC fails with RESOURCE_CONFLICT ("does not support notebook sharing"). The list also returns `cluster_infos`
# MAGIC (no notebook or policy); only P2's `session_info` has `attribution_info`. So these endpoints can't attribute
# MAGIC a workspace's sessions, except per user, run by each session's owner.
# MAGIC
# MAGIC **Safe to paste.** Printed output is status codes, error codes, field paths, value types, enum counts and
# MAGIC aggregate counts. IDs, emails, paths and policy IDs are not printed. The raw API rows are written to
# MAGIC `v_air_sessions_raw` in the scratch schema, inside the workspace, for the attribution notebook to join on.
# MAGIC
# MAGIC Run after `reserved_gpu_attribution.sql` has built its stage tables (needs `v_nb_session_policy`).

# COMMAND ----------

# DBTITLE 1,P0 · settings
SCRATCH_SCHEMA = "<catalog>.<schema>"  # same scratch schema as reserved_gpu_attribution.sql
MAX_PAGES = 100                        # P1 paging cap
SAMPLE_NOTEBOOKS = 5                   # P2-P4: NULL-policy reserved sessions to probe

assert "<" not in SCRATCH_SCHEMA, "Set SCRATCH_SCHEMA to the attribution notebook's scratch schema"

# COMMAND ----------

# DBTITLE 1,P0 · helpers
import json
import re
from collections import Counter
from datetime import datetime, timezone

from databricks.sdk import WorkspaceClient

SESSION_RE = re.compile(r"^\d{4}-\d{6}-[A-Za-z0-9]{1,8}-cnx$")
MASK_RE = re.compile(r"\S+@\S+|\d{4}-\d{6}-[A-Za-z0-9]{1,8}-cnx|[0-9a-fA-F-]{12,}|\d{6,}")


def call(client, path, query=None):
    """GET `path`. Returns (status, body); never raises. Error messages are masked and truncated."""
    try:
        return "200", client.api_client.do("GET", path, query=query) or {}
    except Exception as e:  # noqa: BLE001 - a probe reports every failure the same way
        msg = MASK_RE.sub("<id>", str(e))[:160]
        return type(e).__name__, {"error_code": getattr(e, "error_code", None), "message": msg}


def flatten(obj, prefix=""):
    """Yield (path, scalar) pairs; list indices collapse to []."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        if not obj:
            yield prefix + "[]", None
        for v in obj:
            yield from flatten(v, prefix + "[]")
    else:
        yield prefix, obj


def shape(obj):
    """Field path -> value type names. Values are never included."""
    out = {}
    for p, v in flatten(obj):
        out.setdefault(p, set()).add(type(v).__name__)
    return {p: "|".join(sorted(t)) for p, t in sorted(out.items())}


def first(obj, path_pred=None, value_pred=None):
    for p, v in flatten(obj):
        if (path_pred is None or path_pred(p)) and (value_pred is None or value_pred(v)):
            return v
    return None


def ends(*names):
    return lambda p: any(p == n or p.endswith("." + n) for n in names)


def session_items(body):
    """The first top-level list of objects in a list response, with its key."""
    for k, v in body.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k, v
    return None, []


def parse_session(s):
    return {
        "session_id": first(s, value_pred=lambda v: isinstance(v, str) and bool(SESSION_RE.match(v))),
        "notebook_id": first(s, ends("notebook_id")),
        "policy_id": first(s, ends("budget_policy_id", "usage_policy_id")),
        "accelerator": first(s, ends("hardware_accelerator_type")),
        "state": first(s, ends("state")),
        "started_at": first(s, ends("started_at")),
        "termination_requested_at": first(s, ends("termination_requested_at")),
    }


def as_utc(v):
    """Best-effort timestamp parse: epoch s/ms, RFC 3339, or {seconds, nanos}. None if unparseable."""
    try:
        if isinstance(v, dict) and "seconds" in v:
            v = int(v["seconds"])
        if isinstance(v, str) and v.isdigit():
            v = int(v)
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=timezone.utc)
        if isinstance(v, str):  # fromisoformat takes at most microseconds and no "Z" before Python 3.11
            return datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", v).replace("Z", "+00:00"))
    except (ValueError, OverflowError, TypeError):
        pass
    return None


def report(title, payload):
    print(f"===== {title}")
    print(json.dumps(payload, indent=2, default=str))

# COMMAND ----------

# DBTITLE 1,P1. List AIR sessions
w = WorkspaceClient()

statuses, sessions, list_key, first_body, query = [], [], None, None, {}
for _ in range(MAX_PAGES):
    status, body = call(w, "/api/2.0/ai-compute-manager/sessions", query)
    statuses.append(status)
    if status != "200":
        first_body = first_body or body
        break
    first_body = first_body or body
    key, items = session_items(body)
    list_key = list_key or key
    sessions += items
    token = body.get("next_page_token")
    if not token:
        break
    query = {"page_token": token}

parsed = [parse_session(s) for s in sessions]
starts = [t for t in (as_utc(p["started_at"]) for p in parsed) if t]
report("P1 sessions list", {
    "statuses": dict(Counter(statuses)),
    "pages": len(statuses),
    "hit_page_cap": len(statuses) == MAX_PAGES,
    "top_level_keys": sorted(first_body) if isinstance(first_body, dict) else None,
    "error": first_body if statuses and statuses[-1] != "200" else None,
    "list_key": list_key,
    "sessions": len(sessions),
    "with_session_id": sum(p["session_id"] is not None for p in parsed),
    "with_notebook_id": sum(p["notebook_id"] is not None for p in parsed),
    "with_policy_id": sum(p["policy_id"] is not None for p in parsed),
    "distinct_notebooks": len({p["notebook_id"] for p in parsed if p["notebook_id"]}),
    "states": dict(Counter(str(p["state"]) for p in parsed)),
    "accelerators": dict(Counter(str(p["accelerator"]) for p in parsed)),
    "started_at_types": dict(Counter(type(p["started_at"]).__name__ for p in parsed)),
    "earliest_start": min(starts) if starts else None,
    "latest_start": max(starts) if starts else None,
    "session_fields": shape(sessions[0]) if sessions else None,
})

# COMMAND ----------

# DBTITLE 1,P1a · v_air_sessions_raw
# All fields as strings; `raw` keeps the full JSON for later parsing.
rows = [{k: (None if v is None else str(v)) for k, v in p.items()} | {"raw": json.dumps(s)}
        for p, s in zip(parsed, sessions)]
cols = ["session_id", "notebook_id", "policy_id", "accelerator", "state", "started_at",
        "termination_requested_at", "raw"]
schema = ", ".join(f"{c} string" for c in cols)
(spark.createDataFrame([[r[c] for c in cols] for r in rows], schema)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{SCRATCH_SCHEMA}.v_air_sessions_raw"))
print(f"wrote {len(rows)} rows to {SCRATCH_SCHEMA}.v_air_sessions_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P1b. API coverage of the attribution notebook's reserved sessions
# MAGIC Each reserved session from N3 (`v_nb_session_policy`), bucketed by what the sessions API says about it. The
# MAGIC key rows: 'in API, policy; N3 had NULL' is what the API recovers, and 'policy matches N3' vs 'differs' tests N3's
# MAGIC inference. 'not in API' sessions older than about 65 days are expected (purged).

# COMMAND ----------

# DBTITLE 1,P1b. API coverage of reserved sessions
display(spark.sql(f"""
WITH ours AS (
  SELECT session_id,
         MAX(policy_id)                   AS n3_policy,
         SUM(hold_s * gpus) / 3600.0      AS gpu_hours,
         MIN(hold_start)                  AS first_hold
  FROM {SCRATCH_SCHEMA}.v_nb_session_policy
  GROUP BY 1
),
api AS (
  SELECT session_id, MAX(policy_id) AS api_policy
  FROM {SCRATCH_SCHEMA}.v_air_sessions_raw
  WHERE session_id IS NOT NULL
  GROUP BY 1
)
SELECT CASE WHEN a.session_id IS NULL       THEN '1. not in API'
            WHEN a.api_policy IS NULL       THEN '2. in API, no policy'
            WHEN o.n3_policy IS NULL        THEN '3. in API, policy; N3 had NULL'
            WHEN o.n3_policy = a.api_policy THEN '4. in API, policy matches N3'
            ELSE                                 '5. in API, policy differs from N3' END AS bucket,
       COUNT(*)                                                                   AS sessions,
       SUM(IF(o.first_hold >= current_timestamp() - INTERVAL 60 DAYS, 1, 0))      AS sessions_last_60d,
       ROUND(SUM(o.gpu_hours), 1)                                                 AS gpu_hours,
       MIN(o.first_hold)                                                          AS earliest_hold
FROM ours o
LEFT JOIN api a USING (session_id)
GROUP BY 1
ORDER BY 1
"""))

# COMMAND ----------

# DBTITLE 1,P2-P4. One session and notebook at a time
sample = spark.sql(f"""
  SELECT session_id, notebook_id, MAX(path) AS path
  FROM {SCRATCH_SCHEMA}.v_nb_session_policy
  WHERE policy_id IS NULL
  GROUP BY 1, 2
  ORDER BY MAX(hold_start) DESC  -- most recent first, so they're within the API's retention
  LIMIT {SAMPLE_NOTEBOOKS}
""").collect()

probes = {
    "P2 session detail": lambda r: call(w, f"/api/2.0/ai-compute-manager/sessions/{r.session_id}"),
    "P3 notebook usage-policy": lambda r: call(w, f"/api/2.0/notebooks/{r.notebook_id}/usage-policy"),
    "P4 workspace get-status": lambda r: (call(w, "/api/2.0/workspace/get-status", {"path": r.path})
                                          if r.path else ("no path", {})),
}
for name, probe in probes.items():
    results = [probe(r) for r in sample]
    ok = [b for s, b in results if s == "200"]
    report(name, {
        "probed": len(results),
        "statuses": dict(Counter(s for s, _ in results)),
        "errors": [b for s, b in results if s != "200"][:2],
        "fields": shape(ok[0]) if ok else None,
        "with_policy_value": sum(first(b, ends("budget_policy_id", "usage_policy_id", "policy_id")) is not None
                                 for b in ok),
    })
