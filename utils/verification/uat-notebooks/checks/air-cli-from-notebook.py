# Databricks notebook source
# MAGIC %md
# MAGIC # UAT check: air CLI from a notebook
# MAGIC Can a tester drive the AIR CLI entirely from a serverless notebook (no local machine)?
# MAGIC Installs the CLI from the vendored wheel set (PyPI is unreachable here by design), auths
# MAGIC with the notebook context token, then runs the CLI against the mirrored repo via FUSE.
# MAGIC
# MAGIC Shape: **CPU** (the CLI itself needs no GPU; the workload it submits requests its own).
# MAGIC `air_mode` widget: `dry-run` (default — validate/plan only, no cost) or `submit`
# MAGIC (really submits `workloads/probes/envvar-probe.yaml`, 1×A10, 12-min timeout, and polls
# MAGIC it to terminal). NB air 1.0.0 dry-run skips upload AND submission, so only `submit`
# MAGIC proves the auth + upload path end to end.
# MAGIC Returns JSON via `dbutils.notebook.exit` for the DRIVER; also runnable standalone.

# COMMAND ----------

# 0) Is the CLI preinstalled in the serverless env? Recorded BEFORE %pip (stashed to /tmp —
#    it survives restartPython) — this is the premise of the vendored-wheels approach.
import importlib.util, json as _json, shutil as _shutil, sys as _sys
_pre = {"which_air": _shutil.which("air"),
        "databricks_air_module": importlib.util.find_spec("databricks_air") is not None,
        "python": _sys.version.split()[0]}
open("/tmp/air_preinstalled.json", "w").write(_json.dumps(_pre))
print("pre-%pip state:", _pre)

# COMMAND ----------

# MAGIC %pip install --quiet --no-index --find-links /Workspace/Shared/databricks-air-lab/uat/wheels databricks-air

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("air_mode", "dry-run", "air_mode: dry-run | submit (1xA10, real money)")

import json, os, re, shutil, subprocess, sys, time

REPO = "/Workspace/Shared/databricks-air-lab"
PROBE_YAML = "workloads/probes/envvar-probe.yaml"

results = {}
def record(name, ok, detail=""):
    results[name] = {"ok": (None if ok is None else bool(ok)), "detail": str(detail)[:200]}
    print({True: "✅", False: "❌", None: "⏭️"}[results[name]["ok"]], name, "—", str(detail)[:200])

# COMMAND ----------

# 0b) read back the pre-%pip state: expected NOT preinstalled (vendoring is required).
#     If this ever flips ❌ the runtime started bundling the CLI — retire the wheels dir.
try:
    pre = json.load(open("/tmp/air_preinstalled.json"))
    record("cli_not_preinstalled", pre["which_air"] is None and not pre["databricks_air_module"],
           f"pre-%pip: which(air)={pre['which_air']} module={pre['databricks_air_module']} py{pre['python']}")
except Exception as e:
    record("cli_not_preinstalled", False, f"stash unreadable: {type(e).__name__}: {e}")

# COMMAND ----------

# 1) CLI installed from vendored wheels? (the %pip cell above already succeeded or the
#    notebook died — this asserts the console script is actually invokable)
AIR = shutil.which("air") or os.path.join(os.path.dirname(sys.executable), "air")
try:
    v = subprocess.run([AIR, "--version"], capture_output=True, text=True, timeout=120)
    ver = next((l.strip() for l in v.stdout.splitlines() if re.search(r"v\d+\.\d+", l)), v.stdout[-80:])
    record("cli_install", v.returncode == 0, f"{AIR} -> {ver} (python {sys.version.split()[0]})")
except Exception as e:
    record("cli_install", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# 2) auth: notebook context token -> env vars (the CLI's profile-less auth path)
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"] = ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
record("context_auth_env", True, f"host={os.environ['DATABRICKS_HOST']}")

# COMMAND ----------

# 3) dry-run against the FUSE-mirrored repo (schema + config plumbing; air 1.0.0 does NOT
#    upload or hit the Jobs API here)
try:
    r = subprocess.run([AIR, "run", "--dry-run", "-f", PROBE_YAML],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    sentinel = "dry-run" in (r.stdout + r.stderr).lower()
    record("air_dry_run", r.returncode == 0 and sentinel,
           f"exit={r.returncode}; " + (r.stdout + r.stderr).strip().replace("\n", " | ")[:150])
except Exception as e:
    record("air_dry_run", False, f"{type(e).__name__}: {e}")

# COMMAND ----------

# 4) real submit + poll to terminal (only in submit mode) — the end-to-end proof:
#    snapshot upload from FUSE, Jobs API submission, and the workload's own SUCCESS
if dbutils.widgets.get("air_mode").strip() != "submit":
    record("air_submit", None, "SKIPPED — air_mode=dry-run (set air_mode=submit for the e2e proof)")
    record("air_submitted_run", None, "SKIPPED — air_mode=dry-run")
else:
    run_id = None
    try:
        r = subprocess.run([AIR, "run", "--json", "-f", PROBE_YAML],
                           cwd=REPO, capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
        m = re.search(r'"?run_id"?\D*(\d{8,})', out)
        run_id = m.group(1) if m else None
        record("air_submit", r.returncode == 0 and run_id is not None,
               f"exit={r.returncode}; run_id={run_id}; " + out.strip().replace("\n", " | ")[-120:])
    except Exception as e:
        record("air_submit", False, f"{type(e).__name__}: {e}")

    if run_id:
        from databricks.sdk import WorkspaceClient  # ships with the vendored CLI
        w = WorkspaceClient()
        deadline, state, result = time.time() + 900, "UNKNOWN", None
        while time.time() < deadline:
            rr = w.api_client.do("GET", f"/api/2.2/jobs/runs/get?run_id={run_id}")
            state = rr["state"]["life_cycle_state"]
            result = rr["state"].get("result_state")
            if state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
                break
            time.sleep(20)
        if result == "SUCCESS":
            record("air_submitted_run", True, f"run {run_id} -> SUCCESS")
        elif state in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            record("air_submitted_run", False, f"run {run_id} -> {state}/{result}")
        else:
            record("air_submitted_run", None, f"run {run_id} still {state} after 900s — check manually")
    else:
        record("air_submitted_run", False, "no run_id parsed from air output")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"check": "air-cli-from-notebook", "results": results}))
