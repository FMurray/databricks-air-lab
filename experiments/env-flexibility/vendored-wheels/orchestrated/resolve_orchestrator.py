# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — ORCHESTRATOR
# MAGIC
# MAGIC **Run this on the SERVERLESS / AIR env-v5 notebook** (the environment your workload
# MAGIC actually runs in — the one that CANNOT reach Artifactory).
# MAGIC
# MAGIC The problem this solves: a plain `pip download -r requirements.txt` on the classic side
# MAGIC resolves the **entire** dependency closure — including packages already baked into the
# MAGIC AI v5 image (torch, CUDA libs, numpy, …). If any of those has no wheel on Artifactory,
# MAGIC `--only-binary=:all:` fails with "no matching distribution" even though the target already
# MAGIC has it. The classic resolver has no idea what serverless already provides.
# MAGIC
# MAGIC This orchestrator makes resolution **seamless across the two environments**:
# MAGIC   1. Snapshot THIS env's installed set with `pip freeze` → a constraints file.
# MAGIC   2. Hand `requirements.txt` + constraints to a classic notebook (via Jobs `submit`),
# MAGIC      which resolves the full closure bounded by the constraints and downloads **only the
# MAGIC      delta** (what serverless does NOT already have) into a shared UC Volume.
# MAGIC   3. Prove it: offline `--dry-run` install here. Green = the wheelhouse + this env resolve
# MAGIC      with zero index access.
# MAGIC
# MAGIC `dbutils.notebook.run` runs on the caller's own context, so it cannot target a classic
# MAGIC cluster — the cross-environment hop is a one-time Jobs run submitted with the SDK.

# COMMAND ----------
dbutils.widgets.text("stage_dir", "/Volumes/<catalog>/<schema>/<vol>/vendor-stage",
                     "UC Volume stage dir (shared with classic)")
dbutils.widgets.text("requirements", "",
                     "packages to ADD that AREN'T already in the base env, one per line "
                     "(pinning a package the base env already ships to a different version WILL conflict)")
dbutils.widgets.text("classic_cluster_id", "", "Classic cluster id (Artifactory egress)")
dbutils.widgets.text("worker_notebook_path", "", "Workspace path to resolve_worker")
dbutils.widgets.text("target_python", "", "Interpreter wheels install into (blank = this notebook's)")
dbutils.widgets.text("index_url", "", "Artifactory index URL for the worker (blank = classic pip.conf)")
dbutils.widgets.text("wait_minutes", "45", "Max minutes to wait on the classic run")

# COMMAND ----------
import json, os, re, subprocess, sys
from datetime import timedelta

STAGE          = dbutils.widgets.get("stage_dir").rstrip("/")
WHEELHOUSE     = f"{STAGE}/wheelhouse"
REQ_TEXT       = dbutils.widgets.get("requirements").strip()
CLASSIC_ID     = dbutils.widgets.get("classic_cluster_id").strip()
WORKER_NB      = dbutils.widgets.get("worker_notebook_path").strip()
TARGET_PY      = dbutils.widgets.get("target_python").strip() or sys.executable
INDEX_URL      = dbutils.widgets.get("index_url").strip()
WAIT_MIN       = int(dbutils.widgets.get("wait_minutes") or "45")

assert REQ_TEXT,   ("requirements is empty — list the packages to vendor (those NOT already in the "
                    "base env), one per line")
assert CLASSIC_ID, "classic_cluster_id is required (a running classic cluster with Artifactory egress)"
assert WORKER_NB,  "worker_notebook_path is required (workspace path to resolve_worker)"

os.makedirs(STAGE, exist_ok=True)
os.makedirs(WHEELHOUSE, exist_ok=True)
print(f"stage      : {STAGE}")
print(f"wheelhouse : {WHEELHOUSE}")
print(f"target py  : {TARGET_PY}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Snapshot the target env → constraints.txt
# MAGIC `pip freeze` on the interpreter the wheels will install into. We keep only clean
# MAGIC `name==version` lines: editable installs, VCS/`@ file://` direct references, and comments
# MAGIC cannot serve as constraints and would break the resolve.

# COMMAND ----------
raw_freeze = subprocess.run([TARGET_PY, "-m", "pip", "freeze", "--all"],
                            capture_output=True, text=True, check=True).stdout
_pin = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^ @]+$")
constraints = [ln.strip() for ln in raw_freeze.splitlines() if _pin.match(ln.strip())]
dropped     = [ln.strip() for ln in raw_freeze.splitlines()
               if ln.strip() and not _pin.match(ln.strip())]

with open(f"{STAGE}/constraints.txt", "w") as f:
    f.write("\n".join(constraints) + "\n")
with open(f"{STAGE}/requirements.txt", "w") as f:
    f.write(REQ_TEXT + "\n")

print(f"constraints kept   : {len(constraints)} pinned packages")
print(f"non-pin lines dropped: {len(dropped)}"
      + (f" (e.g. {dropped[:3]})" if dropped else ""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Record the target's wheel tags (for the worker to assert against)

# COMMAND ----------
tag_probe = (
    "import json,sys;"
    "from packaging.tags import sys_tags;"
    "ts=list(sys_tags());"
    "print(json.dumps({"
    "'python_full':sys.version.split()[0],"
    "'python_version':f'{sys.version_info.major}{sys.version_info.minor}',"
    "'abi':ts[0].abi,"
    "'top_tag':str(ts[0]),"
    "'platforms':list(dict.fromkeys(t.platform for t in ts if 'x86_64' in t.platform))"
    "}))"
)
target_env = json.loads(
    subprocess.run([TARGET_PY, "-c", tag_probe], capture_output=True, text=True, check=True).stdout
)
with open(f"{STAGE}/target_env.json", "w") as f:
    json.dump(target_env, f, indent=2)
print(json.dumps(target_env, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Submit the classic worker (Jobs run) and wait
# MAGIC The worker reads `constraints.txt` / `requirements.txt` / `target_env.json` from the stage
# MAGIC dir, resolves the delta with Artifactory access, and populates `wheelhouse/`.

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()
run_name = f"vendor-resolve {os.path.basename(STAGE)}"
waiter = w.jobs.submit(
    run_name=run_name,
    tasks=[jobs.SubmitTask(
        task_key="resolve_delta",
        existing_cluster_id=CLASSIC_ID,
        notebook_task=jobs.NotebookTask(
            notebook_path=WORKER_NB,
            base_parameters={"stage_dir": STAGE, "index_url": INDEX_URL},
        ),
    )],
)
run_id = waiter.run_id
print(f"submitted classic run {run_id}: {w.config.host}/jobs/runs/{run_id}")
run = waiter.result(timeout=timedelta(minutes=WAIT_MIN))
print(f"classic run state: {run.state.result_state}")

# surface the worker's dbutils.notebook.exit(...) payload
task_run_id = run.tasks[0].run_id
out = w.jobs.get_run_output(run_id=task_run_id)
worker_summary = {}
if out.notebook_output and out.notebook_output.result:
    worker_summary = json.loads(out.notebook_output.result)
    print(json.dumps(worker_summary, indent=2))
else:
    print("no notebook_output; check the run URL above for logs")
assert str(run.state.result_state) == "RunResultState.SUCCESS", \
    f"classic worker did not succeed: {run.state.result_state} / {run.state.state_message}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Prove seamless resolution (offline --dry-run install on THIS env)
# MAGIC No index. If pip resolves `requirements.txt` from the wheelhouse + what's already installed
# MAGIC here, with exit 0, the vendored set is complete and conflict-free for the target.

# COMMAND ----------
verify = subprocess.run(
    [TARGET_PY, "-m", "pip", "install", "--dry-run", "--no-index",
     "--find-links", WHEELHOUSE, "-r", f"{STAGE}/requirements.txt",
     "-c", f"{STAGE}/constraints.txt"],
    capture_output=True, text=True,
)
print(verify.stdout)
print(verify.stderr)
seamless = verify.returncode == 0
print("=" * 60)
print(f"VERDICT: {'PASS — vendored set resolves offline against the target env' if seamless else 'FAIL — see stderr above'}")
print("=" * 60)
print("Install line for your workload YAML command:")
print(f'  pip install --no-index --find-links {WHEELHOUSE} \\\n'
      f'    -r {STAGE}/requirements.txt -c {STAGE}/constraints.txt')

dbutils.notebook.exit(json.dumps({
    "seamless": seamless,
    "wheelhouse": WHEELHOUSE,
    "worker_summary": worker_summary,
}))
