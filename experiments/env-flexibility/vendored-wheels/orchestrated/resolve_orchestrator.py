# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — ORCHESTRATOR
# MAGIC
# MAGIC Run this notebook on AIR. It captures the AIR baseline and supported wheel tags, submits the
# MAGIC canonical resolver on a classic cluster with Artifactory access, then verifies the resulting
# MAGIC request-specific wheelhouse offline on the same AIR interpreter.

# COMMAND ----------
dbutils.widgets.text("stage_dir", "/Volumes/<catalog>/<schema>/<vol>/vendor-stage",
                     "UC Volume stage dir shared with classic")
dbutils.widgets.text("requirements", "", "Target requirements, one per line")
dbutils.widgets.text("overrides", "", "Optional explicit AIR package overrides, names only")
dbutils.widgets.text("classic_cluster_id", "", "Classic cluster id with Artifactory access")
dbutils.widgets.text("worker_notebook_path", "", "Workspace path to resolve_worker")
dbutils.widgets.text("target_python", "", "Interpreter the workload uses (blank = this notebook's)")
dbutils.widgets.text("index_url", "", "Artifactory index URL (blank = classic pip.conf)")
dbutils.widgets.text("wait_minutes", "45", "Maximum minutes to wait for the classic run")

# COMMAND ----------
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import timedelta

STAGE = dbutils.widgets.get("stage_dir").rstrip("/")
REQ_TEXT = dbutils.widgets.get("requirements").strip()
OVERRIDE_TEXT = dbutils.widgets.get("overrides").strip()
CLASSIC_ID = dbutils.widgets.get("classic_cluster_id").strip()
WORKER_NB = dbutils.widgets.get("worker_notebook_path").strip()
TARGET_PY = dbutils.widgets.get("target_python").strip() or sys.executable
INDEX_URL = dbutils.widgets.get("index_url").strip()
WAIT_MIN = int(dbutils.widgets.get("wait_minutes") or "45")

assert REQ_TEXT, "requirements is empty — list the packages the workload needs"
assert CLASSIC_ID, "classic_cluster_id is required"
assert WORKER_NB, "worker_notebook_path is required"
os.makedirs(STAGE, exist_ok=True)


def request_fingerprint(stage):
    digest = hashlib.sha256()
    for name in ("constraints.txt", "requirements.txt", "overrides.txt", "target_env.json"):
        path = f"{stage}/{name}"
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(open(path, "rb").read() if os.path.exists(path) else b"")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


print(f"stage {STAGE}\ntarget interpreter {TARGET_PY}\n"
      f"index {'explicit Artifactory URL' if INDEX_URL else 'classic cluster pip.conf'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — capture the AIR baseline and supported tags

# COMMAND ----------
raw = subprocess.run(
    [TARGET_PY, "-m", "pip", "freeze", "--all"],
    capture_output=True, text=True, check=True,
).stdout
pin_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^ @]+$")
constraints = [line.strip() for line in raw.splitlines() if pin_pattern.match(line.strip())]
dropped = [line.strip() for line in raw.splitlines()
           if line.strip() and not pin_pattern.match(line.strip())]

with open(f"{STAGE}/constraints.txt", "w") as handle:
    handle.write("\n".join(constraints) + "\n")
with open(f"{STAGE}/requirements.txt", "w") as handle:
    handle.write(REQ_TEXT + "\n")
with open(f"{STAGE}/overrides.txt", "w") as handle:
    handle.write(OVERRIDE_TEXT + ("\n" if OVERRIDE_TEXT else ""))

tag_probe = r'''
import json
import sys
from packaging.markers import default_environment
from packaging.tags import sys_tags

tags = list(sys_tags())
print(json.dumps({
    "python_full": sys.version.split()[0],
    "python_version": f"{sys.version_info.major}{sys.version_info.minor}",
    "implementation": tags[0].interpreter[:2],
    "abi": tags[0].abi,
    "abis": list(dict.fromkeys(tag.abi for tag in tags)),
    "top_tag": str(tags[0]),
    "platforms": list(dict.fromkeys(tag.platform for tag in tags if "x86_64" in tag.platform)),
    "supported_tags": [str(tag) for tag in tags],
    "marker_environment": default_environment(),
}))
'''
target_env = json.loads(subprocess.run(
    [TARGET_PY, "-c", tag_probe], capture_output=True, text=True, check=True,
).stdout)
with open(f"{STAGE}/target_env.json", "w") as handle:
    json.dump(target_env, handle, indent=2)

REQUEST_ID = request_fingerprint(STAGE)
print(f"constraints kept {len(constraints)} | non-pin lines dropped {len(dropped)}")
print(f"request id {REQUEST_ID} | AIR top tag {target_env['top_tag']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — submit the canonical resolver on classic

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

workspace = WorkspaceClient()
waiter = workspace.jobs.submit(
    run_name=f"vendor-resolve {REQUEST_ID}",
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
print(f"submitted classic run {run_id}: {workspace.config.host}/jobs/runs/{run_id}")
run = waiter.result(timeout=timedelta(minutes=WAIT_MIN))
assert str(run.state.result_state) == "RunResultState.SUCCESS", (
    f"classic worker did not complete: {run.state.result_state} / {run.state.state_message}"
)

task_run_id = run.tasks[0].run_id
output = workspace.jobs.get_run_output(run_id=task_run_id)
assert output.notebook_output and output.notebook_output.result, "classic worker returned no manifest"
manifest = json.loads(output.notebook_output.result)
print(json.dumps({key: value for key, value in manifest.items() if key != "wheel_files"},
                 indent=2, ensure_ascii=False))
assert manifest.get("request_id") == REQUEST_ID, "classic worker returned a stale request"
assert manifest.get("ok"), f"classic worker failed at {manifest.get('stage')}: {manifest.get('error')}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — verify the exact build offline on AIR

# COMMAND ----------
wheelhouse = manifest["wheelhouse"]
effective = manifest["constraints_for_install"]
delta_lock = manifest["delta_lock"]

resolve_verify = subprocess.run(
    [TARGET_PY, "-m", "pip", "install", "--dry-run", "--no-index",
     "--find-links", wheelhouse, "-r", f"{STAGE}/requirements.txt", "-c", effective],
    capture_output=True, text=True,
)
hash_verify = subprocess.run(
    [TARGET_PY, "-m", "pip", "install", "--dry-run", "--no-index", "--no-deps",
     "--require-hashes", "--find-links", wheelhouse, "-r", delta_lock],
    capture_output=True, text=True,
)
print(resolve_verify.stdout)
print(resolve_verify.stderr)
print(hash_verify.stdout)
print(hash_verify.stderr)

seamless = resolve_verify.returncode == 0 and hash_verify.returncode == 0
print("=" * 60)
print("VERDICT:", "PASS — exact vendored delta resolves offline and hashes match"
      if seamless else "FAIL — see the offline resolution/hash output above")
print("=" * 60)
print("Install line for the AIR workload command:")
print(f"  {TARGET_PY} -m pip install --no-index --no-deps --require-hashes \\\n"
      f"    --find-links {wheelhouse} -r {delta_lock}")
print(f"  {TARGET_PY} -m pip check")
assert seamless, "offline wheelhouse verification failed"

dbutils.notebook.exit(json.dumps({
    "seamless": seamless,
    "request_id": REQUEST_ID,
    "manifest": manifest,
}, ensure_ascii=False))
