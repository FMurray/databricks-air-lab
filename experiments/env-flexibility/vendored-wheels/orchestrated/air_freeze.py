# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — AIR FREEZE (run on SERVERLESS / AIR env-v5)
# MAGIC
# MAGIC Phase 1 records the exact AIR interpreter environment and requested packages. The classic
# MAGIC worker resolves against that baseline, removes only dependency pins the requested graph
# MAGIC proves incompatible, and writes a request-specific wheelhouse build. Phase 2 verifies that exact
# MAGIC build offline on AIR.

# COMMAND ----------
dbutils.widgets.text("stage_dir", "/Volumes/<catalog>/<schema>/<vol>/vendor-stage",
                     "UC Volume stage dir shared with the classic workspace")
dbutils.widgets.text("requirements", "", "Target requirements, one per line")
dbutils.widgets.text("overrides", "",
                     "Optional explicit AIR package overrides, names only, one per line")
dbutils.widgets.text("target_python", "",
                     "Interpreter the workload uses (blank = this notebook's)")

# COMMAND ----------
import hashlib
import json
import os
import re
import subprocess
import sys

STAGE = dbutils.widgets.get("stage_dir").rstrip("/")
REQ_TEXT = dbutils.widgets.get("requirements").strip()
OVERRIDE_TEXT = dbutils.widgets.get("overrides").strip()
TARGET_PY = dbutils.widgets.get("target_python").strip() or sys.executable
assert REQ_TEXT, "requirements is empty — list the packages the workload needs"
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


print(f"stage {STAGE}\ntarget interpreter {TARGET_PY}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 1 — capture AIR and stage the request
# MAGIC
# MAGIC `constraints.txt` is the complete clean `pip freeze`. It starts as the compatibility
# MAGIC baseline; the worker writes a build-specific effective copy after removing only detected or
# MAGIC explicitly named conflicts. `supported_tags` is the authoritative compatibility list used
# MAGIC to check every downloaded wheel.

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
print(f"explicit overrides {OVERRIDE_TEXT.splitlines() if OVERRIDE_TEXT else 'none'}")
print(f"\n➡ Run mlr_download_driver on classic with stage_dir={STAGE}, then run Phase 2 here")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 2 — verify the classic build offline on AIR
# MAGIC
# MAGIC The manifest must match the current Phase 1 fingerprint. This prevents a prior request's
# MAGIC constraints or accumulated wheelhouse from being accepted accidentally.

# COMMAND ----------
MANIFEST_PATH = f"{STAGE}/manifest.json"
assert os.path.exists(MANIFEST_PATH), "manifest.json is missing — run the classic worker first"
manifest = json.load(open(MANIFEST_PATH))
current_request = request_fingerprint(STAGE)
assert manifest.get("request_id") == current_request, (
    f"stale manifest: current request {current_request}, manifest {manifest.get('request_id')}"
)
assert manifest.get("ok"), f"classic worker failed at {manifest.get('stage')}: {manifest.get('error')}"

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
