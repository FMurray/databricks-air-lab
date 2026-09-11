# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — AIR FREEZE (run on SERVERLESS / AIR env-v5)
# MAGIC
# MAGIC Step 1 of the two-step flow. This notebook runs in the environment your workload actually
# MAGIC uses (the one that CANNOT reach Artifactory) and captures its installed set so the classic
# MAGIC side can resolve **against** it and download only the delta.
# MAGIC
# MAGIC **Order of operations**
# MAGIC   1. Run **Phase 1** here → writes `constraints.txt` / `requirements.txt` / `target_env.json`
# MAGIC      to the shared UC Volume stage dir.
# MAGIC   2. Run `mlr_download_driver` on your **classic 17.3 cluster** (Artifactory egress) → it
# MAGIC      populates `wheelhouse/` in the same stage dir.
# MAGIC   3. Come back and run **Phase 2** here → offline `--dry-run` proves the vendored set
# MAGIC      resolves against this env with no index.
# MAGIC
# MAGIC The stage dir must be a UC Volume reachable from BOTH workspaces (same metastore/catalog).
# MAGIC If the classic cluster lives in a separate workspace without shared UC, stage there instead
# MAGIC and copy the wheelhouse back with `databricks fs cp`.

# COMMAND ----------
dbutils.widgets.text("stage_dir", "/Volumes/<catalog>/<schema>/<vol>/vendor-stage",
                     "UC Volume stage dir (shared with the classic workspace)")
dbutils.widgets.text("requirements", "",
                     "packages to ADD that AREN'T already in the base env, one per line "
                     "(pinning a package the base env already ships to a different version WILL conflict)")
dbutils.widgets.text("target_python", "",
                     "Interpreter wheels install into (blank = this notebook's)")

# COMMAND ----------
import json, os, re, subprocess, sys

STAGE      = dbutils.widgets.get("stage_dir").rstrip("/")
WHEELHOUSE = f"{STAGE}/wheelhouse"
REQ_TEXT   = dbutils.widgets.get("requirements").strip()
TARGET_PY  = dbutils.widgets.get("target_python").strip() or sys.executable
assert REQ_TEXT, ("requirements is empty — list the packages to vendor (those NOT already in the "
                  "base env), one per line, e.g. via the widget or a staged requirements.txt")
os.makedirs(STAGE, exist_ok=True); os.makedirs(WHEELHOUSE, exist_ok=True)
print(f"stage {STAGE}\ntarget py {TARGET_PY}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 1 — snapshot this env and stage the inputs (run FIRST)
# MAGIC Keep only clean `name==version` lines: editable installs, VCS/`@ file://` direct references,
# MAGIC and comments cannot serve as constraints and would break the classic resolve.

# COMMAND ----------
raw = subprocess.run([TARGET_PY, "-m", "pip", "freeze", "--all"],
                     capture_output=True, text=True, check=True).stdout
_pin = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^ @]+$")
constraints = [ln.strip() for ln in raw.splitlines() if _pin.match(ln.strip())]
dropped     = [ln.strip() for ln in raw.splitlines() if ln.strip() and not _pin.match(ln.strip())]

with open(f"{STAGE}/constraints.txt", "w") as f:
    f.write("\n".join(constraints) + "\n")
with open(f"{STAGE}/requirements.txt", "w") as f:
    f.write(REQ_TEXT + "\n")

tag_probe = (
    "import json,sys;from packaging.tags import sys_tags;ts=list(sys_tags());"
    "print(json.dumps({'python_full':sys.version.split()[0],"
    "'python_version':f'{sys.version_info.major}{sys.version_info.minor}',"
    "'abi':ts[0].abi,'top_tag':str(ts[0]),"
    "'platforms':list(dict.fromkeys(t.platform for t in ts if 'x86_64' in t.platform))}))"
)
target_env = json.loads(subprocess.run([TARGET_PY, "-c", tag_probe],
                                        capture_output=True, text=True, check=True).stdout)
with open(f"{STAGE}/target_env.json", "w") as f:
    json.dump(target_env, f, indent=2)

print(f"constraints kept {len(constraints)} | non-pin lines dropped {len(dropped)}")
print(json.dumps(target_env, indent=2))
print(f"\n➡  Now run mlr_download_driver on your classic 17.3 cluster with stage_dir={STAGE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Phase 2 — prove seamless resolution (run AFTER the classic download completes)
# MAGIC No index. Exit 0 = the wheelhouse + what's already installed here fully satisfy the
# MAGIC requirements for this env.

# COMMAND ----------
verify = subprocess.run(
    [TARGET_PY, "-m", "pip", "install", "--dry-run", "--no-index",
     "--find-links", WHEELHOUSE, "-r", f"{STAGE}/requirements.txt",
     "-c", f"{STAGE}/constraints.txt"],
    capture_output=True, text=True,
)
print(verify.stdout); print(verify.stderr)
seamless = verify.returncode == 0
print("=" * 60)
print("VERDICT:", "PASS — vendored set resolves offline against the target env"
      if seamless else "FAIL — see stderr above")
print("=" * 60)
print("Install line for your workload YAML command:")
print(f"  pip install --no-index --find-links {WHEELHOUSE} \\\n"
      f"    -r {STAGE}/requirements.txt -c {STAGE}/constraints.txt")
