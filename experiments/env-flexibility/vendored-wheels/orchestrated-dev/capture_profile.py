# Databricks notebook source
# MAGIC %md
# MAGIC # Capture a base-environment profile
# MAGIC
# MAGIC Run this notebook **on the target environment itself** (attach a serverless notebook set to
# MAGIC that environment version, or point `target_python` at its interpreter). It records the exact
# MAGIC `pip freeze` and the interpreter's wheel-compatibility metadata, then writes a drop-in profile
# MAGIC (`constraints.txt` + `target_env.json`) that `build_wheelhouse` resolves against.
# MAGIC
# MAGIC One capture per environment: `databricks_ai_v4`, `databricks_ai_v6`, `standard_v5`
# MAGIC (`databricks_ai_v5` already ships). Re-capture only when the managed baseline moves.
# MAGIC
# MAGIC - **AI runtimes** stack on a `base_environment` (`databricks_ai_vN`) — freeze the AI-env
# MAGIC   interpreter at `/opt/databricks-environments/databricks-ai/bin/python`.
# MAGIC - **standard v5** has no AI base — the profile sets `base_environment` to null so the emitted
# MAGIC   `environment.yaml` carries only `environment_version` + `dependencies`. Freeze the standard
# MAGIC   serverless interpreter (this notebook's own, blank `target_python`).

# COMMAND ----------
dbutils.widgets.dropdown(
    "profile_id", "databricks_ai_v6",
    ["databricks_ai_v4", "databricks_ai_v6", "standard_v5", "databricks_ai_v5"],
    "Profile to capture",
)
dbutils.widgets.text("environment_version", "", "environment_version (blank = infer from profile)")
dbutils.widgets.text("base_environment", "",
                     "base_environment to emit (blank = infer; 'none' for standard)")
dbutils.widgets.text("target_python", "",
                     "Interpreter to freeze (blank = this notebook's)")
dbutils.widgets.text("output_dir", "",
                     "UC Volume dir to write <profile_id>/ into (blank = profiles/ beside this notebook)")
dbutils.widgets.text("source_run_id", "", "AIR/job run id for provenance (optional)")

# COMMAND ----------
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

profile_id = dbutils.widgets.get("profile_id").strip()
environment_version = dbutils.widgets.get("environment_version").strip()
base_environment_widget = dbutils.widgets.get("base_environment").strip()
target_python = dbutils.widgets.get("target_python").strip() or sys.executable
output_dir = dbutils.widgets.get("output_dir").strip().rstrip("/")
source_run_id = dbutils.widgets.get("source_run_id").strip()

# environment_version: infer the trailing digits of the profile id when not given.
if not environment_version:
    match = re.search(r"(\d+)$", profile_id)
    assert match, f"cannot infer environment_version from {profile_id!r}; set it explicitly"
    environment_version = match.group(1)

# base_environment: AI runtimes emit their own name; standard emits none (null).
# 'none' (any case) or an explicit blank on a standard_* profile means: omit the line.
if base_environment_widget.lower() == "none" or profile_id.startswith("standard"):
    base_environment = None
elif base_environment_widget:
    base_environment = base_environment_widget
else:
    base_environment = profile_id  # databricks_ai_vN

# provenance
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
try:
    workspace = context.tags().apply("browserHostName") or context.workspaceId().get()
except Exception:
    workspace = "unknown"
captured_on = datetime.date.today().isoformat()

print(f"profile_id          {profile_id}")
print(f"target_python       {target_python}")
print(f"environment_version {environment_version}")
print(f"base_environment    {base_environment!r} (null => no base line in environment.yaml)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Freeze the environment and probe its wheel tags
# MAGIC
# MAGIC `constraints.txt` is the complete clean `pip freeze --all` (pins only). `target_env.json`
# MAGIC records the interpreter's Python/ABI/platform tags — the same fields the checked-in
# MAGIC `databricks_ai_v5` profile carries. `supported_tags` is intentionally not stored: the resolver
# MAGIC reconstructs it from `python_version`/`abis`/`platforms`.

# COMMAND ----------
raw = subprocess.run(
    [target_python, "-m", "pip", "freeze", "--all"],
    capture_output=True, text=True, check=True,
).stdout
pin_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^ @]+$")
pins = [line.strip() for line in raw.splitlines() if pin_pattern.match(line.strip())]
dropped = [line.strip() for line in raw.splitlines()
           if line.strip() and not pin_pattern.match(line.strip())]

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
    "platforms": list(dict.fromkeys(tag.platform for tag in tags if "linux" in tag.platform)),
    "marker_environment": default_environment(),
}))
'''
probe = json.loads(subprocess.run(
    [target_python, "-c", tag_probe], capture_output=True, text=True, check=True,
).stdout)

# Assemble the profile schema in the exact shape build_wheelhouse/resolve_worker read (matches
# the checked-in databricks_ai_v5 profile), with base_environment as the one new, optional field.
target_env = {
    "profile_schema": 1,
    "air_environment": profile_id,
    "base_environment": base_environment,
    "environment_version": environment_version,
    "source": {"workspace": workspace, "run_id": source_run_id, "captured_on": captured_on},
    "python_full": probe["python_full"],
    "python_version": probe["python_version"],
    "implementation": probe["implementation"],
    "abi": probe["abi"],
    "abis": probe["abis"],
    "top_tag": probe["top_tag"],
    "platforms": probe["platforms"],
    "marker_environment": probe["marker_environment"],
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write the profile

# COMMAND ----------
if output_dir:
    dest = Path(output_dir) / profile_id
else:
    notebook_path = context.notebookPath().get()
    notebook_dir = Path("/Workspace") / Path(notebook_path.lstrip("/")).parent
    dest = notebook_dir / "profiles" / profile_id
dest.mkdir(parents=True, exist_ok=True)

constraints_header = (
    f"# {profile_id} exact package baseline ({len(pins)} pins).\n"
    f"# Captured on {workspace}, {captured_on}"
    + (f", run {source_run_id}" if source_run_id else "")
    + ".\n"
)
(dest / "constraints.txt").write_text(constraints_header + "\n".join(pins) + "\n")
(dest / "target_env.json").write_text(json.dumps(target_env, indent=2, ensure_ascii=False) + "\n")

print(f"wrote {dest}/constraints.txt  ({len(pins)} pins; {len(dropped)} non-pin lines dropped)")
print(f"wrote {dest}/target_env.json  (top tag {target_env['top_tag']})")
if output_dir:
    print(f"\n➡ copy {dest} into the repo at "
          f"experiments/env-flexibility/vendored-wheels/orchestrated/profiles/{profile_id}/ and commit")
else:
    print("\n➡ commit the new profiles/ files to the repo")
