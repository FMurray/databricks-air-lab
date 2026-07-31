# Databricks notebook source
# MAGIC %md
# MAGIC # UAT check: package inventory of the baked-in databricks-ai env
# MAGIC Run as a GPU notebook job on env v5 (or interactively). Answers which ML packages the
# MAGIC `/opt/databricks-environments/databricks-ai` environment provides — decides whether a
# MAGIC workload needs vendored wheels or just the AI-env sys.path line.

# COMMAND ----------

import json, os, subprocess, sys

AI_SP = "/opt/databricks-environments/databricks-ai/lib/python3.12/site-packages"
PKGS = ["torch", "xgboost", "sklearn", "transformers", "lightgbm", "peft", "datasets", "mlflow", "pandas", "numpy"]

results = {"ai_env_present": str(os.path.isdir(AI_SP))}
if os.path.isdir(AI_SP):
    # subprocess per package: isolates imports; uses the AI env's own interpreter
    aipy = "/opt/databricks-environments/databricks-ai/bin/python"
    runner = aipy if os.path.exists(aipy) else sys.executable
    for p in PKGS:
        code = f"import {p}; print(getattr({p}, '__version__', 'present'))"
        env = dict(os.environ)
        if runner is not aipy:
            env["PYTHONPATH"] = AI_SP
        r = subprocess.run([runner, "-c", code], capture_output=True, text=True, timeout=120, env=env)
        results[p] = r.stdout.strip() if r.returncode == 0 else "MISSING"
for k, v in results.items():
    print(f"RESULT ai_env.{k} = {v}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results))
