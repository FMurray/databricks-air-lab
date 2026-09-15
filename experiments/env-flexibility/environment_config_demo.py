# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "/Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/environment.yaml"
# environment_version = "5"
# dependencies = [
#   "/Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/cowsay-6.1-py3-none-any.whl",
#   "/Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/humanize-4.16.0-py3-none-any.whl",
# ]
# ///
# MAGIC %md
# MAGIC # Environment configuration on AIR — extending the AI base env (v5) with vendored wheels
# MAGIC
# MAGIC On serverless JPMC workspaces **cannot reach an internal PyPI mirror**. The recommended pattern is: **vendor the wheels into a UC Volume once, then install
# MAGIC the wheel files straight from that Volume.** This notebook walks that end to end and contrasts it with how you'd do the
# MAGIC same thing on MLR.
# MAGIC
# MAGIC ### The environment model: MLR vs AIR
# MAGIC
# MAGIC | | **MLR** (Machine Learning Runtime) | **AIR** (AI Runtime, serverless GPU) |
# MAGIC |---|---|---|
# MAGIC | Compute | long-lived cluster you provision & keep warm | **ephemeral, per-session** — nothing persists between attaches |
# MAGIC | Base image | `DBR x.y ML` pins Python + every library | versioned **AI base environment** (`v5` = the `databricks-ai` venv: torch + ML libs) |
# MAGIC | Add a library | cluster Libraries UI, init scripts, `%pip` on a running cluster | notebook **Environment panel** deps / `%pip`, declared once — *layers on top of the base* |
# MAGIC | Internal PyPI | init script points pip at the internal mirror | **serverless can't reach the mirror** → vendor wheels into a **UC Volume** |
# MAGIC | Reproducibility | "whatever the cluster had that day" + init scripts | the dependency spec **is** the environment; same spec → same env, and it ports to `air run` jobs |
# MAGIC
# MAGIC **Takeaway:** on MLR the *cluster* is the unit of environment. On AIR the *dependency spec* is — a small,
# MAGIC declarative thing you version alongside your code, resolved from Volumes instead of the network.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How it fits together
# MAGIC
# MAGIC ```
# MAGIC   AI base environment v5   ──►   your vendored wheels        ──►   session ready
# MAGIC   (databricks_ai venv:            (/Volumes/.../wheels/*.whl,        torch + ML libs
# MAGIC    torch, pyarrow, ML libs)        read from Volume)                 + your packages
# MAGIC ```
# MAGIC
# MAGIC You never rebuild the base — you **layer on top of it**. Two ways to layer, both shown below:
# MAGIC
# MAGIC 1. **`%pip install` (session-scoped)** — fast, interactive, gone on restart. Good for exploration.
# MAGIC 2. **Environment panel (declarative)** — the base env + dependency list travels with the notebook,
# MAGIC    survives restarts, and is the exact spec an `air run` job reuses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC Widgets default to the Volume staged for this session; override to point at your own wheel Volume.

# COMMAND ----------

dbutils.widgets.text("catalog", "forrest_serverless_stable_2_catalog", "catalog")
dbutils.widgets.text("schema", "air_lab", "schema")
dbutils.widgets.text("wheels_volume", "wheels", "wheels volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
WHEELS_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{dbutils.widgets.get('wheels_volume')}"
print("Wheel Volume:", WHEELS_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0 — Establish the baseline
# MAGIC Prove two things before we install anything: (a) we're really on the **AI v5** base (torch is present —
# MAGIC that's the `databricks_ai` venv), and (b) the package we need is **not** in it. On AIR you don't guess
# MAGIC what's in the base — you check, then layer the difference.

# COMMAND ----------

import sys
from importlib.metadata import version, PackageNotFoundError

print("Python:", sys.version.split()[0])
try:
    print("torch :", version("torch"), "  <- confirms the AI v5 (databricks_ai) venv")
except PackageNotFoundError:
    print("torch : NOT FOUND — this isn't the databricks_ai_v5 base env")

# The packages we intend to vendor — expect these to be absent in the clean base:
for pkg in ("cowsay", "humanize"):
    try:
        print(f"{pkg:9}:", version(pkg))
    except PackageNotFoundError:
        print(f"{pkg:9}: not in base AI v5  (this is what we'll add)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Where the vendored wheels live
# MAGIC A UC Volume is just governed object storage that serverless *can* reach (over FUSE) — unlike the
# MAGIC internal PyPI mirror. Everything is `py3-none-any` (pure-python, universal) so the same wheel works on any
# MAGIC Python/arch the base env ships.
# MAGIC
# MAGIC > **Staging (done off serverless, for reference):**
# MAGIC > ```bash
# MAGIC > # on a box that CAN reach the internal index:
# MAGIC > pip download --no-deps --dest ./wheels \
# MAGIC >   --index-url https://<your-internal-mirror>/simple  cowsay humanize
# MAGIC > databricks fs cp ./wheels/ "dbfs:/Volumes/<cat>/<schema>/wheels/" --recursive
# MAGIC > ```
# MAGIC > Drop `--no-deps` to pull the **full dependency closure** — for real internal packages you must
# MAGIC > vendor every transitive dependency, since serverless can't fetch them at install time.

# COMMAND ----------

import os
print(WHEELS_DIR, "\n")
for f in sorted(os.listdir(WHEELS_DIR)):
    print(" ", f, f"({os.path.getsize(os.path.join(WHEELS_DIR, f)):,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Install from the Volume (session-scoped)
# MAGIC The interactive path. We hand `pip` the wheel files in the Volume by their path — since a Volume path
# MAGIC is a real file on the node, pip installs those files directly and never has to look a package up. `%pip` in a serverless notebook installs into the
# MAGIC notebook-scoped env and **restarts the Python interpreter** when it finishes (that's why the baseline
# MAGIC imports above are cleared afterward — expected).
# MAGIC
# MAGIC Two equivalent forms:
# MAGIC - **Explicit wheel paths** (below) — most robust; pip installs the files you name.
# MAGIC - **`--no-index --find-links <dir>`** — let pip resolve names against the Volume when you have many
# MAGIC   wheels: `%pip install --no-index --find-links {WHEELS_DIR} cowsay humanize`

# COMMAND ----------

# MAGIC %pip install /Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/cowsay-6.1-py3-none-any.whl /Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/humanize-4.16.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Verify
# MAGIC The packages are now in the running session, resolved entirely from the Volume.

# COMMAND ----------

import cowsay, humanize
from importlib.metadata import version
from datetime import timedelta

print("cowsay  :", version("cowsay"))
print("humanize:", version("humanize"), "->", humanize.naturaldelta(timedelta(seconds=9042)))
cowsay.cow("Installed straight from a UC Volume.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — The durable way: the Environment panel
# MAGIC `%pip` is gone on restart. To make the environment **stick** and be **reproducible**, declare it once in
# MAGIC the notebook's **Environment** side panel (the ⚙️ / "Environment" button on the right):
# MAGIC
# MAGIC 1. **Base environment** → `AI v5` (this is the `databricks_ai_v5` venv — torch + ML libs), plus your accelerator.
# MAGIC 2. **Dependencies** → paste the wheel paths, one per line:
# MAGIC    ```
# MAGIC    /Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/cowsay-6.1-py3-none-any.whl
# MAGIC    /Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/humanize-4.16.0-py3-none-any.whl
# MAGIC    ```
# MAGIC 3. **Apply** → the session restarts with the base env **plus** these wheels, every time.
# MAGIC
# MAGIC The panel accepts the same syntax as a base-environment spec — explicit wheel paths, `--no-index`,
# MAGIC `--find-links <dir>`, `-r /Volumes/.../requirements.txt`, pinned names. It does **not** need PyPI as long
# MAGIC as every referenced wheel is in a Volume.
# MAGIC
# MAGIC | | `%pip install` (Step 2) | Environment panel (Step 4) |
# MAGIC |---|---|---|
# MAGIC | Scope | this running session only | every attach of this notebook |
# MAGIC | Survives restart | ❌ | ✅ |
# MAGIC | Reproducible / reviewable | ad hoc | it's a declared spec, versioned with the notebook |
# MAGIC | Ports to `air run` jobs | no | ✅ — same spec (Step 6) |
# MAGIC | Use for | exploration, one-offs | anything you hand off or schedule |
# MAGIC
# MAGIC > **MLR contrast:** this is the AIR analogue of cluster Libraries — but instead of mutating a
# MAGIC > long-lived cluster, you're describing a fresh env that's rebuilt identically on every session.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Turn it into a reusable custom base environment
# MAGIC The panel keeps the spec **inline** in this notebook (the `# /// script` header up top). To make this
# MAGIC exact base + wheels **selectable from any notebook** — a first-class entry in the **Base environment**
# MAGIC dropdown, next to Standard / ML / AI — save it as a **custom base environment**:
# MAGIC
# MAGIC - **From the UI (canonical):** Environment panel → kebab menu (**⋮**) → **Export environment**. Name the
# MAGIC   file and save it to a workspace folder or a UC Volume. Then, in this or any notebook: **Base
# MAGIC   environment** dropdown → **Custom** → browse to that file. The notebook records the choice as a path —
# MAGIC   e.g. `base_environment = "/Volumes/.../wheels/environment.yaml"` in the header (which is exactly what
# MAGIC   this notebook now points at).
# MAGIC - **From code (reproducible):** the cell below writes the same spec, discovering the wheels from the
# MAGIC   Volume so the file can't drift from what's staged — handy to regenerate in a job or check into git.
# MAGIC
# MAGIC It emits **two** files — one for the notebook dropdown, one for `air run`:
# MAGIC
# MAGIC | File | Selected as | how the base is keyed |
# MAGIC |---|---|---|
# MAGIC | `environment.yaml` | **Base environment → Custom** (any notebook) | `base_environment: databricks_ai_v5` + `environment_version: '5'` |
# MAGIC | `requirements.yaml` | `air run` job spec (Step 6) | `version: 'databricks_ai_v5'` |
# MAGIC
# MAGIC Same base env, same wheels — only the key names differ between the notebook surface and the `air` CLI.

# COMMAND ----------

# Self-contained: widgets survive the Step 2 %pip restart, Python vars don't — re-derive WHEELS_DIR.
import os

CATALOG = dbutils.widgets.get("catalog")
SCHEMA  = dbutils.widgets.get("schema")
WHEELS_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{dbutils.widgets.get('wheels_volume')}"

wheels = sorted(f for f in os.listdir(WHEELS_DIR) if f.endswith(".whl"))
deps = "\n".join(f"  - {WHEELS_DIR}/{w}" for w in wheels)

# (a) Serverless base-environment YAML — mirrors this notebook's inline header.
base_env_yaml = f"""base_environment: databricks_ai_v5
environment_version: '5'
dependencies:
{deps}
"""

# (b) air run dependency file — same wheels; the CLI keys the base as `version`.
air_req_yaml = f"""version: 'databricks_ai_v5'
dependencies:
{deps}
"""

for name, content in [("environment.yaml", base_env_yaml), ("requirements.yaml", air_req_yaml)]:
    path = f"{WHEELS_DIR}/{name}"
    with open(path, "w") as f:
        f.write(content)
    print(f"# ---- {path} ----")
    print(content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select it and confirm it carries both layers
# MAGIC Set **Base environment → Custom → `environment.yaml`** (your header already points here) and let the
# MAGIC session restart. Then re-run **Step 0** and **Step 3**: `torch` should be present — the AI v5 venv came
# MAGIC through — **and** `cowsay` / `humanize` import with no `%pip` cell. That's the proof a custom base env
# MAGIC delivers the base **and** your wheels in a single selection. If `torch` is missing, regenerate the file
# MAGIC with **Export environment** (it captures the AI venv exactly) and re-point Custom at it.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Same environment for a training job (`air run`)
# MAGIC The whole point of a declarative spec: the env you validated interactively is the env your multi-node
# MAGIC job runs. Step 5 wrote `requirements.yaml` to the Volume — reference it from your `air` job spec and you
# MAGIC get identical base + identical vendored wheels, no PyPI at job launch.

# COMMAND ----------

# MAGIC %md
# MAGIC **`requirements.yaml`** (the AIR job dependency spec):
# MAGIC ```yaml
# MAGIC version: 'databricks_ai_v5'          # the AI v5 base: torch + ML venv
# MAGIC dependencies:
# MAGIC   - /Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/cowsay-6.1-py3-none-any.whl
# MAGIC   - /Volumes/forrest_serverless_stable_2_catalog/air_lab/wheels/humanize-4.16.0-py3-none-any.whl
# MAGIC ```
# MAGIC
# MAGIC **`training.yaml`** references it, and you launch:
# MAGIC ```yaml
# MAGIC environment:
# MAGIC   dependencies: requirements.yaml     # a path (shown) OR an inline list — same format
# MAGIC ```
# MAGIC ```bash
# MAGIC air run --file ./training.yaml
# MAGIC ```
# MAGIC
# MAGIC The `version: 'databricks_ai_v5'` prefix is what loads the torch/ML venv (requires version ≥ 5); your
# MAGIC wheels install on top of it — exactly like the Environment panel above. Run `air config.environment -h`
# MAGIC for the full field reference.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap — environment management, MLR → AIR
# MAGIC
# MAGIC | You want to… | MLR | AIR |
# MAGIC |---|---|---|
# MAGIC | Pick a base | choose a `DBR ML` version | choose an **AI base env version** (`v5` = torch/ML venv) |
# MAGIC | Add a package interactively | `%pip` on the cluster | `%pip install /Volumes/.../x.whl` (Step 2) |
# MAGIC | Make it stick | cluster Libraries / init script | **Environment panel** dependency spec (Step 4) |
# MAGIC | Use an internal package | pip → internal mirror via init script | **vendor the wheel into a UC Volume**, install it by path |
# MAGIC | Same env in a job | clone the cluster config | reuse the **dependency spec** in `air run` (Step 6) |
# MAGIC
# MAGIC **One idea to leave with:** on AIR the environment is *data you declare*, not a *machine you maintain*.
# MAGIC A Volume of wheels + a short dependency spec is the whole story — reproducible, reviewable, and
# MAGIC identical from notebook to job, with no dependency on reaching PyPI.