# Install Python packages from a UC Volume (no PyPI)

For workspaces where PyPI is unreachable (by policy). You stage packages in a Unity Catalog
Volume **once**, and every workload uses them with **one line of YAML changed**. No Docker,
no admin settings, no internet from the compute.

> Verification status: the mechanics (unpacked packages + `PYTHONPATH`) are verified
> end-to-end via the code-snapshot variant (run 846540776169482, 2026-07-24, emoji + a
> compiled extension importing on an A10). The volume path is identical plumbing —
> re-verify with step 4 on your workspace and record the run id here.

## What you need

- A laptop with [uv](https://docs.astral.sh/uv/) installed (any OS — it cross-builds for the
  GPU nodes), or access to your internal package mirror.
- Write access to a UC Volume, e.g. `/Volumes/<catalog>/<schema>/deps`.
- The Databricks CLI authenticated to your workspace (`databricks auth login`).

## Step 1 — vendor the packages (laptop, one time per package set)

```bash
mkdir -p vendor
uv pip install --target vendor \
  --python-platform x86_64-unknown-linux-gnu \
  --python-version 3.12 \
  --only-binary :all: \
  xgboost scikit-learn        # ← your packages here
```

That's it: `vendor/` now contains the packages **unpacked**, built for the GPU nodes
(linux x86_64, CPython 3.12 — the AIR env's interpreter). No pip will run on the node.

## Step 2 — upload to the volume (one time per package set)

```bash
databricks fs cp -r vendor dbfs:/Volumes/<catalog>/<schema>/deps/vendor -p <profile>
```

## Step 3 — use it in any workload (the one line)

In your workload YAML, keep `dependencies: []` and prefix your command:

```yaml
environment:
  version: "5"
  dependencies: []      # nothing from PyPI — that's the point

command: |
  export PYTHONPATH="/Volumes/<catalog>/<schema>/deps/vendor:$PYTHONPATH"
  python $CODE_SOURCE_PATH/your_training_script.py
```

Your script does `import xgboost` and it just works.

## Step 4 — verify (30 seconds of GPU time)

```bash
air run --file workloads/vendored-wheels-ucvolume.example.yaml -p <profile>
```

Green run + `vendored_verdict = PASS` in the run's MLflow params = the pattern works on
your workspace. (The example installs one pure-Python and one compiled package — if the
compiled one imports, your platform/python targeting is right.)

## When something fails

| Symptom | Cause → fix |
|---|---|
| `ModuleNotFoundError` at run time | `PYTHONPATH` line missing/typo'd, or the volume path is wrong — `ls /Volumes/...` from a notebook to confirm |
| `ImportError: ...so: cannot open` or version-tag errors on ONE package | compiled package vendored for the wrong python/platform — re-run step 1 with `--python-version` matching the node (3.12 today; check with `python -V` in a probe run) |
| Volume unreadable from the run | volume/catalog permissions, or (hardened workspaces) the catalog's storage isn't reachable from serverless — ask your admin |
| Package needs torch | don't vendor torch — env v5 ships it at `/opt/databricks-environments/databricks-ai/bin/python`; run your script with that interpreter instead |

## Two alternatives, for completeness

- **Small package sets (≤ ~50 MB):** skip the volume — put `vendor/` next to your code and
  ship it in the code snapshot (`PYTHONPATH="$CODE_SOURCE_PATH/.../vendor"`). Verified:
  `workloads/vendored-wheels-snapshot.example.yaml`. One trap: the snapshot excludes
  gitignored paths, so don't gitignore `vendor/`.
- **The zero-effort endgame (needs a workspace admin, once):** point the workspace at your
  internal mirror via **Settings → Compute → Default Package Repositories** (secret scope
  `databricks-package-management`). Then plain `dependencies: [xgboost]` works everywhere
  and this whole page becomes unnecessary.
