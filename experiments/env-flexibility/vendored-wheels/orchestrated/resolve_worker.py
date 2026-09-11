# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — WORKER (%run helper)
# MAGIC
# MAGIC Runs on a **CLASSIC cluster that can reach Artifactory** (Python matching the AIR target —
# MAGIC cp312 today). It is `%run` from `mlr_download_driver`; it can also be launched as a Jobs
# MAGIC notebook_task (see `resolve_orchestrator`). Dual-mode:
# MAGIC   - **`%run` mode** — the driver sets `stage_dir` / `index_url` as variables first; this
# MAGIC     notebook leaves a `manifest` dict in the namespace and does NOT call
# MAGIC     `dbutils.notebook.exit` (that would terminate the driver).
# MAGIC   - **job mode** — reads widgets and calls `dbutils.notebook.exit(manifest)`.
# MAGIC
# MAGIC What it does, given the stage dir the AIR freeze populated:
# MAGIC   1. Resolve the FULL closure of `requirements.txt`, **bounded by `constraints.txt`** (the
# MAGIC      serverless env's frozen installed set) — a real incompatibility surfaces here, not at
# MAGIC      runtime on AIR.
# MAGIC   2. Subtract the target's installed set → the **delta**. Baked-in packages (incl. any with
# MAGIC      no Artifactory wheel) are in the constraints, so never in the delta, never downloaded.
# MAGIC   3. `pip download --no-deps` each delta package into the wheelhouse; anything unfetchable
# MAGIC      is a REAL gap and is reported.

# COMMAND ----------
import json, os, re, shutil, subprocess, sys, glob

def _cfg(name, default=""):
    """Prefer a variable set by a %run driver; fall back to a widget; else default."""
    if name in globals() and globals()[name]:
        return globals()[name]
    try:
        v = dbutils.widgets.get(name)
        if v:
            return v
    except Exception:
        pass
    return default

_JOB_MODE = not (("stage_dir" in globals()) and globals().get("stage_dir"))

def canon(name):  # PEP 503 normalization
    return re.sub(r"[-_.]+", "-", name).lower()


def resolve_and_download(stage, index_url):
    stage = stage.rstrip("/")
    wheelhouse = f"{stage}/wheelhouse"
    local_wh = "/local_disk0/vendor_wheelhouse"   # pip writes many small files; stage locally then copy
    os.makedirs(wheelhouse, exist_ok=True)
    shutil.rmtree(local_wh, ignore_errors=True); os.makedirs(local_wh)

    req_path = f"{stage}/requirements.txt"
    con_path = f"{stage}/constraints.txt"
    target_env = json.load(open(f"{stage}/target_env.json"))
    extra = ["--index-url", index_url] if index_url else []
    # Explicit AIR-target wheel selection — do NOT trust host == target. These come straight from
    # the freeze's target_env.json, so the download fetches wheels for AIR's interpreter / ABI /
    # platform regardless of what the classic host runs. `--platform` is repeatable and matches
    # EXACTLY, so we pass every x86_64 manylinux baseline AIR accepts (whichever a package published
    # under, we catch it). `--platform` requires `--only-binary=:all:`, already set.
    tgt = ["--implementation", "cp",
           "--python-version", target_env["python_version"],
           "--abi", target_env["abi"]]
    for plat in target_env.get("platforms", []):
        tgt += ["--platform", plat]

    here_pv = f"{sys.version_info.major}{sys.version_info.minor}"
    py_matched = here_pv == target_env["python_version"]
    if not py_matched:
        print(f"⚠️  cluster python cp{here_pv} != target cp{target_env['python_version']}. "
              f"Downloads are pinned to AIR's tags, but the RESOLVE step runs on cp{here_pv} and "
              f"could pick a version that differs across Python versions — prefer a matching runtime.")
    else:
        print(f"✓ python match: cp{here_pv}; downloads targeted to {target_env['top_tag']}")

    # 1 — resolve full closure, bounded by the target's constraints
    report_path = "/local_disk0/resolve_report.json"
    resolve = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
         "--report", report_path, "--only-binary=:all:",
         "-r", req_path, "-c", con_path] + extra,
        capture_output=True, text=True,
    )
    print(resolve.stdout[-4000:])
    if resolve.returncode != 0:
        print(resolve.stderr[-4000:])
        return {"stage": "resolve", "ok": False,
                "error": "resolution failed — a new requirement likely conflicts with the AIR "
                         "baseline (see stderr). Pin a compatible version in requirements.txt.",
                "stderr_tail": resolve.stderr[-2000:], "python_matched": py_matched}

    report  = json.load(open(report_path))
    closure = {canon(p["metadata"]["name"]): p["metadata"]["version"]
               for p in report.get("install", [])}

    # 2 — subtract the target's installed set → the delta
    target_installed = {canon(ln.split("==", 1)[0]) for ln in open(con_path) if "==" in ln}
    delta = {n: v for n, v in closure.items() if n not in target_installed}
    print(f"closure {len(closure)} | already on target {len(closure) - len(delta)} | delta {len(delta)}")
    for n, v in sorted(delta.items()):
        print(f"  {n}=={v}")

    # 3 — download only the delta (pinned, --no-deps, wheels targeted to AIR's tags), then copy
    downloaded, failures = [], []
    for n, v in sorted(delta.items()):
        spec = f"{n}=={v}"
        r = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
             "-d", local_wh, spec] + tgt + extra,
            capture_output=True, text=True,
        )
        (downloaded if r.returncode == 0 else failures).append(
            spec if r.returncode == 0 else {"spec": spec, "stderr_tail": r.stderr[-600:]})
        if r.returncode != 0:
            print(f"✗ {spec}\n{r.stderr[-600:]}")

    for whl in glob.glob(f"{local_wh}/*"):
        shutil.copy(whl, wheelhouse)
    wheel_files = sorted(os.path.basename(p) for p in glob.glob(f"{wheelhouse}/*"))

    m = {"ok": len(failures) == 0, "closure_count": len(closure),
         "delta": sorted(f"{n}=={v}" for n, v in delta.items()),
         "downloaded": downloaded, "failures": failures,
         "wheelhouse": wheelhouse, "wheel_files": wheel_files, "python_matched": py_matched}
    with open(f"{stage}/manifest.json", "w") as f:
        json.dump(m, f, indent=2)
    return m

# COMMAND ----------
STAGE     = _cfg("stage_dir")
INDEX_URL = _cfg("index_url")
assert STAGE, "stage_dir is required (set it in the driver before %run, or as a widget)"

manifest = resolve_and_download(STAGE, INDEX_URL)
print(json.dumps({k: v for k, v in manifest.items() if k != "wheel_files"}, indent=2, ensure_ascii=False))

if _JOB_MODE:
    dbutils.notebook.exit(json.dumps(manifest, ensure_ascii=False))
# else (%run): the driver reads `manifest` from the namespace
