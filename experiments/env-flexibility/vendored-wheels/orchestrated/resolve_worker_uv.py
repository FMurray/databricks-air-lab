# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — WORKER (uv variant, %run helper)
# MAGIC
# MAGIC Same contract as `resolve_worker` (dual-mode `%run`/job; reads the stage dir the AIR freeze
# MAGIC populated; leaves a `manifest`), but resolves the closure with **`uv pip compile`** instead
# MAGIC of `pip install --dry-run --report`.
# MAGIC
# MAGIC Why a uv variant: `uv pip compile` cross-resolves for the target's `--python-version` /
# MAGIC `--python-platform` from metadata, so **the classic host no longer needs to match AIR's
# MAGIC Python** (the pip variant resolves natively and only warns on mismatch). It's also faster.
# MAGIC
# MAGIC Honest caveat: **uv has no `download` subcommand**, so it cannot build a wheelhouse. This
# MAGIC worker therefore uses uv only for the *resolve*; the delta *fetch* still uses
# MAGIC `pip download` (targeted to AIR's tags). AIR-side consumption is unchanged
# MAGIC (`pip install --no-index --find-links`). A fully-uv, pip-free alternative is
# MAGIC `uv pip install --target <dir> --no-deps …` (unpacked + `PYTHONPATH`, like `vendor_deps.sh`).

# COMMAND ----------
import json, os, re, shutil, subprocess, sys, glob

def _cfg(name, default=""):
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

def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def _ensure_uv():
    """uv is usually absent on the base cluster/AIR image — bootstrap it via pip."""
    uv = shutil.which("uv")
    if uv:
        return uv
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], check=True)
    return shutil.which("uv") or "uv"

def _uv_platform(top_tag):
    """AIR top_tag 'cp312-cp312-manylinux_2_39_x86_64' -> uv --python-platform 'x86_64-manylinux_2_39'."""
    m = re.search(r"manylinux_(\d+)_(\d+)_x86_64", top_tag)
    if m:
        return f"x86_64-manylinux_{m.group(1)}_{m.group(2)}"
    if "manylinux2014" in top_tag:
        return "x86_64-manylinux2014"
    return "x86_64-manylinux2014"  # safe, broadly-compatible fallback


def resolve_and_download(stage, index_url):
    stage = stage.rstrip("/")
    wheelhouse = f"{stage}/wheelhouse"
    local_wh = "/local_disk0/vendor_wheelhouse"
    os.makedirs(wheelhouse, exist_ok=True)
    shutil.rmtree(local_wh, ignore_errors=True); os.makedirs(local_wh)

    req_path = f"{stage}/requirements.txt"
    con_path = f"{stage}/constraints.txt"
    target_env = json.load(open(f"{stage}/target_env.json"))
    extra = ["--index-url", index_url] if index_url else []

    pv_dotted = f"{target_env['python_version'][0]}.{target_env['python_version'][1:]}"  # 312 -> 3.12
    uv_plat = _uv_platform(target_env["top_tag"])
    print(f"resolving for AIR: python {pv_dotted}, platform {uv_plat} (host cp"
          f"{sys.version_info.major}{sys.version_info.minor} — need NOT match with uv)")

    # 1 — resolve the full closure with uv, cross-targeted to AIR, bounded by AIR's constraints
    uv = _ensure_uv()
    resolved = "/local_disk0/resolved.txt"
    r = subprocess.run(
        [uv, "pip", "compile", req_path, "-c", con_path,
         "--python-version", pv_dotted, "--python-platform", uv_plat,
         "--only-binary", ":all:", "--no-header", "--no-annotate",
         "-o", resolved] + extra,
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:]); print(r.stderr[-3000:])
    if r.returncode != 0:
        return {"engine": "uv", "stage": "resolve", "ok": False,
                "error": "uv pip compile failed — a new requirement likely conflicts with the AIR "
                         "baseline (see stderr). Pin a compatible version in requirements.txt.",
                "stderr_tail": r.stderr[-2000:]}

    # parse the pinned closure: 'name==version' lines (drop comments / options / env-markers)
    closure = {}
    for ln in open(resolved):
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("-"):
            continue
        ln = ln.split(";", 1)[0].strip()          # drop environment markers
        if "==" in ln:
            n, v = ln.split("==", 1)
            closure[canon(n.strip())] = v.strip()

    # 2 — subtract the target's installed set → the delta
    target_installed = {canon(l.split("==", 1)[0]) for l in open(con_path) if "==" in l}
    delta = {n: v for n, v in closure.items() if n not in target_installed}
    print(f"closure {len(closure)} | already on target {len(closure) - len(delta)} | delta {len(delta)}")
    for n, v in sorted(delta.items()):
        print(f"  {n}=={v}")

    # 3 — fetch the delta as wheels, targeted to AIR's tags (uv has no download → use pip download)
    tgt = ["--implementation", "cp",
           "--python-version", target_env["python_version"], "--abi", target_env["abi"]]
    for plat in target_env.get("platforms", []):
        tgt += ["--platform", plat]
    downloaded, failures = [], []
    for n, v in sorted(delta.items()):
        spec = f"{n}=={v}"
        d = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
             "-d", local_wh, spec] + tgt + extra,
            capture_output=True, text=True,
        )
        (downloaded if d.returncode == 0 else failures).append(
            spec if d.returncode == 0 else {"spec": spec, "stderr_tail": d.stderr[-600:]})
        if d.returncode != 0:
            print(f"✗ {spec}\n{d.stderr[-600:]}")

    for whl in glob.glob(f"{local_wh}/*"):
        shutil.copy(whl, wheelhouse)
    wheel_files = sorted(os.path.basename(p) for p in glob.glob(f"{wheelhouse}/*"))

    m = {"engine": "uv", "ok": len(failures) == 0, "closure_count": len(closure),
         "delta": sorted(f"{n}=={v}" for n, v in delta.items()),
         "downloaded": downloaded, "failures": failures,
         "wheelhouse": wheelhouse, "wheel_files": wheel_files}
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
