# Databricks notebook source
# MAGIC %md
# MAGIC # Vendored-wheel resolver — WORKER (%run helper)
# MAGIC
# MAGIC Runs on a **CLASSIC cluster that can reach the index** (Python matching the AIR target —
# MAGIC cp312 today). It is `%run` from `mlr_download_driver`; it can also be launched as a Jobs
# MAGIC notebook_task (see `resolve_orchestrator`). Dual-mode:
# MAGIC   - **`%run` mode** — the driver sets `stage_dir` / `index_url` as variables first; this
# MAGIC     notebook leaves a `manifest` dict in the namespace and does NOT call
# MAGIC     `dbutils.notebook.exit` (that would terminate the driver).
# MAGIC   - **job mode** — reads widgets and calls `dbutils.notebook.exit(manifest)`.
# MAGIC
# MAGIC What it does, given the stage dir the AIR freeze populated:
# MAGIC   1. Resolve `requirements.txt` bounded by `constraints.txt` (the frozen target env). If the
# MAGIC      requirements need a DIFFERENT version of a baseline package, **reconcile**: resolve the
# MAGIC      requirements unconstrained to see what they actually want, then unpin the clashing
# MAGIC      baseline pins so the resolve can succeed. Pure-Python packages and self-contained
# MAGIC      compiled extensions (jiter, pydantic-core, …) are safe to override; ABI-foundational
# MAGIC      packages (numpy/torch/… — things other native wheels are built against) are blocked
# MAGIC      unless `allow_compiled_override` is set. The reduced set is written to
# MAGIC      `constraints.effective.txt` and is what the runtime install must use.
# MAGIC   2. **Version-AWARE delta**: reuse a baked-in package ONLY when the resolved version matches
# MAGIC      what the target already has. Anything new — OR resolved to a different version (an
# MAGIC      override) — goes in the delta and is fetched. The wants side wins; a stale baked-in copy
# MAGIC      no longer masks a needed version (this is why an override like jiter was silently skipped
# MAGIC      before — the delta was computed by name, not version).
# MAGIC   3. `pip download --no-deps` each delta package as AIR-targeted wheels; anything unfetchable
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

# Overriding these can break OTHER native packages / torch that were compiled against their ABI, so
# never auto-override them. Everything else compiled (jiter, pydantic-core, pycryptodomex, awscrt,
# rapidfuzz, …) is a self-contained extension — safe to vendor/override if an AIR-tagged wheel exists.
_ABI_FOUNDATIONAL = {"numpy", "scipy", "pandas", "pyarrow", "torch", "torchvision", "torchaudio",
                     "triton"}
def _foundational(n):
    n = canon(n)
    return n in _ABI_FOUNDATIONAL or n.startswith("nvidia-")

def _pure_wheel(dl):  # a pure-Python wheel is py*-none-any (no ABI/platform tag)
    return os.path.basename((dl or {}).get("url", "")).endswith("-none-any.whl")

def _load_pins(path):
    pins = {}
    for ln in open(path):
        s = ln.strip()
        if "==" in s and not s.startswith("#"):
            n, v = s.split("==", 1); pins[canon(n)] = v.strip()
    return pins

def _pip_resolve(req_path, con_path, report_path, extra):
    """--dry-run resolve to a --report. con_path='' means resolve UNconstrained."""
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
           "--report", report_path, "--only-binary=:all:", "-r", req_path]
    if con_path:
        cmd += ["-c", con_path]
    return subprocess.run(cmd + extra, capture_output=True, text=True)


def resolve_and_download(stage, index_url, allow_compiled_override=False):
    stage = stage.rstrip("/")
    wheelhouse = f"{stage}/wheelhouse"
    local_wh = "/local_disk0/vendor_wheelhouse"   # pip writes many small files; stage locally then copy
    os.makedirs(wheelhouse, exist_ok=True)
    shutil.rmtree(local_wh, ignore_errors=True); os.makedirs(local_wh)

    req_path = f"{stage}/requirements.txt"
    con_path = f"{stage}/constraints.txt"
    eff_path = f"{stage}/constraints.effective.txt"   # baseline minus overrides; used at runtime too
    target_env = json.load(open(f"{stage}/target_env.json"))
    extra = ["--index-url", index_url] if index_url else []
    # Explicit AIR-target wheel selection — do NOT trust host == target. From the freeze's
    # target_env.json, so downloads fetch wheels for AIR's interpreter / ABI / platform regardless of
    # the classic host. `--platform` is repeatable and matches EXACTLY, so we pass every x86_64
    # manylinux baseline AIR accepts. `--platform` requires `--only-binary=:all:`, already set.
    tgt = ["--implementation", "cp", "--python-version", target_env["python_version"],
           "--abi", target_env["abi"]]
    for plat in target_env.get("platforms", []):
        tgt += ["--platform", plat]

    here_pv = f"{sys.version_info.major}{sys.version_info.minor}"
    py_matched = here_pv == target_env["python_version"]
    print(f"✓ python match: cp{here_pv}; downloads targeted to {target_env['top_tag']}" if py_matched
          else f"⚠️  cluster python cp{here_pv} != target cp{target_env['python_version']}; downloads "
               f"are pinned to AIR's tags but the RESOLVE runs here — prefer a matching runtime.")

    baseline = _load_pins(con_path)

    # 1 — resolve bounded by the FULL baseline. If it fails, RECONCILE: find what the requirements
    #     actually want (unconstrained) and unpin the baseline pins they can't coexist with.
    rpt = "/local_disk0/resolve_report.json"
    resolve = _pip_resolve(req_path, con_path, rpt, extra)
    con_used, overrides, blocked = con_path, {}, {}
    if resolve.returncode != 0:
        unc = _pip_resolve(req_path, "", "/local_disk0/unconstrained.json", extra)
        if unc.returncode != 0:
            print(unc.stdout[-3000:]); print(unc.stderr[-4000:])
            return {"stage": "resolve", "ok": False, "python_matched": py_matched,
                    "error": "the requirements don't resolve even WITHOUT the baseline — a genuine "
                             "incompatibility inside the requested packages (see stderr).",
                    "stderr_tail": unc.stderr[-2000:]}
        want = {canon(e["metadata"]["name"]): (e["metadata"]["version"], _pure_wheel(e.get("download_info")))
                for e in json.load(open("/local_disk0/unconstrained.json")).get("install", [])}
        for n, (v, pure) in want.items():
            if n in baseline and baseline[n] != v:                 # a clash the baseline would block
                (overrides if (pure or not _foundational(n)) else blocked)[n] = (baseline[n], v)
        if blocked and not allow_compiled_override:
            print("✗ requirements need different versions of ABI-foundational baseline packages:")
            for n, (bv, wv) in sorted(blocked.items()):
                print(f"    {n}: baseline {bv} -> wants {wv}")
            return {"stage": "reconcile", "ok": False, "python_matched": py_matched,
                    "error": "would override ABI-foundational packages (numpy/torch/…), which can "
                             "break the base env's native stack. Pin the requirement to the baseline "
                             "version, or set allow_compiled_override=true to force it.",
                    "blocked": {n: {"baseline": bv, "wanted": wv} for n, (bv, wv) in blocked.items()}}
        if allow_compiled_override:
            overrides.update(blocked)
        drop = set(overrides)
        with open(eff_path, "w") as f:
            for ln in open(con_path):
                is_pin = "==" in ln and not ln.lstrip().startswith("#")
                nm = canon(ln.split("==", 1)[0].strip()) if is_pin else None
                if nm not in drop:
                    f.write(ln if ln.endswith("\n") else ln + "\n")
        print(f"reconcile: unpinned {len(drop)} baseline pin(s) so the requirements can resolve:")
        for n, (bv, wv) in sorted(overrides.items()):
            print(f"  ↑ {n}: baseline {bv} -> vendoring {wv}")
        resolve = _pip_resolve(req_path, eff_path, rpt, extra)
        con_used = eff_path

    print(resolve.stdout[-3000:])
    if resolve.returncode != 0:
        print(resolve.stderr[-4000:])
        return {"stage": "resolve", "ok": False, "python_matched": py_matched,
                "error": "resolution failed after reconcile — see stderr.",
                "stderr_tail": resolve.stderr[-2000:]}

    closure = {canon(p["metadata"]["name"]): p["metadata"]["version"]
               for p in json.load(open(rpt)).get("install", [])}

    # 2 — VERSION-AWARE delta: reuse a baked-in package only when the resolved version MATCHES what
    #     the target already has. New packages AND overrides (resolved != baked-in) go in the delta.
    delta = {n: v for n, v in closure.items() if baseline.get(n) != v}
    print(f"closure {len(closure)} | reused from target {len(closure) - len(delta)} | delta {len(delta)}")
    for n, v in sorted(delta.items()):
        print(f"  {n}=={v}" + (f"   (override: target has {baseline[n]})" if n in baseline else "   (new)"))

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
         "overrides": {n: {"baseline": baseline[n], "vendored": v}
                       for n, v in delta.items() if n in baseline},
         "downloaded": downloaded, "failures": failures,
         "wheelhouse": wheelhouse, "wheel_files": wheel_files,
         "constraints_for_install": con_used, "python_matched": py_matched}
    with open(f"{stage}/manifest.json", "w") as f:
        json.dump(m, f, indent=2)
    if con_used != con_path:
        print(f"\n⚠️  overrode baseline packages — the AIR runtime install MUST use {con_used}\n"
              f"    (not constraints.txt), or the old pins will re-conflict.")
    return m

# COMMAND ----------
STAGE          = _cfg("stage_dir")
INDEX_URL      = _cfg("index_url")
ALLOW_COMPILED = str(_cfg("allow_compiled_override", "")).strip().lower() in ("1", "true", "yes")
assert STAGE, "stage_dir is required (set it in the driver before %run, or as a widget)"

manifest = resolve_and_download(STAGE, INDEX_URL, ALLOW_COMPILED)
print(json.dumps({k: v for k, v in manifest.items() if k != "wheel_files"}, indent=2, ensure_ascii=False))

if _JOB_MODE:
    dbutils.notebook.exit(json.dumps(manifest, ensure_ascii=False))
# else (%run): the driver reads `manifest` from the namespace
