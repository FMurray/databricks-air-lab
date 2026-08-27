"""Env topology diagnostic — is serverless_gpu available on env v5, and where (with torch)?

Answers the open question left by the UCVolumeDataset probe: `serverless_gpu` was absent from the
databricks-ai venv (which has torch), while env5_survey found a `databricks.serverless_gpu`
DISTRIBUTION in the root python (which had no torch). This maps EVERY interpreter on the container
for {torch, serverless_gpu, databricks.serverless_gpu, ...} + finds the package on disk + its
distribution name, so we learn (a) whether any single interpreter has BOTH torch and the
serverless-GPU data API, and (b) if not, the pip/dist name to try installing.

Pure stdlib (no torch, no serverless_gpu import at module level) so it cannot hang on the thing it
tests. Prints ENVPROBE / DISKPROBE marker lines; run via the ROOT python (plain `python3`), which
then subprocess-probes the other interpreters too, so one run covers the whole container.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

# Snippet each candidate interpreter runs (via -c). Reports importability + module file for the
# modules we care about, plus any installed distribution whose name mentions serverless/torch.
CHECK = r'''
import importlib, sys, json
out = {"exe": sys.executable, "py": sys.version.split()[0]}
for m in ["torch", "serverless_gpu", "serverless_gpu.data",
          "databricks", "databricks.serverless_gpu", "databricks.serverless_gpu.data"]:
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__file__", None) or "namespace-pkg"
    except Exception as e:
        out[m] = f"FAIL:{type(e).__name__}"
try:
    import importlib.metadata as md
    out["dists"] = sorted(d.metadata["Name"] for d in md.distributions()
                          if any(k in (d.metadata["Name"] or "").lower()
                                 for k in ("serverless", "torch")))
except Exception as e:
    out["dists"] = f"FAIL:{e}"
print("ENVPROBE " + json.dumps(out), flush=True)
'''


def discover_interpreters() -> list[str]:
    found = []
    for pat in ("/opt/databricks-environments/*/bin/python*",
                "/databricks/python3/bin/python*",
                "/databricks/*/bin/python*",
                "/usr/bin/python3*"):
        found += glob.glob(pat)
    for name in ("python3", "python"):
        w = shutil.which(name)
        if w:
            found.append(w)
    found.append(sys.executable)
    # de-dup by realpath, keep only files
    uniq = {}
    for p in found:
        try:
            rp = os.path.realpath(p)
            if os.path.isfile(rp):
                uniq[rp] = p
        except Exception:                                  # noqa: BLE001
            pass
    return sorted(uniq)


def disk_search() -> None:
    """Glob site-packages across the image for the serverless_gpu package directory + its dist-info,
    so we see where it physically lives even if no interpreter imports it by default."""
    hits = []
    for pat in ("/opt/databricks-environments/*/lib/python*/site-packages/serverless_gpu",
                "/opt/databricks-environments/*/lib/python*/site-packages/databricks/serverless_gpu",
                "/databricks/*/lib/python*/site-packages/serverless_gpu",
                "/databricks/*/lib/python*/site-packages/databricks/serverless_gpu",
                "/usr/lib/python3*/*-packages/serverless_gpu",
                "/opt/databricks-environments/*/lib/python*/site-packages/*serverless_gpu*.dist-info",
                "/databricks/*/lib/python*/site-packages/*serverless_gpu*.dist-info"):
        hits += glob.glob(pat)
    print("DISKPROBE serverless_gpu_on_disk=" + json.dumps(sorted(set(hits))), flush=True)


def main() -> int:
    print(f"ENVPROBE_LAUNCHER exe={sys.executable} py={sys.version.split()[0]}", flush=True)
    interps = discover_interpreters()
    print(f"ENVPROBE_INTERPRETERS n={len(interps)} {json.dumps(interps)}", flush=True)
    for py in interps:
        try:
            r = subprocess.run([py, "-c", CHECK], capture_output=True, text=True, timeout=60)
            line = (r.stdout or "").strip() or f"(no stdout) stderr={r.stderr.strip()[:200]}"
            print(f"[{py}] {line}", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"[{py}] SUBPROCESS_FAIL {type(e).__name__}: {e}", flush=True)
    disk_search()
    print("ENVPROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
