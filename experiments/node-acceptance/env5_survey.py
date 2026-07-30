"""Env v5 survey (Gen-AI task path): what GPU/python stack does the image provide?

v5 ships job-plane logs (unlike v4), so stdout is the receipt channel here; MLflow
receipt is attempted only if mlflow imports. Informs how gpu-burn gets a GPU stack
on v5 (vendor vs ctypes-on-image-libs).
"""
import ctypes.util
import glob
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys

print(f"SURVEY python={sys.version.split()[0]} exe={sys.executable}", flush=True)

for mod in ["torch", "numpy", "mlflow", "pynvml", "pandas", "pyarrow"]:
    try:
        m = importlib.import_module(mod)
        print(f"SURVEY import {mod}=OK version={getattr(m, '__version__', '?')}", flush=True)
    except Exception as e:
        print(f"SURVEY import {mod}=MISSING ({type(e).__name__})", flush=True)

nsmi = shutil.which("nvidia-smi")
print(f"SURVEY nvidia-smi={nsmi}", flush=True)
if nsmi:
    out = subprocess.run([nsmi, "-L"], capture_output=True, text=True, timeout=60)
    print(f"SURVEY nvidia-smi -L: {out.stdout.strip() or out.stderr.strip()}", flush=True)

for lib in ["cuda", "cudart", "cublas", "nvidia-ml", "nccl"]:
    print(f"SURVEY ctypes find_library({lib})={ctypes.util.find_library(lib)}", flush=True)

for so in ["libnccl.so.2", "libnccl.so"]:
    try:
        lib = ctypes.CDLL(so)
        ver = ctypes.c_int()
        lib.ncclGetVersion(ctypes.byref(ver))
        print(f"SURVEY dlopen {so}=OK ncclGetVersion={ver.value}", flush=True)
        break
    except OSError as e:
        print(f"SURVEY dlopen {so}=FAILED ({e})", flush=True)

for pat in ["/usr/local/cuda*/lib64/libcublas*", "/usr/local/cuda*/lib64/libcudart*",
            "/usr/lib/x86_64-linux-gnu/libcuda*", "/usr/lib/x86_64-linux-gnu/libcublas*",
            "/usr/lib/x86_64-linux-gnu/libnccl*", "/usr/local/cuda*/lib64/libnccl*",
            "/opt/**/libnccl*", "/usr/local/lib/libnccl*"]:
    hits = glob.glob(pat)
    print(f"SURVEY glob {pat} -> {hits[:4]}", flush=True)

dists = sorted({d.metadata["Name"] for d in importlib.metadata.distributions() if d.metadata["Name"]})
print(f"SURVEY n_dists={len(dists)}", flush=True)
for i in range(0, len(dists), 20):
    print("SURVEY dists:", ",".join(dists[i:i + 20]), flush=True)

# torch hunt: other interpreters / site-packages on the image (the launched interpreter
# lacking torch does NOT prove the image does — challenge from review, settle it here)
for pat in ["/databricks/*/bin/python*", "/opt/*/bin/python*", "/usr/bin/python3*",
            "/opt/conda/envs/*/bin/python*", "/local_disk0/*/bin/python*"]:
    hits = sorted(set(glob.glob(pat)))
    print(f"SURVEY pythons {pat} -> {hits[:6]}", flush=True)
for pat in ["/databricks/**/site-packages/torch/version.py",
            "/opt/**/site-packages/torch/version.py",
            "/usr/lib/python3*/dist-packages/torch/version.py",
            "/local_disk0/**/torch/version.py"]:
    hits = glob.glob(pat, recursive=True)
    print(f"SURVEY torch-hunt {pat} -> {hits[:4]}", flush=True)
r = subprocess.run(["find", "/databricks", "/opt", "-maxdepth", "6", "-name", "torch",
                    "-type", "d"], capture_output=True, text=True, timeout=120)
print(f"SURVEY find-torch-dirs: {r.stdout.strip().splitlines()[:6]}", flush=True)

# RDMA counter exposure (receipt channel for fabric stress)
ib = glob.glob("/sys/class/infiniband/*")
print(f"SURVEY infiniband devices: {ib[:8]}", flush=True)
ctr = glob.glob("/sys/class/infiniband/*/ports/*/hw_counters/*")
print(f"SURVEY hw_counters n={len(ctr)} sample={[c.rsplit('/',1)[-1] for c in ctr[:12]]}", flush=True)

print(f"SURVEY MLFLOW_RUN_ID={'set' if os.environ.get('MLFLOW_RUN_ID') else 'unset'}", flush=True)
try:
    from mlflow.tracking import MlflowClient
    rid = os.environ.get("MLFLOW_RUN_ID")
    if rid:
        MlflowClient().log_param(rid, "env5_survey", "SEE_LOGS_v5_ships_them")
        print("SURVEY mlflow receipt=OK", flush=True)
except Exception as e:
    print(f"SURVEY mlflow receipt=FAILED {e}", flush=True)

print("ENV5_SURVEY_DONE", flush=True)
