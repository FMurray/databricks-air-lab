# Databricks notebook source
# MAGIC %md
# MAGIC # UAT check: GPU smoke (notebook surface)
# MAGIC **Env required**: Environment panel → Base environment **AI**, Accelerator **A10**
# MAGIC (H100 for the acceptance pass) — see `../environment.yml`. On a non-GPU env this
# MAGIC check reports `skipped_no_gpu` rather than failing, so the DRIVER can run everywhere.
# MAGIC Verifies: CUDA visible, device identity, a measured bf16 matmul (sanity TFLOPS),
# MAGIC and NVML health if available.

# COMMAND ----------

import json, time

results = {}

def record(name, ok, detail=""):
    results[name] = {"ok": bool(ok) if ok is not None else None, "detail": str(detail)[:200]}
    print({True: "✅", False: "❌", None: "⏭️"}[ok], name, "—", str(detail)[:200])

# COMMAND ----------

try:
    import torch
    has_gpu = torch.cuda.is_available()
except ImportError:
    torch, has_gpu = None, False

if torch is None or not has_gpu:
    record("gpu_available", None, "skipped_no_gpu — attach serverless GPU (AI env + Accelerator) to run this check")
    dbutils.notebook.exit(json.dumps({"check": "gpu-smoke", "results": results}))

# COMMAND ----------

n = torch.cuda.device_count()
record("gpu_available", True, f"{n}x {torch.cuda.get_device_name(0)}")

# COMMAND ----------

# shape assertion — DRIVER passes expect_gpus per shape (e.g. 8 for GPU_8xH100)
try:
    expect = dbutils.widgets.get("expect_gpus").strip()
except Exception:
    expect = ""
if expect:
    record("gpu_count_matches_shape", n == int(expect), f"saw {n}, expected {expect}")

# COMMAND ----------

# measured matmul throughput — sanity bar only (A10 bf16 ≳ 20 TFLOPS effective)
N = 8192
a = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
b = torch.randn(N, N, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    _ = a @ b
torch.cuda.synchronize()
t0 = time.time()
iters = 20
for _ in range(iters):
    _ = a @ b
torch.cuda.synchronize()
tflops = 2 * N**3 * iters / (time.time() - t0) / 1e12
record("matmul_tflops", tflops > 5, f"{tflops:.1f} TFLOPS bf16 {N}x{N}")

# COMMAND ----------

try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    record("nvml_health", True,
           f"temp={pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)}C "
           f"power={pynvml.nvmlDeviceGetPowerUsage(h)/1000:.0f}W")
except Exception as e:
    record("nvml_health", None, f"skipped: {type(e).__name__}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"check": "gpu-smoke", "results": results}))
