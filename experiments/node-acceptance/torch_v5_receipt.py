"""Is torch usable on env v5 (Gen-AI task path)? Receipt-driven successor to the inline
torch-v5-probe command: this workspace's GPU-plane log blackout eats stdout (run
401340759545364 FAILED with zero retrievable evidence), so every finding lands as an
MLflow param the moment it's known. Default v5 python has mlflow (survey 491958602140255)
but NO torch; survey 96419244890099 found torch files under the image's databricks-ai env
— this probe proves (or refutes) that the interpreter actually runs torch + CUDA.

Sentinel: param probe_sentinel=TORCH_V5_RECEIPT_DONE (reachable even if every check fails
— the receipts, not exit status, carry the verdict).
"""
import json
import os
import subprocess
import sys

import mlflow

AI_ENV = "/opt/databricks-environments/databricks-ai"
AIPY = f"{AI_ENV}/bin/python"

mlflow.start_run(run_name="torch-v5-receipt")
log = mlflow.log_param

def sh(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()

log("default_python", f"{sys.version.split()[0]} {sys.executable}")
try:
    import torch  # noqa: F401 — expected MISSING on v5 default env
    log("torch_default_env", f"PRESENT {torch.__version__}")
except Exception as e:
    log("torch_default_env", f"MISSING ({type(e).__name__})")

envs = os.listdir("/opt/databricks-environments") if os.path.isdir("/opt/databricks-environments") else []
log("image_environments", json.dumps(envs)[:490])
log("ai_env_python_exists", str(os.path.exists(AIPY)))

if os.path.exists(AIPY):
    rc, out = sh([AIPY, "-c",
                  "import torch; print(torch.__version__, '|cuda_available', "
                  "torch.cuda.is_available(), '|devices', torch.cuda.device_count())"])
    log("torch_via_ai_env", f"rc={rc} {out[-400:]}")
    if rc == 0:
        # the claim that matters: a real CUDA op through the ai-env interpreter
        rc2, out2 = sh([AIPY, "-c",
                        "import torch; x = torch.randn(1024, 1024, device='cuda'); "
                        "print('matmul_ok', float((x @ x).abs().sum()) > 0, "
                        "torch.cuda.get_device_name(0))"], timeout=300)
        log("torch_cuda_matmul", f"rc={rc2} {out2[-400:]}")
    log("torchrun_in_ai_env", str(os.path.exists(f"{AI_ENV}/bin/torchrun")))

# PYTHONPATH injection into the DEFAULT interpreter (the tempting shortcut — record
# whether it works or exactly how it breaks, e.g. compiled-ext/env mismatch)
sp = f"{AI_ENV}/lib/python3.12/site-packages"
if os.path.isdir(sp):
    rc, out = sh([sys.executable, "-c", "import torch; print(torch.__version__)"],
                 timeout=120)  # control: no injection
    env = dict(os.environ, PYTHONPATH=sp)
    r = subprocess.run([sys.executable, "-c",
                        "import torch; print(torch.__version__, torch.cuda.is_available())"],
                       capture_output=True, text=True, timeout=300, env=env)
    log("torch_via_pythonpath", f"rc={r.returncode} {(r.stdout + r.stderr).strip()[-400:]}")
else:
    log("torch_via_pythonpath", f"SKIPPED — no {sp}")

log("probe_sentinel", "TORCH_V5_RECEIPT_DONE")
mlflow.end_run()
print("TORCH_V5_RECEIPT_DONE (see MLflow params — stdout may never be delivered here)")
