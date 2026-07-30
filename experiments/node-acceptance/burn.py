"""GPU burn + health check (UAT A1). Single process, all GPUs concurrently.

Two load backends, same receipt schema:
- torch (env v4): round-robin async fp16 matmuls per device (see git history for the mp
  deadlock that forced single-process, run 603656373172623).
- ctypes+cuBLAS (env v5, which has NO torch on the Gen-AI task path but ships the full
  CUDA 12.9 toolkit — survey run 491958602140255): cublasGemmEx fp16 in/fp32 accumulate,
  one handle per device, async launches on each device's default stream.

Health checks via pynvml (native on v5; vendored fallback for v4-without-pip).
Receipts via MLflow client API params/metrics (v4 has the log blackout; docs/06).

Env knobs: BURN_SECONDS (default 120), EXPECT_GPUS (assert count if set), MATMUL_N (default 8192).
Exit 0 only if: all expected GPUs enumerate, uncorrected ECC delta is 0 on every GPU,
and no HW_SLOWDOWN throttle reason was observed under load.
"""
import ctypes
import ctypes.util
import os
import signal
import sys
import time

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:  # PyPI is by-design unavailable on hardened workspaces — use vendored copy
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    try:
        import pynvml
    except ImportError:
        pynvml = None

BURN_SECONDS = int(os.environ.get("BURN_SECONDS", "120"))
MATMUL_N = int(os.environ.get("MATMUL_N", "8192"))
# nvmlClocksThrottleReasons: HW_SLOWDOWN | HW_THERMAL | HW_POWER_BRAKE
HW_THROTTLE_MASK = 0x8 | 0x40 | 0x80


def log_receipt(params=None, metrics=None):
    """MLflow receipt — the durable channel where run logs never land (docs/06-uat-suite.md).
    alarm() so a blocked tracking call fails the receipt, not the burn."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        # client API, not start_run(): resuming transitions the launcher-owned run's status
        # and failed silently on the job plane (run 1021313559310836 — start receipt landed,
        # end receipt didn't)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        for k, v in (params or {}).items():
            client.log_param(run_id, k, v)
        for k, v in (metrics or {}).items():
            client.log_metric(run_id, k, v)
    except Exception as e:
        print(f"receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def gpu_uuid(idx):
    """Stable hardware identity — hostnames are identical across AIR nodes, UUIDs are not."""
    if pynvml is None:
        return "nvml-missing"
    u = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(idx))
    return u.decode() if isinstance(u, bytes) else u


def gpu_name(idx):
    if torch is not None:
        return torch.cuda.get_device_name(idx)
    n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(idx))
    return n.decode() if isinstance(n, bytes) else n


def nvml_snapshot(idx):
    if pynvml is None:
        return {}
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    snap = {
        "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
        "throttle": pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h),
    }
    try:
        snap["ecc_uncorrected"] = pynvml.nvmlDeviceGetTotalEccErrors(
            h, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC)
    except pynvml.NVMLError:  # ECC not supported (e.g. A10 consumer-mode) — record, don't fail
        snap["ecc_uncorrected"] = None
    return snap


class TorchBackend:
    """One fp16 matmul per device per round; async launches, sync on demand."""

    def __init__(self, n_gpus):
        self.mats = []
        for i in range(n_gpus):
            with torch.cuda.device(i):
                a = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.float16)
                b = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.float16)
                self.mats.append((a, b))

    def device_count(self):
        return torch.cuda.device_count()

    def launch(self, i):
        a, b = self.mats[i]
        with torch.cuda.device(i):
            a @ b  # async; sync() below

    def sync(self, i):
        torch.cuda.synchronize(i)


class CublasBackend:
    """ctypes cuBLAS GemmEx: fp16 in, fp32 accumulate (tensor cores). No torch needed."""

    CUDA_R_16F = 2
    CUBLAS_COMPUTE_32F = 68
    CUBLAS_GEMM_DEFAULT = -1
    CUBLAS_OP_N = 0

    def __init__(self, n_gpus):
        self.cudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
        self.cublas = ctypes.CDLL(ctypes.util.find_library("cublas") or "libcublas.so.12")
        self.handles, self.bufs = [], []
        nbytes = 2 * MATMUL_N * MATMUL_N
        for i in range(n_gpus):
            self._ck(self.cudart.cudaSetDevice(i), "cudaSetDevice")
            ptrs = []
            for _ in range(3):  # A, B, C
                p = ctypes.c_void_p()
                self._ck(self.cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)),
                         "cudaMalloc")
                # 0x3c3c as fp16 ≈ 1.06 — nonzero data so the multipliers actually toggle
                self._ck(self.cudart.cudaMemset(p, 0x3C, ctypes.c_size_t(nbytes)), "cudaMemset")
                ptrs.append(p)
            self.bufs.append(ptrs)
            h = ctypes.c_void_p()
            self._ck(self.cublas.cublasCreate_v2(ctypes.byref(h)), "cublasCreate")
            self.handles.append(h)

    @staticmethod
    def _ck(rc, what):
        assert rc == 0, f"{what} failed with status {rc}"

    def device_count(self):
        n = ctypes.c_int()
        self._ck(self.cudart.cudaGetDeviceCount(ctypes.byref(n)), "cudaGetDeviceCount")
        return n.value

    def launch(self, i):
        self._ck(self.cudart.cudaSetDevice(i), "cudaSetDevice")
        a, b, c = self.bufs[i]
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        n = MATMUL_N
        rc = self.cublas.cublasGemmEx(
            self.handles[i], self.CUBLAS_OP_N, self.CUBLAS_OP_N, n, n, n,
            ctypes.byref(alpha), a, self.CUDA_R_16F, n, b, self.CUDA_R_16F, n,
            ctypes.byref(beta), c, self.CUDA_R_16F, n,
            self.CUBLAS_COMPUTE_32F, self.CUBLAS_GEMM_DEFAULT)
        self._ck(rc, "cublasGemmEx")

    def sync(self, i):
        self._ck(self.cudart.cudaSetDevice(i), "cudaSetDevice")
        self._ck(self.cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def main():
    if pynvml is not None:
        pynvml.nvmlInit()
    if torch is not None:
        n = torch.cuda.device_count()
        backend_name = "torch"
    else:
        assert pynvml is not None, "neither torch nor pynvml available — cannot burn"
        n = pynvml.nvmlDeviceGetCount()
        backend_name = "cublas-ctypes"
    print(f"RESULT gpus_visible={n} backend={backend_name}", flush=True)
    uuids = [gpu_uuid(i) for i in range(n)]
    log_receipt(params={
        "gpus_visible": n,
        "backend": backend_name,
        "gpu_name": gpu_name(0) if n else "none",
        "gpu_uuids": ",".join(uuids),
        "nvml": "available" if pynvml is not None else "missing",
        "torch_version": torch.__version__ if torch is not None else "absent",
        "burn_seconds_requested": BURN_SECONDS,
    })
    expect = os.environ.get("EXPECT_GPUS")
    assert not expect or n == int(expect), f"expected {expect} GPUs, saw {n}"
    assert n > 0, "no CUDA devices visible"

    backend = TorchBackend(n) if torch is not None else CublasBackend(n)
    start = {i: nvml_snapshot(i) for i in range(n)}
    stats = {i: {"iters": 0, "max_temp_c": 0, "max_power_w": 0.0, "hw_throttle_samples": 0}
             for i in range(n)}

    flops_per_mm = 2 * MATMUL_N**3
    t0 = time.time()
    loops = 0
    while time.time() - t0 < BURN_SECONDS:
        for i in range(n):
            backend.launch(i)
            stats[i]["iters"] += 1
        loops += 1
        if loops % 25 == 0:
            for i in range(n):
                backend.sync(i)
                s = nvml_snapshot(i)
                if s:
                    st = stats[i]
                    st["max_temp_c"] = max(st["max_temp_c"], s["temp_c"])
                    st["max_power_w"] = max(st["max_power_w"], s["power_w"])
                    if s["throttle"] & HW_THROTTLE_MASK:
                        st["hw_throttle_samples"] += 1
    for i in range(n):
        backend.sync(i)
    elapsed = time.time() - t0

    failures = []
    per_gpu_metrics = {}
    for i in range(n):
        st = stats[i]
        end = nvml_snapshot(i)
        ecc_delta = None
        if start[i].get("ecc_uncorrected") is not None:
            ecc_delta = end["ecc_uncorrected"] - start[i]["ecc_uncorrected"]
        tflops = flops_per_mm * st["iters"] / elapsed / 1e12
        print(f"RESULT gpu={i} name='{gpu_name(i)}' tflops={tflops:.1f} "
              f"max_temp_c={st['max_temp_c']} max_power_w={st['max_power_w']:.0f} "
              f"hw_throttle_samples={st['hw_throttle_samples']} ecc_uncorrected_delta={ecc_delta}",
              flush=True)
        per_gpu_metrics[f"tflops_fp16_gpu{i}"] = tflops
        per_gpu_metrics[f"max_temp_c_gpu{i}"] = st["max_temp_c"]
        per_gpu_metrics[f"max_power_w_gpu{i}"] = st["max_power_w"]
        per_gpu_metrics[f"hw_throttle_samples_gpu{i}"] = st["hw_throttle_samples"]
        per_gpu_metrics[f"ecc_uncorrected_delta_gpu{i}"] = -1 if ecc_delta is None else ecc_delta
        if ecc_delta not in (None, 0):
            failures.append(f"gpu{i}: {ecc_delta} uncorrected ECC errors")
        if st["hw_throttle_samples"] > 0:
            failures.append(f"gpu{i}: HW throttle observed under load")
    if failures:
        log_receipt(params={"burn": "FAIL", "burn_failures": "; ".join(failures)[:490]},
                    metrics=per_gpu_metrics)
        raise AssertionError("; ".join(failures))
    log_receipt(params={"burn": "PASS"}, metrics=per_gpu_metrics)
    print(f"RESULT burn=PASS gpus={n} seconds={BURN_SECONDS}", flush=True)


if __name__ == "__main__":
    main()
