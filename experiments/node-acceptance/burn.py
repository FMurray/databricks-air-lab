"""GPU burn + health check (UAT A1). Single process, all GPUs concurrently.

CUDA kernel launches are async: one thread keeps every device busy by round-robin queueing
matmuls on each device's default stream. Deliberately NO multiprocessing — mp spawn/Manager
deadlocked inside the AIR launcher (observed 2026-07-24, run 603656373172623 hung until
timeout with BURN_SECONDS=30).

Env knobs: BURN_SECONDS (default 120), EXPECT_GPUS (assert count if set), MATMUL_N (default 8192).
Exit 0 only if: all expected GPUs enumerate, uncorrected ECC delta is 0 on every GPU,
and no HW_SLOWDOWN throttle reason was observed under load.
"""
import os
import time

import torch

try:
    import pynvml
except ImportError:  # nvidia-ml-py in the workload deps; degrade gracefully without it
    pynvml = None

BURN_SECONDS = int(os.environ.get("BURN_SECONDS", "120"))
MATMUL_N = int(os.environ.get("MATMUL_N", "8192"))
# nvmlClocksThrottleReasons: HW_SLOWDOWN | HW_THERMAL | HW_POWER_BRAKE
HW_THROTTLE_MASK = 0x8 | 0x40 | 0x80


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


def main():
    if pynvml is not None:
        pynvml.nvmlInit()
    n = torch.cuda.device_count()
    print(f"RESULT gpus_visible={n}", flush=True)
    expect = os.environ.get("EXPECT_GPUS")
    assert not expect or n == int(expect), f"expected {expect} GPUs, saw {n}"
    assert n > 0, "no CUDA devices visible"

    mats = []
    for i in range(n):
        with torch.cuda.device(i):
            a = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.float16)
            b = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.float16)
            mats.append((a, b))
    start = {i: nvml_snapshot(i) for i in range(n)}
    stats = {i: {"iters": 0, "max_temp_c": 0, "max_power_w": 0.0, "hw_throttle_samples": 0}
             for i in range(n)}

    flops_per_mm = 2 * MATMUL_N**3
    t0 = time.time()
    loops = 0
    while time.time() - t0 < BURN_SECONDS:
        for i, (a, b) in enumerate(mats):
            with torch.cuda.device(i):
                c = a @ b  # noqa: F841 — async launch; sync below
            stats[i]["iters"] += 1
        loops += 1
        if loops % 25 == 0:
            for i in range(n):
                torch.cuda.synchronize(i)
                s = nvml_snapshot(i)
                if s:
                    st = stats[i]
                    st["max_temp_c"] = max(st["max_temp_c"], s["temp_c"])
                    st["max_power_w"] = max(st["max_power_w"], s["power_w"])
                    if s["throttle"] & HW_THROTTLE_MASK:
                        st["hw_throttle_samples"] += 1
    for i in range(n):
        torch.cuda.synchronize(i)
    elapsed = time.time() - t0

    failures = []
    for i in range(n):
        st = stats[i]
        end = nvml_snapshot(i)
        ecc_delta = None
        if start[i].get("ecc_uncorrected") is not None:
            ecc_delta = end["ecc_uncorrected"] - start[i]["ecc_uncorrected"]
        tflops = flops_per_mm * st["iters"] / elapsed / 1e12
        print(f"RESULT gpu={i} name='{torch.cuda.get_device_name(i)}' tflops={tflops:.1f} "
              f"max_temp_c={st['max_temp_c']} max_power_w={st['max_power_w']:.0f} "
              f"hw_throttle_samples={st['hw_throttle_samples']} ecc_uncorrected_delta={ecc_delta}",
              flush=True)
        if ecc_delta not in (None, 0):
            failures.append(f"gpu{i}: {ecc_delta} uncorrected ECC errors")
        if st["hw_throttle_samples"] > 0:
            failures.append(f"gpu{i}: HW throttle observed under load")
    assert not failures, "; ".join(failures)
    print(f"RESULT burn=PASS gpus={n} seconds={BURN_SECONDS}", flush=True)


if __name__ == "__main__":
    main()
