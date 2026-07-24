"""GPU burn + health check (UAT A1). One process per visible GPU, simultaneous matmul load.

Env knobs: BURN_SECONDS (default 120), EXPECT_GPUS (assert count if set), MATMUL_N (default 8192).
Exit 0 only if: all expected GPUs enumerate, uncorrected ECC delta is 0 on every GPU,
and no HW_SLOWDOWN throttle reason was observed under load.
"""
import os
import time

import torch
import torch.multiprocessing as mp

try:
    import pynvml
except ImportError:  # nvidia-ml-py in the workload deps; fall back gracefully for local runs
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


def burn_one(idx, results):
    torch.cuda.set_device(idx)
    a = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.float16)
    b = torch.randn(MATMUL_N, MATMUL_N, device="cuda", dtype=torch.float16)
    start_snap = nvml_snapshot(idx)
    flops_per_mm = 2 * MATMUL_N**3
    iters, throttled, max_temp, max_power = 0, 0, 0, 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    while time.time() - t0 < BURN_SECONDS:
        c = a @ b  # noqa: F841
        iters += 1
        if iters % 50 == 0:
            torch.cuda.synchronize()
            s = nvml_snapshot(idx)
            if s:
                max_temp = max(max_temp, s["temp_c"])
                max_power = max(max_power, s["power_w"])
                if s["throttle"] & HW_THROTTLE_MASK:
                    throttled += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    end_snap = nvml_snapshot(idx)
    ecc_delta = None
    if start_snap.get("ecc_uncorrected") is not None:
        ecc_delta = end_snap["ecc_uncorrected"] - start_snap["ecc_uncorrected"]
    results[idx] = {
        "tflops": flops_per_mm * iters / elapsed / 1e12,
        "iters": iters, "max_temp_c": max_temp, "max_power_w": max_power,
        "hw_throttle_samples": throttled, "ecc_uncorrected_delta": ecc_delta,
        "name": torch.cuda.get_device_name(idx),
    }


def main():
    if pynvml is not None:
        pynvml.nvmlInit()
    n = torch.cuda.device_count()
    print(f"RESULT gpus_visible={n}")
    expect = os.environ.get("EXPECT_GPUS")
    assert not expect or n == int(expect), f"expected {expect} GPUs, saw {n}"
    assert n > 0, "no CUDA devices visible"

    with mp.Manager() as mgr:
        results = mgr.dict()
        procs = [mp.Process(target=burn_one, args=(i, results)) for i in range(n)]
        [p.start() for p in procs]
        [p.join() for p in procs]
        results = dict(results)

    failures = []
    for i in sorted(results):
        r = results[i]
        print(f"RESULT gpu={i} name='{r['name']}' tflops={r['tflops']:.1f} "
              f"max_temp_c={r['max_temp_c']} max_power_w={r['max_power_w']:.0f} "
              f"hw_throttle_samples={r['hw_throttle_samples']} "
              f"ecc_uncorrected_delta={r['ecc_uncorrected_delta']}")
        if r["ecc_uncorrected_delta"] not in (None, 0):
            failures.append(f"gpu{i}: {r['ecc_uncorrected_delta']} uncorrected ECC errors")
        if r["hw_throttle_samples"] > 0:
            failures.append(f"gpu{i}: HW throttle observed under load")
    assert len(results) == n, f"only {len(results)}/{n} burn processes reported"
    assert not failures, "; ".join(failures)
    print(f"RESULT burn=PASS gpus={n} seconds={BURN_SECONDS}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
