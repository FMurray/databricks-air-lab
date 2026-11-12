"""Off-cluster end-to-end pre-flight: 2 simulated nodes x N procs, crash mid-epoch, resume, verify.

Runs the real AIR SDK data/checkpoint code (AIR_LAB_LOCAL_FS=1 lifts only its /Volumes prefix
check) on CPU/gloo. Two torchrun agents rendezvous with --nnodes=2 exactly like two AIR nodes.

    PYTHONPATH=~/universe/ai-compute/databricks-sdk-air \\
      uv run --no-project --with torch --with mlflow-skinny --with numpy \\
      python local_e2e.py --root /tmp/mer-e2e [--fsdp] [--shuffle-seed 7] [--sigterm]

Scenario (defaults): world=4 (2 nodes x 2), num_workers=2 -> 8 streams, 16 files x 64 rows,
batch 8 -> 32 steps per rank-epoch. Checkpoint every 25 steps; attempt 1 is hard-killed at step
60 (mid epoch 1, 10 steps after the step-50 checkpoint); attempt 2 resumes at 50 and runs to 80.
With --sigterm (400 steps), attempt 1 instead gets SIGTERM ~1 s after its first checkpoint and
must agree across ranks, checkpoint on the way out, and exit 143; attempt 2 resumes there.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def launch(args, extra, port):
    env = dict(os.environ, AIR_LAB_LOCAL_FS="1", NUM_NODES="2", OMP_NUM_THREADS="1")
    procs = []
    for node in range(2):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=2", f"--node_rank={node}",
               f"--nproc_per_node={args.nproc}", "--master_addr=127.0.0.1", f"--master_port={port}",
               "--max-restarts=0", os.path.join(HERE, "train.py"),
               "--data-dir", f"{args.root}/data", "--ckpt-dir", f"{args.root}/ckpt",
               "--ledger-dir", f"{args.root}/ledger", "--make-data", "16", "--rows-per-file", "64",
               "--seq", "32", "--dim", "64", "--layers", "2", "--heads", "2",
               "--batch-size", "8", "--num-workers", "2", "--steps", "80",
               "--ckpt-every-steps", "25", "--log-every", "10"] + extra
        if args.fsdp:
            cmd.append("--fsdp")
        if args.shuffle_seed is not None:
            cmd += ["--shuffle-seed", str(args.shuffle_seed)]
        log = open(f"{args.root}/node{node}_{port}.log", "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True), log, node))
    return procs


def wait(procs, timeout=240):
    codes = []
    for p, log, _ in procs:
        try:
            codes.append(p.wait(timeout))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            codes.append("timeout")
        log.close()
    return codes


def show(args, port, pattern=("COLD_START", "RESUMED", "CKPT_SAVED", "FORCED", "SIGTERM",
                               "TRAIN_DONE", "DATA ", "ENV ", "Traceback", "Error")):
    for node in range(2):
        with open(f"{args.root}/node{node}_{port}.log") as f:
            for line in f:
                if any(k in line for k in pattern):
                    print(f"  [node{node}] {line.rstrip()[:220]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/mer-e2e")
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--sigterm", action="store_true")
    args = ap.parse_args()
    shutil.rmtree(args.root, ignore_errors=True)
    os.makedirs(args.root)

    print("== attempt 1 (cold; will be interrupted)")
    long_run = ["--steps", "400"] if args.sigterm else []   # long enough to interrupt mid-run
    if args.sigterm:
        procs = launch(args, long_run, 29701)
        ckpt_dir = f"{args.root}/ckpt"
        deadline = time.time() + 300
        while time.time() < deadline:   # wait for the first periodic checkpoint, then SIGTERM
            if os.path.isdir(ckpt_dir) and any(n.startswith("step_") for n in os.listdir(ckpt_dir)):
                break
            time.sleep(0.5)
        time.sleep(1)
        for p, _, _ in procs:          # what a platform stop does: SIGTERM the launcher
            os.killpg(p.pid, signal.SIGTERM)
    else:
        procs = launch(args, ["--fail-at-step", "60"], 29701)
    print(f"   exit codes: {wait(procs)}")
    show(args, 29701)

    print("== attempt 2 (same command, as a max_retries resubmit would run it)")
    procs = launch(args, ["--fail-at-step", "60"] + long_run, 29702)   # fault flag stays; resumed runs skip it
    codes = wait(procs)
    print(f"   exit codes: {codes}")
    show(args, 29702)
    if codes != [0, 0]:
        return 1

    if args.sigterm:   # the emergency checkpoint must be the one attempt 2 resumed from
        import re
        a1 = "".join(open(f"{args.root}/node0_29701.log").readlines())
        a2 = "".join(open(f"{args.root}/node0_29702.log").readlines())
        m1 = re.search(r"SIGTERM_CHECKPOINTED step=(\d+)", a1)
        m2 = re.search(r"RESUMED_FROM_STEP=(\d+)", a2)
        good = bool(m1 and m2 and m1.group(1) == m2.group(1))
        print(f"SIGTERM_RESUME {'OK' if good else 'FAIL'} emergency_ckpt_step="
              f"{m1 and m1.group(1)} resumed_from={m2 and m2.group(1)}")
        if not good:
            return 1

    print("== verify ledger")
    r = subprocess.run([sys.executable, os.path.join(HERE, "verify_ledger.py"), "--ledger-dir",
                        f"{args.root}/ledger", "--data-dir", f"{args.root}/data", "--expect-resume"])
    print("== checkpoints kept:", sorted(os.listdir(f"{args.root}/ckpt")))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
