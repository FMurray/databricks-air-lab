"""A1 coverage driver v3: adaptive concurrency to detect the GPU_8xH100 cap being raised.

Same contract as v2 (16 real node outcomes, quota refusals don't count) plus:
- restart-safe: re-reads the outcomes file and adopts my in-flight gpu-burn runs
- width starts at known_cap+1 (probe slot); quota refusal -> width=cap, 15 min cooldown;
  an admitted probe raises the known cap and immediately probes one higher.
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--profile", required=True)
_ap.add_argument("--user", required=True, help="workspace user email (adopts their in-flight gpu-burn runs)")
_ap.add_argument("--target", type=int, default=16)
_ap.add_argument("--known-cap", type=int, default=4)
_ap.add_argument("--outcomes", default="burn-outcomes.txt")
_ap.add_argument("--repo", default=".")
_args = _ap.parse_args()
REPO = _args.repo
PROFILE = _args.profile
TARGET = _args.target
KNOWN_CAP = _args.known_cap
PROBE_COOLDOWN_S = 900
OUT = _args.outcomes


def now():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def submit():
    cmd = ["air", "run", "--file", "workloads/gpu-burn.example.yaml", "-p", PROFILE,
           "--override", "compute.accelerator_type=GPU_8xH100", "compute.num_accelerators=8",
           "env_variables.EXPECT_GPUS=8", "env_variables.BURN_SECONDS=900"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=300)
    for line in (r.stdout + r.stderr).splitlines():
        if "Job Run ID:" in line:
            return line.rsplit(":", 1)[1].strip()
    print(f"{now()} SUBMIT FAILED: {(r.stdout + r.stderr)[-300:]}", flush=True)
    return None


def state_of(run_id):
    r = subprocess.run(["databricks", "jobs", "get-run", run_id, "-p", PROFILE],
                       capture_output=True, text=True, timeout=120)
    try:
        s = json.loads(r.stdout).get("state", {})
    except Exception:
        return "UNKNOWN", ""
    return (s.get("result_state") or s.get("life_cycle_state") or "UNKNOWN",
            s.get("state_message") or "")


def adopt_active():
    """Adopt my in-flight gpu-burn 8xH100 runs from a previous driver instance."""
    r = subprocess.run(["databricks", "jobs", "list-runs", "--active-only",
                        "--user", _args.user, "-p", PROFILE,
                        "--output", "json"], capture_output=True, text=True, timeout=120)
    try:
        rs = json.loads(r.stdout)
        rs = rs if isinstance(rs, list) else rs.get("runs", [])
    except Exception:
        rs = []
    ids = [str(x["run_id"]) for x in rs if "gpu-burn" in (x.get("run_name") or "")]
    return ids


outcomes = []
if os.path.exists(OUT):
    with open(OUT) as f:
        outcomes = [tuple(line.strip().split(",")) for line in f if "," in line]
active = adopt_active()
print(f"{now()} adopted {len(active)} active runs {active}; {len(outcomes)} prior outcomes", flush=True)

cap = KNOWN_CAP
width = cap + 1  # probe slot
probe_block_until = 0
quota_refusals = 0

while len(outcomes) < TARGET:
    effective = cap if time.time() < probe_block_until else width
    while len(active) < effective and len(active) + len(outcomes) < TARGET + (effective - cap):
        rid = submit()
        if rid:
            active.append(rid)
            print(f"{now()} submitted {rid} (active={len(active)}/{effective} done={len(outcomes)} cap={cap})", flush=True)
        else:
            probe_block_until = time.time() + 120
            break
    time.sleep(60)
    for rid in list(active):
        st, msg = state_of(rid)
        if st in ("SUCCESS", "FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR"):
            active.remove(rid)
            if "GPU quota" in msg:
                quota_refusals += 1
                probe_block_until = time.time() + PROBE_COOLDOWN_S
                print(f"{now()} {rid} QUOTA-REFUSED (cap confirmed {cap}; next probe in {PROBE_COOLDOWN_S//60}m)", flush=True)
            else:
                outcomes.append((rid, st))
                with open(OUT, "a") as f:
                    f.write(f"{rid},{st}\n")
                print(f"{now()} {rid} -> {st} ({len(outcomes)}/{TARGET})", flush=True)
    # cap-raise detection: more admitted runs than the known cap
    if len(active) > cap:
        cap = len(active)
        width = cap + 1
        probe_block_until = 0
        print(f"{now()} *** CAP RAISED: {cap} concurrent admitted — probing {width} next ***", flush=True)
    print(f"{now()} heartbeat active={len(active)} done={len(outcomes)} cap={cap} refusals={quota_refusals}", flush=True)

print(f"DONE outcomes={len(outcomes)} quota_refusals={quota_refusals} final_cap={cap}", flush=True)
for rid, st in outcomes:
    print(f"FINAL {rid} {st}", flush=True)
