"""Prove exactly-once data delivery across interrupted attempts, from train.py's --ledger-dir.

The ledger holds, per attempt and rank, the (step, epoch, sample ids) of every trained batch up
to each checkpoint. Work done after an attempt's last checkpoint is never flushed (it is lost on
a crash and retrained after resume), so the ledger is exactly the effective training trajectory.
If a later attempt re-covers a step (it resumed from an older checkpoint), the later record wins.

Checks (all must pass for LEDGER_EXACTLY_ONCE_OK):
  contiguous   every rank has steps 1..N with no gaps, same N on all ranks
  no_dupes     within each (rank, epoch) no sample id appears twice
  disjoint     within each epoch no sample id is trained by two ranks
  exact_cover  every epoch that all ranks completed covers the full dataset exactly once
  resumed      (with --expect-resume) >= 2 attempts contributed, i.e. a resume really happened

    python verify_ledger.py --ledger-dir /Volumes/.../ledger --data-dir /Volumes/.../data \\
        [--expect-resume]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_ledger(root):
    attempts = sorted(d for d in os.listdir(root) if d.startswith("attempt_"))
    by_rank = defaultdict(dict)       # rank -> step -> (attempt, epoch, ids)
    for a in attempts:
        adir = os.path.join(root, a)
        for name in sorted(os.listdir(adir)):
            if not name.endswith(".jsonl"):
                continue
            rank = int(name.split("_")[1])
            with open(os.path.join(adir, name)) as f:
                for line in f:
                    r = json.loads(line)
                    by_rank[rank][r["step"]] = (a, r["epoch"], r["ids"])  # later attempt wins
    return attempts, by_rank


def all_ids(data_dir):
    ids = []
    for name in sorted(os.listdir(data_dir)):
        if name.endswith(".npy"):
            ids.append(np.load(os.path.join(data_dir, name), mmap_mode="r")[:, 0].copy())
    return set(np.concatenate(ids).tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--expect-resume", action="store_true")
    a = ap.parse_args()

    attempts, by_rank = load_ledger(a.ledger_dir)
    universe = all_ids(a.data_dir)
    ranks = sorted(by_rank)
    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= passed
        print(f"LEDGER {name} {'OK' if passed else 'FAIL'} {detail}", flush=True)

    # contiguous
    maxes = {r: max(by_rank[r]) for r in ranks}
    gaps = {r: sorted(set(range(1, maxes[r] + 1)) - set(by_rank[r]))[:5] for r in ranks}
    report("contiguous", all(not g for g in gaps.values()) and len(set(maxes.values())) == 1,
           f"ranks={len(ranks)} steps={sorted(set(maxes.values()))} first_gaps={ {r: g for r, g in gaps.items() if g} }")

    # per (rank, epoch) id lists
    per = defaultdict(list)            # (rank, epoch) -> ids
    contrib = defaultdict(set)         # attempt -> ranks that contributed steps
    for r in ranks:
        for step in sorted(by_rank[r]):
            att, ep, ids = by_rank[r][step]
            per[(r, ep)].extend(ids)
            contrib[att].add(r)
    dupes = {k: len(v) - len(set(v)) for k, v in per.items() if len(v) != len(set(v))}
    report("no_dupes", not dupes, f"rank_epochs={len(per)} with_dupes={dict(list(dupes.items())[:5])}")

    epochs = sorted({ep for _, ep in per})
    overlaps, covered = {}, {}
    last_epoch = {r: max(ep for rr, ep in per if rr == r) for r in ranks}
    for ep in epochs:
        seen, clash = set(), 0
        for r in ranks:
            s = set(per.get((r, ep), []))
            clash += len(seen & s)
            seen |= s
        if clash:
            overlaps[ep] = clash
        if all(last_epoch[r] > ep for r in ranks):          # every rank finished this epoch
            covered[ep] = (len(seen) == len(universe) and seen == universe, len(seen))
    report("disjoint", not overlaps, f"epochs={epochs} cross_rank_overlaps={overlaps}")
    report("exact_cover", all(v[0] for v in covered.values()),
           f"dataset_rows={len(universe)} complete_epochs={ {e: n for e, (_, n) in covered.items()} } "
           f"(epochs still in progress on some rank are checked for dupes/disjointness only)")
    if not covered:
        print("LEDGER note: no epoch was completed by every rank; exact_cover is vacuous — run "
              "more steps than one epoch to make it bite", flush=True)

    used = [att for att in attempts if contrib.get(att)]
    if a.expect_resume:
        report("resumed", len(used) >= 2, f"attempts_contributing={used}")
    else:
        print(f"LEDGER attempts_contributing={used}", flush=True)

    if ok:
        print(f"LEDGER_EXACTLY_ONCE_OK ranks={len(ranks)} steps={max(maxes.values())} "
              f"attempts={len(used)} complete_epochs={len(covered)}", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
