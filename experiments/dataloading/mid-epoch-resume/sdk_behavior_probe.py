"""Assertion-based probe of the AIR SDK's data-checkpointing semantics (and of the wrapper
patterns built on it). Each check prints ``PROBE <name> OK|FAIL ...``; the final line is
``SDK_BEHAVIOR_PROBE_OK`` only if every check matched its expectation.

Runs anywhere the SDK imports. On AIR it uses the real /Volumes path you pass; off-cluster set
``AIR_LAB_LOCAL_FS=1`` and pass any local dir (see _local_fs.py):

    AIR_LAB_LOCAL_FS=1 PYTHONPATH=~/universe/ai-compute/databricks-sdk-air \\
      python sdk_behavior_probe.py --root /tmp/mer-probe

Checks
  file_level_exactly_once  SDK DataLoader(num_workers=3) resume is exactly-once at file level
  order_after_resume       rows, num_workers=3: exactly-once; reports whether the post-resume
                           *order* matched (it can differ: round-robin restarts at worker 0)
  grid_pinned              loading a num_workers=2 state into num_workers=3 raises
  delegating_wrapper       a wrapper delegating state_dict() to UCVolumeDataset: docs say the seam
                           file is replayed; observed: its unread tail is skipped     [EXPECTED BUG]
  doc_wrapper_first_file   the SDK docs' wrapper: checkpoint during worker 1's first file ->
                           restore raises (blob stamped worker_id=0 in __init__)        [EXPECTED BUG]
  doc_wrapper_epoch2       the SDK docs' wrapper: checkpoint during epoch 2's first file ->
                           resume silently skips rows                                     [EXPECTED BUG]
  rows_first_file          ResumableRows, same scenario as doc_wrapper_first_file: exactly-once
  rows_epoch2              ResumableRows, same scenario as doc_wrapper_epoch2: exactly-once
  rows_shuffle_order       ResumableRows(shuffle_seed) num_workers=1: resumed order == uninterrupted
  dcp_nontensor_dedup      DCP: per-rank non-tensor values under ONE key collapse to one value
  dcp_json_roundtrip       DCP: gathered JSON string leaf round-trips via UCVolumeWriter/Reader
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.utils.data import IterableDataset  # noqa: E402

from make_synthetic_dataset import parse_npy_rows, write_dataset  # noqa: E402
from resumable_data import ResumableRows, import_air_data  # noqa: E402

AIR, AIR_PATH = import_air_data()
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"PROBE {name} {'OK' if ok else 'FAIL'} {detail}", flush=True)


def ids_of(batch):
    return [int(i) for i in batch["id"]]


# ---- the SDK docs' wrapper example, transcribed (CLAUDE.md "Checkpointing a wrapper dataset") --
class DocExamplesFromFiles(IterableDataset):
    def __init__(self, inner):
        self._inner = inner
        self._file_start_state = inner.state_dict()
        self._examples_emitted = 0
        self._skip = 0

    def __iter__(self):
        skip, self._skip = self._skip, 0
        for local_path in self._inner:
            for i, example in enumerate(parse_npy_rows(local_path, 0)):
                if i < skip:
                    continue
                self._examples_emitted = i + 1
                yield example
            skip = 0
            self._file_start_state = self._inner.state_dict()
            self._examples_emitted = 0

    def state_dict(self):
        return {"inner": self._file_start_state, "examples_emitted": self._examples_emitted}

    def load_state_dict(self, state):
        self._inner.load_state_dict(state["inner"])
        self._file_start_state = state["inner"]
        self._examples_emitted = state["examples_emitted"]
        self._skip = state["examples_emitted"]


class DelegatingRows(IterableDataset):
    """The SDK docs' "delegate state_dict() straight to the inner dataset" option."""

    def __init__(self, inner):
        self._inner = inner

    def __iter__(self):
        for local_path in self._inner:
            yield from parse_npy_rows(local_path, 0)

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state):
        self._inner.load_state_dict(state)


def rows_loader(data_dir, nw, kind, bs=4, seed=None):
    files = AIR.UCVolumeDataset(data_dir)
    if kind == "doc":
        ds = DocExamplesFromFiles(files)
    elif kind == "delegate":
        ds = DelegatingRows(files)
    else:
        ds = ResumableRows(files, parse_npy_rows, shuffle_seed=seed)
    return AIR.DataLoader(ds, batch_size=bs, num_workers=nw), ds


def run_epoch_rest(it):
    out = []
    for b in it:
        out += ids_of(b)
    return out


# ------------------------------------------------------------------------------------------------
def probe_file_level(data_dir, all_ids_by_file):
    # batch_size=None -> one local path per item; we decode the file ids to check coverage
    def load(nw):
        return AIR.DataLoader(AIR.UCVolumeDataset(data_dir), batch_size=None, num_workers=nw)

    def file_ids(path):
        return [r["id"] for r in parse_npy_rows(path, 0)]

    ref = []
    for p in load(3):
        ref.append(file_ids(p)[0])
    a = load(3)
    it = iter(a)
    seen = [file_ids(next(it))[0] for _ in range(5)]
    state = a.state_dict()
    del it, a
    b = load(3)
    b.load_state_dict(state)
    gc.collect()
    rest = [file_ids(p)[0] for p in b]
    got = Counter(seen + rest)
    exact = got == Counter(ref) and all(v == 1 for v in got.values())
    check("file_level_exactly_once", exact,
          f"files={len(ref)} before={len(seen)} after={len(rest)} dupes="
          f"{sum(v - 1 for v in got.values() if v > 1)} missing={len(set(ref) - set(got))} "
          f"per_worker_state={state['per_worker']}")
    del b
    gc.collect()


def probe_order(data_dir, all_ids):
    # Rows, 3 workers: torch hands out worker batches round-robin. After a resume the re-forked
    # pool restarts the round-robin at worker 0, so the *set* is exactly-once but the *order* of
    # the remaining batches can differ from an uninterrupted run (no bit-exact replay at nw>1).
    ref_loader, _ = rows_loader(data_dir, 3, "rows")
    ref = run_epoch_rest(iter(ref_loader))
    del ref_loader
    gc.collect()
    before, after, err = interrupted(data_dir, "rows", 3, 0, 4)
    dup, miss = summarize(all_ids, before, after or [])
    resumed = before + (after or [])
    check("order_after_resume", err is None and dup == 0 and miss == 0,
          f"exactly_once=True(dupes={dup},missing={miss}) identical_order={resumed == ref} "
          f"uninterrupted_next={ref[16:24]} resumed_next={(after or [])[:8]} (informational)")


def probe_grid(data_dir):
    a, _ = rows_loader(data_dir, 2, "rows")
    it = iter(a)
    next(it); next(it)
    st = a.state_dict()
    del it
    b, _ = rows_loader(data_dir, 3, "rows")
    try:
        b.load_state_dict(st)
        check("grid_pinned", False, "no error loading nw=2 state into nw=3")
    except ValueError as e:
        check("grid_pinned", True, f"ValueError: {str(e)[:110]}...")


def interrupted(data_dir, kind, nw, warm_epochs, k_batches, seed=None):
    """Iterate `warm_epochs` full epochs, then k batches of the next epoch, checkpoint, rebuild,
    resume and finish that epoch. Returns (rows delivered in the interrupted epoch, error)."""
    a, _ = rows_loader(data_dir, nw, kind, seed=seed)
    for _ in range(warm_epochs):
        run_epoch_rest(iter(a))
    it = iter(a)
    before = []
    for _ in range(k_batches):
        before += ids_of(next(it))
    state = a.state_dict()
    del it, a
    gc.collect()
    b, ds = rows_loader(data_dir, nw, kind, seed=seed)
    try:
        if kind == "rows":
            ds.prime_epoch(warm_epochs)
        b.load_state_dict(state)
        after = run_epoch_rest(iter(b))
    except Exception as e:  # noqa: BLE001 - the buggy pattern raises from a worker
        return before, None, f"{type(e).__name__}: {str(e).splitlines()[-1][:140]}"
    finally:
        del b, ds
        gc.collect()
    return before, after, None


def summarize(all_ids, before, after):
    got = Counter(before + after)
    dup = sum(v - 1 for v in got.values() if v > 1)
    missing = len(set(all_ids) - set(got))
    return dup, missing


def probe_wrappers(data_dir, all_ids):
    # docs: delegating gives "file-granularity resume ... the seam file is replayed (at-least-once)".
    # The inner stream counts a file delivered when it is YIELDED (before its rows are emitted), so
    # the seam file's remaining rows are skipped instead: at-most-once, silent data loss.
    before, after, err = interrupted(data_dir, "delegate", 2, 0, 2)
    dup, miss = summarize(all_ids, before, after or [])
    check("delegating_wrapper", err is None and miss > 0 and dup == 0,
          f"[docs say at-least-once replay] observed: err={err} dupes={dup} missing={miss} "
          f"(missing>0, dupes=0 => the seam file's tail is SKIPPED: at-most-once)")
    # first-file scenario: nw=2, 2 batches -> each worker delivered 1 batch, both mid first file
    before, after, err = interrupted(data_dir, "doc", 2, 0, 2)
    check("doc_wrapper_first_file", err is not None and "different stream" in err,
          f"[expected bug reproduces] error={err}")
    before, after, err = interrupted(data_dir, "rows", 2, 0, 2)
    dup, miss = summarize(all_ids, before, after or [])
    check("rows_first_file", err is None and dup == 0 and miss == 0,
          f"err={err} before={len(before)} after={len(after or [])} dupes={dup} missing={miss}")

    # epoch-2 scenario: one full epoch, then 2 batches of epoch 2 (still in each worker's 1st file)
    before, after, err = interrupted(data_dir, "doc", 2, 1, 2)
    dup, miss = summarize(all_ids, before, after or [])
    check("doc_wrapper_epoch2", err is None and miss > 0,
          f"[expected bug reproduces] err={err} epoch2_rows_delivered={len(before) + len(after or [])} "
          f"of {len(all_ids)} missing={miss} dupes={dup}")
    before, after, err = interrupted(data_dir, "rows", 2, 1, 2)
    dup, miss = summarize(all_ids, before, after or [])
    check("rows_epoch2", err is None and dup == 0 and miss == 0,
          f"err={err} before={len(before)} after={len(after or [])} dupes={dup} missing={miss}")


def probe_shuffle(data_dir):
    # shuffle determinism: single worker -> the resumed sequence must equal the uninterrupted one
    ref_loader, _ = rows_loader(data_dir, 1, "rows", seed=7)
    run_epoch_rest(iter(ref_loader))                      # epoch 0
    ref = run_epoch_rest(iter(ref_loader))                # epoch 1 (uninterrupted)
    before, after, err = interrupted(data_dir, "rows", 1, 1, 13, seed=7)
    seq = before + (after or [])
    epoch0 = run_epoch_rest(iter(rows_loader(data_dir, 1, "rows", seed=7)[0]))
    check("rows_shuffle_order", err is None and seq == ref and ref != epoch0,
          f"err={err} resumed_seq==uninterrupted={seq == ref} epoch1_differs_from_epoch0={ref != epoch0}")


# ------------------------------------------------------------------------------------------------
def _dcp_worker(rank, world, root, port, q):
    import json

    import torch.distributed.checkpoint as dcp
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mod, _ = import_air_data()
    out = {}
    # (1) per-rank values under one key
    path1 = os.path.join(root, "dcp_dedup")
    dcp.save({"t": torch.zeros(2), "who": f"rank{rank}"}, storage_writer=mod.UCVolumeWriter(path1))
    sd = {"t": torch.zeros(2), "who": ""}
    dcp.load(sd, storage_reader=mod.UCVolumeReader(path1))
    out["dedup_loaded"] = sd["who"]
    # (2) replicated JSON string from all_gather_object
    everyone = [None] * world
    dist.all_gather_object(everyone, {"rank": rank, "loader": {"per_worker": {0: {"rows": rank}}}})
    blob = json.dumps(everyone)
    path2 = os.path.join(root, "dcp_json")
    dcp.save({"t": torch.ones(2), "data": blob}, storage_writer=mod.UCVolumeWriter(path2))
    sd2 = {"t": torch.zeros(2), "data": ""}
    dcp.load(sd2, storage_reader=mod.UCVolumeReader(path2))
    out["json_ok"] = sd2["data"] == blob and json.loads(sd2["data"])[rank]["rank"] == rank
    out["metadata"] = os.path.exists(os.path.join(path2, ".metadata"))
    q.put((rank, out))
    dist.destroy_process_group()


def probe_dcp(root):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_dcp_worker, args=(r, 2, root, 29611, q)) for r in range(2)]
    for p in procs:
        p.start()
    res = dict(q.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(60)
    loaded = {r: res[r]["dedup_loaded"] for r in res}
    collapsed = len(set(loaded.values())) == 1
    check("dcp_nontensor_dedup", collapsed,
          f"saved rank0->'rank0', rank1->'rank1' under one key; loaded={loaded} "
          f"(collapsed={collapsed}: per-rank values under a shared key do NOT survive)")
    check("dcp_json_roundtrip", all(res[r]["json_ok"] and res[r]["metadata"] for r in res),
          f"{ {r: res[r] for r in res} }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="scratch dir (a /Volumes path on AIR)")
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--rows-per-file", type=int, default=10)
    a = ap.parse_args()
    data_dir = os.path.join(a.root, "data")
    shutil.rmtree(a.root, ignore_errors=True)
    write_dataset(data_dir, a.files, a.rows_per_file, seq=8)
    all_ids = list(range(a.files * a.rows_per_file))
    print(f"PROBE_ENV sdk={AIR_PATH} file={AIR.__file__} torch={torch.__version__} "
          f"python={sys.version.split()[0]} root={a.root}", flush=True)

    probe_file_level(data_dir, None)
    probe_order(data_dir, all_ids)
    probe_grid(data_dir)
    probe_wrappers(data_dir, all_ids)
    probe_shuffle(data_dir)
    probe_dcp(a.root)

    n_ok = sum(RESULTS)
    print(f"PROBE_SUMMARY {n_ok}/{len(RESULTS)} checks matched expectations", flush=True)
    if all(RESULTS):
        print("SDK_BEHAVIOR_PROBE_OK", flush=True)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
