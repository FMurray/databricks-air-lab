"""UCVolumeDataset API-contract introspection — why did it yield 0 files?

Single process (no torchrun, no DataLoader workers): initialize a world=1 process group, then
(1) dump UCVolumeDataset / DataLoader signatures + docstrings + __iter__ source, (2) confirm the
volume dir is visible from this container (os.listdir), (3) iterate UCVolumeDataset DIRECTLY and
count what it yields. This isolates whether the 0-yield is a worker-subprocess-has-no-PG issue
(direct iteration would then yield all files) or a path/listing/visibility issue (direct iteration
also yields 0). Run with the PYTHONPATH bridge + the torch-having databricks-ai python.
"""
from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.distributed as dist


def dump(obj, name):
    try:
        print(f"INTRO_SIG {name} {inspect.signature(obj.__init__)}", flush=True)
    except Exception as e:                                 # noqa: BLE001
        print(f"INTRO_SIG {name} <no sig: {e}>", flush=True)
    doc = (getattr(obj, "__doc__", "") or "").strip().replace("\n", " ")
    print(f"INTRO_DOC {name} {doc[:600]}", flush=True)


def main() -> int:
    data_dir = sys.argv[1]
    from serverless_gpu.data import UCVolumeDataset, DataLoader
    import serverless_gpu.data as sgd
    print(f"INTRO_MODULE serverless_gpu.data attrs={[a for a in dir(sgd) if not a.startswith('_')]}",
          flush=True)
    dump(UCVolumeDataset, "UCVolumeDataset")
    dump(DataLoader, "DataLoader")
    for meth in ("__iter__", "_list_files", "list_files", "_files", "files"):
        fn = getattr(UCVolumeDataset, meth, None)
        if fn and callable(fn):
            try:
                src = inspect.getsource(fn)
                print(f"INTRO_SRC {meth}:\n{src[:1800]}\nINTRO_SRC_END", flush=True)
            except Exception as e:                         # noqa: BLE001
                print(f"INTRO_SRC {meth} <no source: {e}>", flush=True)

    # container's FUSE view of the dir
    try:
        listing = sorted(os.listdir(data_dir))
        print(f"INTRO_LISTDIR n={len(listing)} first={listing[:3]} dir={data_dir}", flush=True)
    except Exception as e:                                 # noqa: BLE001
        print(f"INTRO_LISTDIR FAIL {type(e).__name__}: {e} dir={data_dir}", flush=True)

    # world=1 process group so UCVolumeDataset can read rank info at iteration time
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "gloo"
    try:
        torch.cuda.set_device(0)
        backend = "nccl"
    except Exception:                                      # noqa: BLE001
        pass
    dist.init_process_group(backend, rank=0, world_size=1)
    print(f"INTRO_PG backend={backend} rank={dist.get_rank()} world={dist.get_world_size()}",
          flush=True)

    # DIRECT iteration (no DataLoader, no workers, main process WITH the PG)
    for variant in (data_dir, data_dir.rstrip("/"), os.path.dirname(data_dir)):
        try:
            ucv = UCVolumeDataset(variant)
            n, first = 0, []
            for p in ucv:
                if n < 3:
                    first.append((type(p).__name__, repr(p)[:160]))
                n += 1
            print(f"INTRO_DIRECT_ITER dir={variant} count={n} first={first}", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"INTRO_DIRECT_ITER dir={variant} FAIL {type(e).__name__}: {e}", flush=True)

    # --- DataLoader matrix: isolate whether num_workers>0 and/or the decode wrapper break it ---
    from torch.utils.data import IterableDataset

    class _Wrap(IterableDataset):
        def __init__(self, base):
            self.base = base
        def __iter__(self):
            for p in self.base:
                yield (decode_int(p), p)

    def _identity(b):
        return b

    def count_via_loader(make_ds, nw, label):
        try:
            loader = DataLoader(make_ds(), batch_size=8, num_workers=nw, collate_fn=_identity)
            n = 0
            for batch in loader:
                n += len(batch)
            print(f"INTRO_LOADER label={label} num_workers={nw} count={n}", flush=True)
        except Exception as e:                             # noqa: BLE001
            print(f"INTRO_LOADER label={label} num_workers={nw} FAIL {type(e).__name__}: {e}",
                  flush=True)

    for nw in (0, 1, 2):
        count_via_loader(lambda: UCVolumeDataset(data_dir), nw, "raw-ucv")
    for nw in (0, 2):
        count_via_loader(lambda: _Wrap(UCVolumeDataset(data_dir)), nw, "wrapped")

    print("INTRO_DONE", flush=True)
    return 0


def decode_int(path: str) -> int:
    with open(path, "rb") as f:
        raw = f.read()
    return int.from_bytes(raw[-4:], "big")


if __name__ == "__main__":
    sys.exit(main())
