"""Mid-epoch-resumable data pipeline for Databricks AIR, built on the AIR SDK's own data stack.

Copy this file into your project. It depends only on torch + the AIR SDK data module.

What the SDK already gives you (``databricks.air.data``, formerly ``serverless_gpu.data`` >= 0.5.21)
-------------------------------------------------------------------------------------------------
- ``UCVolumeDataset`` streams the files of a UC volume directory, partitioned into
  ``world_size x num_workers`` disjoint per-(rank, worker) streams. Its ``state_dict()`` is
  "how many files this stream has delivered".
- ``DataLoader`` harvests every worker's dataset state as batches are **delivered to your loop**
  (not as they are prefetched), and exposes ``state_dict()`` / ``load_state_dict()``. The state is
  pinned to the grid (``world_size``, ``num_workers``); resuming on a different grid raises.

What the SDK does NOT give you, and this module adds
----------------------------------------------------
1. **Sub-file resume.** The SDK resumes at *file* granularity: the file a stream was part-way
   through is replayed from its first row. With large files that is a lot of duplicated data, so
   ``ResumableRows`` fans each file into rows and checkpoints the row cursor too.
   The SDK's documented wrapper example has two first-file bugs this class avoids (both
   demonstrated by ``sdk_behavior_probe.py``):
     a. it snapshots the inner state in ``__init__`` (the main process), so the blob is stamped
        ``worker_id=0`` (and rank 0 if built before ``init_process_group``). A checkpoint taken
        while worker k>0 is still on its first file then fails to restore (stream mismatch).
     b. at the start of epoch >= 2 (persistent workers) its snapshot still holds the previous
        epoch's terminal state, so a checkpoint taken during an epoch's first file resumes past
        the whole epoch slice -- silently skipping data.
2. **Per-rank state in one checkpoint.** ``gather_data_state`` all-gathers every rank's loader
   state into a single JSON string that is identical on all ranks, so it rides safely in a DCP
   checkpoint (DCP keeps one copy of a non-tensor value per key -- per-rank values under a shared
   key would silently collapse to one rank's).
3. **A frozen-dataset guard.** ``dataset_fingerprint`` -- loader offsets are positions in the
   sorted file listing, so the listing must not change between attempts.
4. **Lockstep control.** ``StepControl`` does one tiny all-reduce per step so every rank agrees on
   "checkpoint now" (time-based triggers drift between ranks) and "SIGTERM received".
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import random
import signal
import sys
import time
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

# The SDK's UCVolumeDataset state layout this module reads (see ResumableRows.__iter__).
_INNER_STATE_VERSION = 1
_MIN_SDK_HINT = "serverless_gpu >= 0.5.21 or databricks-sdk-air >= 0.1.0"


# ---------------------------------------------------------------------------------------------
# SDK resolution
# ---------------------------------------------------------------------------------------------
def import_air_data():
    """Return ``(module, import_path)`` for the AIR SDK data module.

    The package was renamed: GPU env v6+ ships ``databricks.air.data`` (dist ``databricks-sdk-air``);
    env v5 ships ``serverless_gpu.data`` (dist ``databricks_serverless_gpu``). Both expose the same
    classes. Fails loudly if the installed SDK predates loader checkpointing.
    """
    if os.environ.get("AIR_LAB_LOCAL_FS") == "1":  # off-cluster testing only; see _local_fs.py
        import _local_fs  # noqa: F401

    errors = {}
    for path in ("databricks.air.data", "serverless_gpu.data", "databricks.serverless_gpu.data"):
        try:
            mod = importlib.import_module(path)
        except Exception as e:  # noqa: BLE001 - report every attempt
            errors[path] = f"{type(e).__name__}: {e}"
            continue
        if not hasattr(mod.DataLoader, "load_state_dict") or not hasattr(mod, "Checkpointable"):
            raise ImportError(
                f"{path} is too old for loader checkpointing (need {_MIN_SDK_HINT}); "
                f"found at {getattr(mod, '__file__', '?')}"
            )
        return mod, path
    raise ImportError(
        f"AIR SDK data module not importable from {sys.executable}. tried={errors}. On the GPU env "
        "v5 CLI path, serverless_gpu lives in the base python; bridge it with "
        "PYTHONPATH=/databricks/python3/lib/python3.12/site-packages (see the workload YAML)."
    )


# ---------------------------------------------------------------------------------------------
# Row-level resumable wrapper
# ---------------------------------------------------------------------------------------------
ParseFn = Callable[[str, int], Iterable[Any]]
"""``parse_file(local_path, start_row) -> iterable of rows [start_row:]``.

Must be deterministic (same file -> same rows, same order): on resume the seam file is re-opened
and rows before the cursor are skipped. Seek efficiently if your format allows it (mmap'd .npy,
parquet row groups, ...); a plain ``itertools.islice`` fallback is correct but re-reads the head.
The local path is ephemeral (the SDK cache evicts it once this worker reads its next file), so
read the file inside this call -- never stash the path.
"""


class ResumableRows(IterableDataset):
    """Fan each file of a checkpointable file stream into rows, with exact row-level resume.

    Args:
        files: the SDK ``UCVolumeDataset`` (or any ``Checkpointable`` dataset yielding local paths
            whose ``state_dict()`` has the ``{version: 1, delivered: int, ...}`` layout).
        parse_file: see ``ParseFn``.
        shuffle_seed: if set, rows *within each file* are permuted by a pure function of
            ``(seed, epoch, file ordinal)``, so each epoch sees a different order and a resumed
            run reproduces it exactly. Materializes one file's rows in memory. ``UCVolumeDataset``
            itself never reorders files (sorted listing, fixed stride per stream), so pre-shuffle
            rows across files when you write the dataset.

    State (per stream): ``{"version", "epoch", "files": <inner state at the start of the current
    file>, "rows": <rows of that file already delivered>}``. JSON-serialisable.
    """

    STATE_VERSION = 1

    def __init__(self, files: IterableDataset, parse_file: ParseFn,
                 shuffle_seed: Optional[int] = None) -> None:
        super().__init__()
        self._files = files
        self._parse = parse_file
        self._seed = shuffle_seed
        self._epoch = -1                       # incremented at the top of every fresh __iter__
        self._file_start: Optional[Dict[str, Any]] = None
        self._rows_emitted = 0
        self._resume: Optional[tuple] = None   # (epoch, rows_to_skip) armed by load_state_dict

    # -- epoch bookkeeping -------------------------------------------------------------------
    def prime_epoch(self, epoch: int) -> None:
        """Call in the main process on resume, before the first iteration.

        Workers that had delivered nothing at checkpoint time have no saved state, so they would
        otherwise restart their epoch counter at 0 (only matters for ``shuffle_seed``). Forked
        workers inherit this value; workers with saved state take the epoch from it instead.
        """
        self._epoch = epoch - 1

    @property
    def epoch(self) -> int:
        return self._epoch

    # -- iteration ---------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Any]:
        if self._resume is not None:
            self._epoch, skip = self._resume
            self._resume = None
        else:
            self._epoch += 1
            skip = 0
        for local_path in self._files:
            # Snapshot *after* the inner stream yields this file (inside the worker, so the blob
            # carries this stream's own rank/worker_id and this epoch's count). The SDK counts a
            # file as delivered the moment it is yielded, so "about to start this file" is one less.
            start = dict(self._files.state_dict())
            if start.get("version") != _INNER_STATE_VERSION or "delivered" not in start:
                raise ValueError(f"unexpected inner dataset state layout: {start!r} "
                                 f"(ResumableRows understands version {_INNER_STATE_VERSION})")
            start["delivered"] -= 1
            self._file_start = start

            if self._seed is None:
                rows: Iterable[Any] = self._parse(local_path, skip)
                for i, row in enumerate(rows, start=skip):
                    self._rows_emitted = i + 1     # count before yield: the loader snapshots
                    yield row                       # state while paused at this yield
            else:
                all_rows = list(self._parse(local_path, 0))
                order = list(range(len(all_rows)))
                random.Random(f"{self._seed}:{self._epoch}:{start['delivered']}").shuffle(order)
                for i in range(skip, len(order)):
                    self._rows_emitted = i + 1
                    yield all_rows[order[i]]
            skip = 0

    # -- Checkpointable ------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        if self._resume is not None:           # restored but not yet iterated
            epoch, rows = self._resume
        else:
            epoch, rows = self._epoch, self._rows_emitted
        return {"version": self.STATE_VERSION, "epoch": epoch, "files": self._file_start,
                "rows": rows}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("version") != self.STATE_VERSION:
            raise ValueError(f"unsupported ResumableRows state version {state.get('version')!r}")
        if state["files"] is not None:
            self._files.load_state_dict(state["files"])
        self._file_start = state["files"]
        self._rows_emitted = state["rows"]
        self._resume = (state["epoch"], state["rows"])


def islice_parse(open_rows: Callable[[str], Iterable[Any]]) -> ParseFn:
    """Adapt a plain ``path -> rows`` reader to ``ParseFn`` by skipping the head (correct, slow)."""
    def parse(local_path: str, start: int) -> Iterable[Any]:
        return itertools.islice(open_rows(local_path), start, None)
    return parse


def dataset_fingerprint(data_dir: str) -> str:
    """Hash of the sorted file listing (names + sizes) -- what ``UCVolumeDataset`` streams are
    built from. Loader offsets are positions in that listing: if files are added, removed, or
    rewritten between attempts, every stream silently resumes at the wrong data. Store this in the
    checkpoint and refuse to resume on a mismatch (write new data to a new, versioned directory)."""
    import hashlib
    h = hashlib.sha256()
    for e in sorted(os.scandir(data_dir), key=lambda e: e.path):
        if e.is_file(follow_symlinks=False):
            h.update(f"{e.name}\0{e.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# Folding every rank's loader state into one checkpoint entry
# ---------------------------------------------------------------------------------------------
def gather_data_state(loader, epoch: int, extra: Optional[Dict[str, Any]] = None) -> str:
    """Collective: every rank contributes ``{epoch, loader.state_dict()}``; returns the same JSON
    string (a list indexed by rank) on every rank. Store it as one non-tensor DCP entry."""
    mine = {"epoch": epoch, "loader": loader.state_dict(), **(extra or {})}
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world == 1:
        return json.dumps([mine])
    everyone = [None] * world
    dist.all_gather_object(everyone, mine)
    return json.dumps(everyone)


def my_data_state(data_state_json: str) -> Dict[str, Any]:
    """This rank's entry from ``gather_data_state``'s output. Raises if the world size changed --
    the SDK pins (world_size, num_workers); DCP can reshard the model, but not the data streams."""
    states = json.loads(data_state_json)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(states) != world:
        raise ValueError(f"checkpoint has data state for {len(states)} ranks but world_size={world}; "
                         "resume on the same node/GPU count it was taken on")
    return states[rank]


def restore_loader(loader, rows_ds: Optional[ResumableRows], data_state_json: str) -> int:
    """Restore this rank's loader position (call before the first ``iter(loader)``). Returns the
    rank's epoch."""
    mine = my_data_state(data_state_json)
    if rows_ds is not None:
        rows_ds.prime_epoch(mine["epoch"])
    loader.load_state_dict(mine["loader"])
    return mine["epoch"]


# ---------------------------------------------------------------------------------------------
# Lockstep control: checkpoint cadence + SIGTERM
# ---------------------------------------------------------------------------------------------
def ignore_sigterm_in_worker(worker_id: int) -> None:  # noqa: ARG001 - worker_init_fn signature
    """``worker_init_fn`` for the DataLoader: DataLoader worker processes ignore SIGTERM.

    torchrun forwards SIGTERM to each rank's whole process group, which includes its DataLoader
    workers. If they die, torch raises "DataLoader worker ... killed by signal: Terminated" in the
    rank -- typically in the middle of the emergency checkpoint. Ignoring SIGTERM in workers lets
    the rank finish the save; workers still exit when the rank exits (torch watches the parent).
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class StepControl:
    """One small all-reduce per step so all ranks agree on *when* to checkpoint / stop.

    A checkpoint is a collective (DCP save + the data-state gather). If ranks decided on their own
    clocks, one rank would enter the save while another runs the next forward pass -> hang. Any
    rank that is "due" (step cadence, wall-clock cadence, or SIGTERM) makes every rank save.
    Pair with ``ignore_sigterm_in_worker`` as the DataLoader's ``worker_init_fn``.
    """

    def __init__(self, device: torch.device, every_steps: int = 0, every_seconds: float = 0.0,
                 handle_sigterm: bool = True) -> None:
        self.device = device
        self.every_steps = every_steps
        self.every_seconds = every_seconds
        self._last_save = time.monotonic()
        self._sigterm = False
        if handle_sigterm:
            signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum, frame) -> None:  # noqa: ARG002
        self._sigterm = True                         # act on it at the next step boundary

    def sync(self, step: int) -> tuple:
        """Returns ``(save_now, stop_now)``, identical on every rank."""
        due = (self.every_steps > 0 and step % self.every_steps == 0) or (
            self.every_seconds > 0 and time.monotonic() - self._last_save >= self.every_seconds)
        flags = torch.tensor([float(due), float(self._sigterm)], device=self.device)
        if dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        save_now, stop_now = bool(flags[0] > 0), bool(flags[1] > 0)
        return save_now or stop_now, stop_now

    def mark_saved(self) -> None:
        self._last_save = time.monotonic()
