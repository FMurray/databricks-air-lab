"""DCP checkpointing to a UC volume with the AIR SDK's UCVolumeWriter/UCVolumeReader.

Copy this file into your project alongside ``resumable_data.py``.

One checkpoint = one ``<root>/step_<NNNNNNNNN>/`` directory holding a single DCP save of
``{"model", "optim", "app": {"step"}, "data": <JSON str of every rank's loader state>}``. Saving
the data position *in the same DCP save* as the weights is what makes it atomic: the SDK writer
publishes ``.metadata`` only after every shard landed, so a directory with ``.metadata`` has
weights and data cursor from the same step, and a directory without one is ignored.

Rules this class enforces (each one is a way the naive version breaks multi-node):
- **Rank 0 decides which checkpoint to resume from and broadcasts it.** Ranks listing a FUSE
  directory independently can disagree while a save or prune is in flight.
- **Save is a collective.** Call it on every rank at the same step (``resumable_data.StepControl``).
- **One async save in flight.** The previous save is awaited before the next starts, before
  pruning, and before exit. Gather the data state *after* ``wait()`` too.
- **Checkpoint collectives get their own gloo process group.** ``dcp.async_save`` finishes its
  upload with collectives on a background thread. On the default group they can interleave with
  the training loop's own collectives (DDP all-reduce, object gathers) -> "collective mismatch"
  crash or hang. A dedicated group keeps the two streams apart (hit for real in local_e2e.py).
- **Prune only behind a complete save**, keeping the newest ``keep_last`` valid checkpoints.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional, Tuple

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

_STEP_DIR = re.compile(r"^step_(\d+)$")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _broadcast_obj(obj, group=None):
    if not dist.is_initialized():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0, group=group)
    return box[0]


def _wait(fut) -> None:
    # torch < 2.9: dcp.async_save returns a Future; newer versions may return an
    # AsyncSaveResponse whose upload_completion is the future that matters.
    if fut is None:
        return
    target = getattr(fut, "upload_completion", fut)
    target.result()


class Checkpointer:
    def __init__(self, root: str, air_data, keep_last: int = 3, use_async: bool = True) -> None:
        self.root = root.rstrip("/")
        self.air = air_data
        self.keep_last = keep_last
        self.use_async = use_async
        self._inflight = None
        # Collective: every rank creates the group in the same order.
        self.pg = dist.new_group(backend="gloo") if dist.is_initialized() else None
        if _rank() == 0:
            os.makedirs(self.root, exist_ok=True)
        _barrier()

    # -- discovery -----------------------------------------------------------------------------
    def step_dir(self, step: int) -> str:
        return f"{self.root}/step_{step:09d}"

    def _list_valid(self):
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in os.listdir(self.root):
            m = _STEP_DIR.match(name)
            if m and os.path.exists(os.path.join(self.root, name, ".metadata")):
                out.append((int(m.group(1)), os.path.join(self.root, name)))
        return sorted(out)

    def latest_valid(self) -> Optional[Tuple[int, str]]:
        """Collective. Newest complete checkpoint, as seen by rank 0, on every rank."""
        found = None
        if _rank() == 0:
            valid = self._list_valid()
            found = valid[-1] if valid else None
        return _broadcast_obj(found, group=self.pg)

    # -- save / load ---------------------------------------------------------------------------
    def save(self, step: int, model, optim, data_state_json: str, blocking: bool = False) -> str:
        """Collective. Starts (or, with ``blocking``, completes) a save of this step."""
        self.wait()                    # one save in flight; also makes the prune below safe
        self._prune()
        path = self.step_dir(step)
        if _rank() == 0 and os.path.exists(path):
            # A torn save of the same step from a previous attempt (no .metadata). Clear it so
            # stale shard files can't mix with this attempt's.
            shutil.rmtree(path, ignore_errors=True)
        _barrier()
        msd, osd = get_state_dict(model, optim)
        state = {"model": msd, "optim": osd, "app": {"step": step}, "data": data_state_json}
        writer = self.air.UCVolumeWriter(path)
        if self.use_async and not blocking:
            # Stages a CPU copy now (so the state is this step's), uploads in the background.
            # Needs a CPU backend in the process group: init_process_group("cpu:gloo,cuda:nccl").
            self._inflight = dcp.async_save(state, storage_writer=writer, process_group=self.pg)
        else:
            dcp.save(state, storage_writer=writer, process_group=self.pg)
        return path

    def wait(self) -> None:
        """Block until the in-flight async save (if any) has fully landed on the volume."""
        _wait(self._inflight)
        self._inflight = None

    def close(self) -> None:
        """Collective. Drain the in-flight save, then apply retention. Call before exit."""
        self.wait()
        _barrier()
        self._prune()
        _barrier()

    def load(self, path: str, model, optim) -> Tuple[int, str]:
        """Collective. Loads weights + optimizer in place; returns ``(step, data_state_json)``."""
        msd, osd = get_state_dict(model, optim)
        state = {"model": msd, "optim": osd, "app": {"step": -1}, "data": ""}
        dcp.load(state, storage_reader=self.air.UCVolumeReader(path), process_group=self.pg)
        set_state_dict(model, optim, model_state_dict=state["model"], optim_state_dict=state["optim"])
        return int(state["app"]["step"]), state["data"]

    # -- retention -----------------------------------------------------------------------------
    def _prune(self) -> None:
        if self.keep_last <= 0 or _rank() != 0:
            return
        valid = self._list_valid()
        keep = {p for _, p in valid[-self.keep_last:]}
        newest = valid[-1][0] if valid else -1
        for name in os.listdir(self.root) if os.path.isdir(self.root) else []:
            m = _STEP_DIR.match(name)
            if not m:
                continue
            p = os.path.join(self.root, name)
            torn_and_old = int(m.group(1)) < newest and not os.path.exists(os.path.join(p, ".metadata"))
            if (p not in keep and os.path.exists(os.path.join(p, ".metadata"))) or torn_and_old:
                shutil.rmtree(p, ignore_errors=True)
