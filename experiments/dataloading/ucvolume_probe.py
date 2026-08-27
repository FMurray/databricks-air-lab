"""UCVolumeDataset data-loading probe on AIR — three assertion-gated proofs.

The docs (docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading) make three
load-bearing claims about `serverless_gpu.data.UCVolumeDataset` + `serverless_gpu.data.DataLoader`.
None is verified in this repo. This probe proves each behind its own sentinel that is unreachable
unless its assertion held ("Exited 0" is not evidence):

  1. UCVOLUME_SHARDING_OK  — files are partitioned across ranks × DataLoader workers with NO
                             duplication and NO gaps: the union of every (rank, worker)'s files
                             is exactly the dataset, once. This is the claim that lets you drop
                             the hand-written DistributedSampler.
  2. UCVOLUME_CACHE_OK     — files are copied to a fast LOCAL cache on first access and yielded
                             from there (the yielded path is NOT under /Volumes), so multi-epoch
                             training does not re-read the volume over FUSE.
  3. UCVOLUME_NWORKERS_OK  — every rank used the SAME num_workers. The partitioning uses a global
                             stride over world_size × num_workers; a mismatch silently duplicates
                             or drops samples (docs). This proves the invariant held (it does not
                             deliberately desync ranks — that would hang the collective).

Completion line:
  UCVOLUME_DATALOADER_COMPLETE — all three (the data-loading acceptance line). Emitted only at
                                 world>=2 where sharding is non-vacuous.

Design (mirrors experiments/foundation-models/fsdp/train_fsdp.py conventions):
  * Self-contained + egress-free. Rank 0 WRITES a known dataset of N tiny files into the UC volume
    (each file's payload is its own index), so the probe needs no pre-staged data — only a
    writable volume. The write is probe-guarded (BR-2 403 → BLOCKED, not a crash).
  * Assertion is content-based, not filename-based: each rank reads the integer payload of every
    file it is handed and we assert the gathered multiset of payloads == range(N) exactly once.
    Robust to whatever ordering / path scheme UCVolumeDataset yields.
  * `serverless_gpu.data` exists only on AIR. `--local` swaps in a stand-in IterableDataset that
    replicates the documented global-stride partitioning so the coverage harness (all_gather +
    exact-cover assertion + report) can be pre-flighted on CPU/gloo. The stand-in proves the
    HARNESS, not UCVolumeDataset — the real proof is the AIR run. This is labeled everywhere.

Pre-flight (single host, CPU/gloo, real partitioning across spawned procs+workers):
    python3 ucvolume_probe.py --local --local-world 2 --num-workers 2 --files 40
On AIR it is launched by torchrun (see workloads/ucvolume-dataloader.example.yaml).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import textwrap
import traceback as _tb
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

# ------------------------------------------------------------------------------------------
# Pre-registered constants — pinned here (and in NOTES.md) BEFORE submit. These are exact
# invariants: the dataset is N files with integer payloads [0, N), so exact-cover is bit-exact,
# not a tolerance.
# ------------------------------------------------------------------------------------------
DEFAULT_FILES = 96           # dataset size; divisible by common world×workers products (2·2, 2·6, 8·6-ish)
PAYLOAD_MAGIC = b"UCVOL"     # each file: MAGIC + 4-byte big-endian index, so a truncated/garbled read fails loud


# ==========================================================================================
# Acceptance report — COPIED VERBATIM from .claude/skills/acceptance-report/references/renderer.py.
# Do NOT import (each AIR YAML snapshots only its own dir; a shared import vanishes at runtime).
# Edit the renderer reference first, then re-sync this copy. Only WORKLOAD, the checks, and the
# call site are workload-specific.
# ==========================================================================================
WORKLOAD = "UCVOLUME DATA LOADING"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
NA = "N/A-at-this-scale"


@dataclass
class Check:
    """One acceptance check. `status` is one of the five enum values; `traceback` is retained
    (never swallowed) and fenced under the verdict when the run has any FAIL."""
    name: str
    status: str
    measured: str
    threshold: str
    what_why: str
    sufficient: str
    likely_means: str = ""
    traceback: str = ""


def _fail_from_exc(name, threshold, what_why, likely_means, exc) -> Check:
    """Turn an exception into a FAIL record (principle 1: record, don't re-raise) so the report
    still renders and the verdict/exit code can be derived from it. Trace is kept verbatim."""
    return Check(name=name, status=FAIL, measured=f"raised {type(exc).__name__}: {exc}",
                 threshold=threshold, what_why=what_why,
                 sufficient="A raised exception means the property could not be established.",
                 likely_means=likely_means, traceback="".join(_tb.format_exception(exc)))


def _wrap(text: str, indent: str = "               ") -> str:
    """Wrap a long field to ~92 cols, hanging-indented under its dotted label."""
    return textwrap.fill(text, width=96, initial_indent="", subsequent_indent=indent)


def _receipt(checks: "list[Check]", verdict: str, exit_code: int, test_id: str) -> None:
    """Dual-sink the verdict into MLflow params (the durable leg). stdout is the report's
    primary sink but it depends on the env's log delivery and expires with job-run retention
    (format spec §"Preconditions") — the receipt makes an absent stdout report disambiguable:
    receipt present = logs didn't ship; receipt absent = the run died before the verdict.
    Client API bound to MLFLOW_RUN_ID (never start_run — resuming the launcher-owned run
    fails silently on the job plane); alarm-guarded so a blocked tracking call can't hang
    the run; skips cleanly when MLFLOW_RUN_ID is unset (local)."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.log_param(run_id, "acceptance_verdict", verdict)
        client.log_param(run_id, "acceptance_exit", exit_code)
        if test_id:
            client.log_param(run_id, "acceptance_test_id", test_id)
        for i, c in enumerate(checks, 1):
            client.log_param(run_id, f"acceptance_check_{i}", f"{c.status} — {c.name}"[:490])
    except Exception as e:                                 # noqa: BLE001 — receipt is best-effort
        print(f"acceptance receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def render_report(checks: "list[Check]", run_id: str, profile: str, shape: str,
                  scope: str, runtime: str, sentinels: str, test_id: str = "") -> int:
    """Render every check identically and DERIVE the exit code last. Returns the exit code:
    any FAIL ⇒ 1; BLOCKED / SKIPPED / N/A alone ⇒ 0. Verdict is generated from scope + statuses
    so a run cannot claim a proof it did not perform (smoke ⇒ capped at ACCEPTED WITH CAVEATS).
    `test_id` is the UAT results-registry id (utils/verification/results/registry.py) — the
    join key shared by the registry row, the sheet row, and the MLflow receipt."""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    W = 70
    out = []
    out.append("=" * 20 + f" {WORKLOAD} ACCEPTANCE REPORT " + "=" * 20)
    out.append(f"Run {run_id}   Profile {profile}   Shape {shape} ( {scope} )")
    out.append(f"Runtime {runtime}   When {when}")
    out.append("")
    out.append(_wrap("Attests to what rank 0 observed. On multi-node the CLI streams node 0 "
                     "only (`air logs <id> --node N`). If this report is absent, treat it as a "
                     "failure.", indent="  "))
    out.append("-" * W)

    has_fail = False
    for i, c in enumerate(checks, 1):
        if c.status == FAIL:
            has_fail = True
        out.append(f"CHECK {i} — {c.name}")
        out.append(f"  Status ....... {c.status}")
        out.append(f"  Measured ..... {c.measured}   Threshold: {c.threshold}")
        out.append(f"  What & why ... {_wrap(c.what_why)}")
        out.append(f"  Sufficient ... {_wrap(c.sufficient)}")
        out.append("-" * W)

    # Verdict — derived from scope + statuses (never a parallel narrative).
    softs = [c for c in checks if c.status in (BLOCKED, SKIPPED, NA)]
    if has_fail:
        verdict, exit_code = "NOT ACCEPTED", 1
        vline = "One or more checks did not clear their threshold at this shape."
    elif scope == "smoke" or softs:
        verdict, exit_code = "ACCEPTED WITH CAVEATS", 0
        capped = "smoke scope (single-process): distributed properties are vacuous here" \
            if scope == "smoke" else \
            "some checks were blocked / skipped / not applicable at this scale"
        vline = f"Every check that ran passed, but {capped} — see the rows above."
    else:
        verdict, exit_code = "ACCEPTED", 0
        vline = f"All checks passed at {shape}."
    out.append(f"VERDICT: {verdict}")
    out.append(f"  {vline}   Sentinels: {sentinels}   Test-id: {test_id or '-'}   "
               f"Exit: {exit_code}")

    # On FAIL — plain English first, then the raw trace (format spec §"On FAIL"). Never swallowed.
    if has_fail:
        out.append("")
        out.append("WHAT THIS LIKELY MEANS")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL:
                out.append(_wrap(f"CHECK {i} failed: {c.measured} did not meet "
                                 f"{c.threshold}. {c.likely_means}", indent="  "))
        out.append("")
        out.append("FOR SUPPORT — raw traceback")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL and c.traceback:
                out.append(f"  [CHECK {i} — {c.name}]")
                out.append(c.traceback.rstrip())

    print("\n" + "\n".join(out), flush=True)
    # Receipt AFTER the print: report delivery is priority one; the receipt is the durable leg.
    _receipt(checks, verdict, exit_code, test_id)
    return exit_code


# ==========================================================================================
# Dataset — N tiny files, payload = own index. Rank 0 writes; all ranks read via UCVolumeDataset.
# ==========================================================================================
def encode(idx: int) -> bytes:
    return PAYLOAD_MAGIC + idx.to_bytes(4, "big")


def decode(path: str) -> int:
    """Read a file yielded by the dataset and recover its integer payload. Fails loud on a
    truncated/garbled read (which would otherwise masquerade as a coverage gap)."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(PAYLOAD_MAGIC) or len(raw) != len(PAYLOAD_MAGIC) + 4:
        raise ValueError(f"bad payload in {path!r}: {raw!r}")
    return int.from_bytes(raw[len(PAYLOAD_MAGIC):], "big")


def _solo_probe_write(path: str, timeout_s: int = 15) -> bool:
    """Write+fsync+remove a tiny sentinel under an alarm. A SOLO write IS safely abortable
    (unlike a collective). Proves permission/403 (the BR-2 failure mode), not throughput."""
    def _timeout(signum, frame):
        raise TimeoutError(f"probe write to {path} exceeded {timeout_s}s")
    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout_s)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"probe")
            f.flush()
            os.fsync(f.fileno())
        os.remove(path)
        return True
    except Exception as e:                                 # noqa: BLE001 — any failure ⇒ probe fail
        print(f"[rank{_rank()}] probe write FAILED: {type(e).__name__}: {e}", flush=True)
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def write_dataset(data_dir: str, n: int) -> None:
    """Rank 0 only. Idempotent: clears any stale shard_*.bin first (the UC volume persists across
    runs, so leftovers from a prior --files count would masquerade as 'extra' files in Check 1),
    then writes exactly n files shard_{i}.bin with payload i."""
    os.makedirs(data_dir, exist_ok=True)
    for name in os.listdir(data_dir):
        if name.startswith("shard_") and name.endswith(".bin"):
            os.remove(os.path.join(data_dir, name))
    for i in range(n):
        with open(os.path.join(data_dir, f"shard_{i:05d}.bin"), "wb") as f:
            f.write(encode(i))


# ==========================================================================================
# Local stand-in — replicates the DOCUMENTED global-stride partitioning so the coverage harness
# can be pre-flighted on CPU. This is NOT UCVolumeDataset; it proves the harness, not the product.
# ==========================================================================================
class _LocalVolumeStandin(IterableDataset):
    """Yields absolute file paths, partitioned across world_size × num_workers by a global stride
    of (rank·num_workers + worker_id). Mirrors the docs' description; used only under --local."""
    def __init__(self, data_dir: str, rank: int, world: int, num_workers: int):
        self.files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                            if f.endswith(".bin"))
        self.rank, self.world = rank, world
        self.num_workers = max(1, num_workers)

    def __iter__(self):
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        stride = self.world * self.num_workers
        offset = self.rank * self.num_workers + worker_id
        for i in range(offset, len(self.files), stride):
            yield self.files[i]


# Candidate import paths for the serverless-GPU data API. The public docs show `serverless_gpu.data`,
# but the installed DISTRIBUTION is `databricks.serverless_gpu` (env survey — node-acceptance/NOTES.md),
# so the importable module may live under either name depending on the env image. Try both and report
# which resolved — the resolved path is itself a finding (docs vs reality).
_SGC_DATA_PATHS = ("serverless_gpu.data", "databricks.serverless_gpu.data")


def import_sgc_data():
    """Return (UCVolumeDataset, DataLoader, module_path). Raises ImportError with a diagnostic
    (which paths were tried + any serverless-ish installed dists + the interpreter) if none resolve."""
    import importlib
    errors = {}
    for path in _SGC_DATA_PATHS:
        try:
            mod = importlib.import_module(path)
            return mod.UCVolumeDataset, mod.DataLoader, path
        except Exception as e:                             # noqa: BLE001
            errors[path] = f"{type(e).__name__}: {e}"
    try:
        from importlib import metadata
        dists = sorted(d.metadata["Name"] for d in metadata.distributions()
                       if "serverless" in (d.metadata["Name"] or "").lower())
    except Exception:                                      # noqa: BLE001
        dists = ["<enumerate-failed>"]
    raise ImportError(f"serverless-GPU data API not found. tried={errors} "
                      f"serverless_dists={dists} python={sys.executable}")


READ_BATCH = 8         # DataLoader batch size — the docs use a real batch size, NOT batch_size=None.


def _identity(batch):
    """Keep each batch as a python list of (payload, parent) tuples — no tensor coercion, so the
    tally logic is trivial and default_collate never chokes on the string parent path."""
    return batch


class _PathDecode(IterableDataset):
    """The documented 'wrap UCVolumeDataset in a second IterableDataset' pattern: consume the
    path-yielding base dataset and yield (payload:int, cache_parent:str) per file. Reads the file
    in the SAME iteration the path is yielded (yielded paths are ephemeral). Used for both the AIR
    UCVolumeDataset and the local stand-in so the two paths exercise identical loop mechanics."""
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

    def __iter__(self):
        for path in self.path_dataset:
            if isinstance(path, bytes):
                path = path.decode()
            path = str(path)
            yield decode(path), os.path.dirname(os.path.realpath(path))


def build_dataset(data_dir: str, rank: int, world: int, num_workers: int, local: bool):
    """Return (loader, cache_expected: bool). Follows the docs exactly: base path-dataset wrapped
    in a decode IterableDataset, a real batch_size, identity collate."""
    if local:
        base = _LocalVolumeStandin(data_dir, rank, world, num_workers)
        from torch.utils.data import DataLoader as TorchDataLoader
        loader = TorchDataLoader(_PathDecode(base), batch_size=READ_BATCH,
                                 num_workers=num_workers, collate_fn=_identity)
        return loader, False   # stand-in yields the /Volumes-equivalent path, no cache leg
    # AIR: the real thing (import path discovered at runtime).
    UCVolumeDataset, DataLoader, _ = import_sgc_data()
    base = UCVolumeDataset(data_dir)
    loader = DataLoader(_PathDecode(base), batch_size=READ_BATCH,
                        num_workers=num_workers, collate_fn=_identity)
    return loader, True


# ==========================================================================================
# Distributed helpers.
# ==========================================================================================
def _rank() -> int:
    try:
        return dist.get_rank()
    except Exception:                                      # noqa: BLE001 — pre-init
        return int(os.environ.get("RANK", 0))


def _iter_indices(loader) -> "tuple[list[int], set[str]]":
    """Drain the loader; return (payloads seen, set of parent dirs of yielded paths). The parent
    dirs let us assert caching: on AIR the yielded path must live under a LOCAL cache, not under
    the /Volumes FUSE data dir. Each batch is a list of (payload, parent) tuples (identity collate)."""
    seen: list[int] = []
    parents: set[str] = set()
    for batch in loader:
        for payload, parent in batch:
            seen.append(payload)
            parents.add(parent)
    return seen, parents


# ==========================================================================================
# MLflow — tracking endpoint only (log_param/log_metric), bound to the AIR-injected run via the
# CLIENT API (NOT mlflow.start_run(run_id=…), which resumes the launcher-owned run and fails
# silently on the job plane — same pattern as train_fsdp.py). Populates the run as the probe
# progresses so it is self-documenting even before the acceptance receipt. Degrades to stdout
# when MLFLOW_RUN_ID is unset (local) or the endpoint is unreachable.
# ==========================================================================================
class _MlflowReceipt:
    def __init__(self, client, run_id: str):
        self._c, self._run_id = client, run_id

    def log_metric(self, key, value, step=None):
        self._c.log_metric(self._run_id, key, value, step=step if step is not None else 0)

    def log_param(self, key, value):
        self._c.log_param(self._run_id, key, str(value))


def open_mlflow(rank: int):
    """ALARM-GUARDED (like the renderer's _receipt): a blocked tracking endpoint must not hang the
    run — a prior run stalled ~13 min in an unguarded log_param. On timeout/error, degrade to
    stdout. Only guards establishing the connection; if reachable, subsequent calls are fast."""
    if rank != 0:
        return None
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        print("[rank0] MLFLOW_RUN_ID unset — logging to stdout only", flush=True)
        return None
    signal.alarm(60)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.log_param(run_id, "mlflow_tracking_reachable", "yes")
        return _MlflowReceipt(client, run_id)
    except Exception as e:                                 # noqa: BLE001
        print(f"[rank0] mlflow tracking endpoint NOT reachable/timed out: {e} — stdout only",
              flush=True)
        return None
    finally:
        signal.alarm(0)


# ==========================================================================================
# Worker — one rank.
# ==========================================================================================
def worker(rank: int, world: int, args) -> int:
    backend = "gloo" if args.local else "nccl"
    if args.local:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.master_port))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group(backend, rank=rank, world_size=world)
    if not args.local:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cpu" if args.local else "cuda")  # NCCL can't all_reduce a CPU tensor

    data_dir = os.path.join(args.volume_dir, "dataset")
    runtime_str = f"torch {torch.__version__}, backend {backend}, local={args.local}"

    sgc_ver, sgc_module = "n/a(local)", "n/a(local)"
    if not args.local:
        try:
            _, _, sgc_module = import_sgc_data()
            sgc_ver = "resolved"
        except Exception as e:                             # noqa: BLE001
            sgc_ver = f"import-failed:{type(e).__name__}"
            sgc_module = str(e)
    if rank == 0:
        print(f"UCVOLUME_VERSIONS torch={torch.__version__} serverless_gpu={sgc_ver} "
              f"world={world} num_workers={args.num_workers} files={args.files} "
              f"local={args.local}", flush=True)
        print(f"UCVOLUME_SGC_MODULE {sgc_module}", flush=True)

    # MLflow: bind to the AIR-injected run and log params NOW so the run is populated early
    # (not only at the acceptance receipt). rank 0 only.
    mlf = open_mlflow(rank)
    if mlf:
        mlf.log_param("torch_version", torch.__version__)
        mlf.log_param("world", world)
        mlf.log_param("num_workers", args.num_workers)
        mlf.log_param("files", args.files)
        mlf.log_param("epochs", args.epochs)
        mlf.log_param("sgc_module", sgc_module)
        mlf.log_param("volume_dir", args.volume_dir)

    # --- BR-2 gate + dataset write (rank 0) -------------------------------------------------
    # Probe the volume is writable BEFORE anyone reads. If not writable ⇒ BLOCKED-on-BR-2 for
    # the whole probe (no dataset to read), reported cleanly rather than a crash.
    writable = True
    if rank == 0:
        writable = _solo_probe_write(os.path.join(data_dir, ".probe_write"))
        if writable:
            write_dataset(data_dir, args.files)
            print(f"[rank0] wrote {args.files} files to {data_dir}", flush=True)
    flag = torch.tensor([1.0 if writable else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    dist.barrier()                                         # all ranks wait for the dataset to land

    # UC FUSE write→list LAG: a dataset built immediately after the write saw 0 files
    # (run 455033116690072 yielded 0; the identical read against a settled dir yielded 96 —
    # run 1105654492853413). Confirm the just-written files are visible on every rank before
    # building UCVolumeDataset, which lists this same FUSE dir at construction.
    if flag.item() >= 1.0:
        import time
        visible = 0
        for attempt in range(15):
            try:
                visible = sum(1 for f in os.listdir(data_dir) if f.endswith(".bin"))
            except FileNotFoundError:
                visible = 0
            if visible >= args.files:
                break
            time.sleep(2)
        if rank == 0:
            print(f"UCVOLUME_PHASE dataset-visible files={visible}/{args.files} "
                  f"retries={attempt}", flush=True)
        dist.barrier()

    checks: list[Check] = []
    sentinels: list[str] = []
    # The exact-cover property runs over world × num_workers partition units, so it is non-vacuous
    # whenever that product > 1 — i.e. a single A10 (world=1) with num_workers≥2 still exercises the
    # worker axis of the global stride. Only the cross-RANK num_workers-agreement check (Check 3)
    # needs more than one rank.
    cover_vacuous = (world * args.num_workers) <= 1
    rank_vacuous = world == 1

    if flag.item() < 1.0:
        blocked_note = ("volume not writable (BR-2 403) — the dataset could not be staged")
        for nm, th in (("Files partition across ranks×workers with no gaps or duplicates",
                        "exact cover of the dataset, once"),
                       ("Files are cached locally, not re-read from the volume over FUSE",
                        "yielded path under a local cache, not /Volumes"),
                       ("Every rank used the same num_workers",
                        "num_workers identical across all ranks")):
            checks.append(Check(name=nm, status=BLOCKED, measured=blocked_note, threshold=th,
                                what_why="Data loading needs a readable dataset in the volume.",
                                sufficient="Blocked by an external precondition (UC-volume write "
                                           "permission), not a fault in data loading itself.",
                                likely_means="The volume path could not be written — a "
                                             "permissions/BR-2 block, not a data-loading failure."))
    else:
        # --- read via UCVolumeDataset across `--epochs` epochs -----------------------------
        try:
            loader, cache_expected = build_dataset(
                data_dir, rank, world, args.num_workers, args.local)
            if rank == 0:
                print("UCVOLUME_PHASE dataset-built", flush=True)
            epoch_payloads = []
            all_parents: set[str] = set()

            # Read watchdog: a hung DataLoader/UCVolumeDataset worker or FUSE stall must SELF-ABORT
            # (exit 1 → v5 ships logs) rather than silently run to the platform timeout — a
            # cancelled run ships NO logs, so a silent hang is undiagnosable. AIR only (SIGALRM).
            def _watchdog(signum, frame):
                raise TimeoutError(
                    f"read watchdog fired after {args.read_timeout}s with "
                    f"num_workers={args.num_workers} — likely a DataLoader worker startup or "
                    f"UCVolumeDataset/FUSE stall; retry with --num-workers 0 to isolate")
            if not args.local:
                signal.signal(signal.SIGALRM, _watchdog)
                signal.alarm(args.read_timeout)
            try:
                for ep in range(args.epochs):
                    seen, parents = _iter_indices(loader)
                    epoch_payloads.append(sorted(seen))
                    all_parents |= parents
                    if rank == 0:
                        print(f"UCVOLUME_PHASE epoch{ep}-done files={len(seen)}", flush=True)
            finally:
                if not args.local:
                    signal.alarm(0)
            my_payloads = epoch_payloads[0]
        except Exception as e:                             # noqa: BLE001
            c = _fail_from_exc(
                "Files partition across ranks×workers with no gaps or duplicates",
                "exact cover of the dataset, once",
                "UCVolumeDataset + serverless_gpu.data.DataLoader must hand each rank a disjoint "
                "slice of the files that together cover the dataset exactly once.",
                "The dataset/loader raised before yielding — check UCVOLUME_VERSIONS for a "
                "serverless_gpu import/API mismatch and send this report.", e)
            checks = [c]
            _render_and_exit(checks, ["UCVOLUME_DATALOADER_INCOMPLETE"], rank, world, runtime_str,
                             args)
            return 0

        # --- gather every rank's payloads + worker count -----------------------------------
        gathered: list = [None] * world
        dist.all_gather_object(gathered, my_payloads)
        nworkers_all: list = [None] * world
        dist.all_gather_object(nworkers_all, args.num_workers)

        # CHECK 1 — exact cover (no dup, no gap) across ranks×workers.
        union = [p for sub in gathered for p in sub]
        expected = list(range(args.files))
        dup = len(union) != len(set(union))
        missing = sorted(set(expected) - set(union))
        extra = sorted(set(union) - set(expected))
        cover_ok = (not dup) and (not missing) and (not extra)
        if rank == 0 and cover_ok and not cover_vacuous:
            print(f"UCVOLUME_SHARDING_OK files={args.files} world={world} "
                  f"workers={args.num_workers} units={world * args.num_workers} "
                  f"union={len(union)} unique={len(set(union))}", flush=True)
        checks.append(Check(
            name="Files partition across ranks×workers with no gaps or duplicates",
            status=(NA if cover_vacuous else (PASS if cover_ok else FAIL)),
            measured=(f"{len(union)} files yielded across {world} ranks; "
                      f"{len(set(union))} unique; missing={missing[:8]} extra={extra[:8]}"),
            threshold=f"exact cover of {args.files} files, each seen exactly once",
            what_why="UCVolumeDataset is supposed to shard files across ranks and DataLoader "
                     "workers for you, so you never write a DistributedSampler. If it double-"
                     "feeds or skips files, every epoch silently trains on the wrong data.",
            sufficient="An exact cover (no missing, no duplicate) proves the automatic "
                       "partitioning is correct at this shape. It runs over world × num_workers "
                       "units, so one GPU with num_workers≥2 still tests the worker axis; only "
                       "world × num_workers = 1 makes it N/A.",
            likely_means="Files were duplicated or dropped — usually a num_workers mismatch "
                         "across ranks, or an unexpected partitioning scheme. Send this report "
                         "and the UCVOLUME_VERSIONS line."))

        # CHECK 2 — caching: yielded paths live under a LOCAL cache, not the /Volumes FUSE dir.
        real_data = os.path.realpath(data_dir)
        under_volume = any(p == real_data or p.startswith(real_data + os.sep)
                           for p in all_parents)
        # Must actually have seen files — an empty yield can't confirm caching (was a false PASS).
        saw_files = bool(all_parents)
        cache_ok = (saw_files and not under_volume) if cache_expected else True
        sample_parent = sorted(all_parents)[0] if all_parents else "(none)"
        if rank == 0 and cache_ok and cache_expected:
            print(f"UCVOLUME_CACHE_OK sample_cache_parent={sample_parent} "
                  f"data_dir={real_data}", flush=True)
        checks.append(Check(
            name="Files are cached locally, not re-read from the volume over FUSE",
            status=(NA if not cache_expected else (PASS if cache_ok else FAIL)),
            measured=f"yielded-path parent(s)={sorted(all_parents)[:3]}; volume data_dir={real_data}",
            threshold="yielded path under a local cache directory, not under /Volumes",
            what_why="On first access UCVolumeDataset copies each file to fast local storage and "
                     "yields the local path, so multi-epoch training reads from local disk instead "
                     "of paying the FUSE round-trip to the volume every epoch.",
            sufficient="A yielded path outside the /Volumes data dir means caching is active. "
                       "Under --local there is no cache leg (a stand-in dataset), so this is N/A.",
            likely_means="Files were served straight from the FUSE mount (no local cache) — "
                         "multi-epoch throughput will suffer; confirm the serverless_gpu version."))

        # CHECK 3 — num_workers identical across ranks (the invariant the global stride requires).
        nworkers_consistent = len(set(nworkers_all)) == 1
        if rank == 0 and nworkers_consistent and not rank_vacuous:
            print(f"UCVOLUME_NWORKERS_OK num_workers={args.num_workers} "
                  f"all_ranks={nworkers_all}", flush=True)
        checks.append(Check(
            name="Every rank used the same num_workers",
            status=(NA if rank_vacuous else (PASS if nworkers_consistent else FAIL)),
            measured=f"num_workers per rank = {nworkers_all}",
            threshold="identical num_workers on every rank",
            what_why="The partitioning uses a global stride over world_size × num_workers. If "
                     "ranks disagree on num_workers the stride is inconsistent and files are "
                     "silently duplicated or dropped — the docs call this out explicitly.",
            sufficient="Identical num_workers across ranks means the stride is well-defined and "
                       "Check 1's exact cover is trustworthy. At world=1 there is one rank, so "
                       "there is nothing to agree on — N/A.",
            likely_means="Ranks launched with different num_workers — align the DataLoader "
                         "num_workers across all ranks (same value everywhere)."))

    # completion sentinel — only when all three PASS at world>=2 (never on the vacuous world=1 gate)
    all_pass = all(c.status == PASS for c in checks)
    complete = (world >= 2 and all_pass and len(checks) == 3)
    if rank == 0 and complete:
        print("UCVOLUME_DATALOADER_COMPLETE proofs=1,2,3 (sharding+cache+nworkers)", flush=True)
        sentinels = ["UCVOLUME_DATALOADER_COMPLETE"]
    else:
        sentinels = ["UCVOLUME_DATALOADER_INCOMPLETE"]

    if rank == 0 and mlf:
        for i, c in enumerate(checks, 1):
            mlf.log_param(f"check_{i}", f"{c.status} — {c.name}"[:250])
        mlf.log_metric("checks_passed", sum(1 for c in checks if c.status == PASS))
        mlf.log_metric("checks_total", len(checks))
        mlf.log_param("completion", sentinels[0])

    dist.barrier()
    _render_and_exit(checks, sentinels, rank, world, runtime_str, args)
    return 0


def _render_and_exit(checks, sentinels, rank, world, runtime_str, args) -> None:
    """Rank-0-only report render; exit code derived from the verdict (guarded)."""
    exit_code = 0
    if rank == 0:
        scope = "smoke" if world == 1 else "acceptance"
        shape = f"world={world}, workers={args.num_workers}, files={args.files}"
        try:
            run_id = (os.environ.get("MLFLOW_RUN_ID")
                      or os.environ.get("MLFLOW_RUN_NAME") or "local")
            exit_code = render_report(
                checks, run_id=run_id,
                profile=("local-cpu" if args.local else "air"),
                shape=shape, scope=scope, runtime=runtime_str,
                sentinels=" ".join(sentinels),
                test_id="dataloading")  # utils/verification/results/registry.py
        except Exception:                                  # noqa: BLE001 — never lose the verdict
            _tb.print_exc()
            exit_code = 1
    try:
        dist.destroy_process_group()
    except Exception:                                      # noqa: BLE001
        pass
    if rank == 0 and exit_code:
        sys.exit(exit_code)


# ==========================================================================================
# Entry point.
# ==========================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true",
                   help="single-host CPU/gloo pre-flight; spawns --local-world procs with a "
                        "stand-in dataset (proves the HARNESS, not UCVolumeDataset)")
    p.add_argument("--local-world", type=int, default=2, help="world size for --local (≥2 non-vacuous)")
    p.add_argument("--master-port", type=int, default=29530)
    p.add_argument("--volume-dir", default=None,
                   help="writable dir: a UC volume path on AIR (/Volumes/<cat>/<schema>/<vol>/...); "
                        "defaults to a /tmp dir under --local")
    p.add_argument("--files", type=int, default=DEFAULT_FILES, help="dataset size (N files)")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers per rank (SAME on every rank — Check 3)")
    p.add_argument("--epochs", type=int, default=2, help="epochs to drain (≥2 exercises caching)")
    p.add_argument("--read-timeout", type=int, default=300,
                   help="AIR-only watchdog (s): self-abort a hung read so logs ship (a cancelled "
                        "run ships none)")
    args = p.parse_args()

    if args.volume_dir is None:
        if not args.local:
            print("ERROR: --volume-dir (a UC volume path) is required on AIR", file=sys.stderr)
            return 2
        args.volume_dir = "/tmp/ucvolume_probe_local"

    if args.local:
        import torch.multiprocessing as mp
        try:
            mp.spawn(worker, args=(args.local_world, args), nprocs=args.local_world, join=True)
        except Exception:                                  # noqa: BLE001
            _tb.print_exc()
            return 1
        return 0
    else:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        return worker(rank, world, args) or 0


if __name__ == "__main__":
    sys.exit(main())
