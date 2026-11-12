"""Stage a deterministic synthetic token dataset on a UC volume (or a local dir for testing).

Layout: ``<dir>/part-00000.npy ...``, each an int64 array of shape ``(rows_per_file, 1 + seq + 1)``:
column 0 is a globally unique sample id (so a ledger can prove exactly-once delivery), columns
1.. are ``seq + 1`` tokens (input + shifted target). Tokens are a learnable pure function of the
id, so loss goes down and reruns are bit-identical.

Swap in your own format by replacing ``parse_npy_rows`` with a ``ParseFn`` for it (see
``resumable_data.ParseFn``) -- e.g. parquet: open the file, map ``start_row`` to a row group,
iterate from there.

    python make_synthetic_dataset.py --out /Volumes/<cat>/<schema>/<vol>/mid-epoch-resume/data \
        --files 64 --rows-per-file 512 --seq 128
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch

VOCAB = 4096


def make_file(file_idx: int, rows: int, seq: int) -> np.ndarray:
    ids = np.arange(file_idx * rows, (file_idx + 1) * rows, dtype=np.int64)
    t = np.arange(seq + 1, dtype=np.int64)
    base = (ids * 1_000_003) % VOCAB
    stride = 1 + (ids % 7)
    tokens = (base[:, None] + t[None, :] * stride[:, None]) % VOCAB
    return np.concatenate([ids[:, None], tokens], axis=1)


def write_dataset(out_dir: str, n_files: int, rows_per_file: int, seq: int) -> int:
    """Idempotent: skips files that already exist. Writes via a local temp file + one copy so a
    partially written file never appears under its final name (FUSE has no atomic rename for
    large files, so we copy-then-rename a small-ish file; good enough for staging)."""
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for i in range(n_files):
        final = os.path.join(out_dir, f"part-{i:05d}.npy")
        if os.path.exists(final):
            continue
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            np.save(tmp, make_file(i, rows_per_file, seq))
        partial = final + ".partial"
        with open(tmp.name, "rb") as src, open(partial, "wb") as dst:
            dst.write(src.read())
        os.remove(tmp.name)
        os.replace(partial, final)
        written += 1
    return written


def parse_npy_rows(local_path: str, start: int):
    """``ParseFn`` for this layout: memory-maps the cached file and seeks straight to ``start``
    (no re-reading of the file head on resume)."""
    arr = np.load(local_path, mmap_mode="r")
    for i in range(start, arr.shape[0]):
        row = np.array(arr[i])                    # copy out of the mmap before the file is evicted
        yield {"id": int(row[0]), "tokens": torch.from_numpy(row[1:])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--files", type=int, default=64)
    p.add_argument("--rows-per-file", type=int, default=512)
    p.add_argument("--seq", type=int, default=128)
    a = p.parse_args()
    n = write_dataset(a.out, a.files, a.rows_per_file, a.seq)
    print(f"DATASET_READY dir={a.out} files={a.files} rows_per_file={a.rows_per_file} "
          f"seq={a.seq} newly_written={n}", flush=True)


if __name__ == "__main__":
    main()
