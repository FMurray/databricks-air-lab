# Load training data on AIR

Goal: get training data into a serverless-GPU job on the **CLI/torchrun path** (the path the
customer engagement uses — multi-node is CLI-only).

!!! note "Epistemic status: verified on A10, one axis pending"
    The `UCVolumeDataset` pattern below is verified end-to-end on `GPU_1xA10`
    (run `227734973662034`, 2026-08-27): import via the PYTHONPATH bridge, auto-sharding across
    DataLoader **workers** (exact cover of 96 files, no dup/gap), and local `/tmp` caching. The
    cross-**rank** axis (world ≥ 2) uses the same `files[global :: world×workers]` algorithm but is
    not yet run on 8×H100. Full receipts: `experiments/dataloading/NOTES.md`.

## Two ingest paths, by data shape

| Data | Pattern |
|---|---|
| **Tabular** (Delta) | Spark Connect → `spark.table(...).toPandas()`; for large tables, export to Parquet on a UC Volume once and read the Parquet directly (keeps Spark out of the GPU loop). |
| **Files / unstructured** | `serverless_gpu.data.UCVolumeDataset` + `serverless_gpu.data.DataLoader` — auto-shards + caches. Covered below. |

## The sharp edge: `serverless_gpu` isn't in the torch venv on the CLI path

`serverless_gpu` (v0.5.22, dist `databricks_serverless_gpu`) is preinstalled for **notebook**
sessions, but on the CLI/torchrun path it lives **only in the base python**
(`/databricks/python3/.../site-packages`), which has **no torch** — while torch lives in the
`databricks-ai` venv, which has **no `serverless_gpu`**. `serverless_gpu.data` hard-requires torch,
so it fails to import in either by default (verified: runs `961855733100717`, `838281261554839`).

Bridge the on-image package onto the torch venv — **no pip, no egress** (matters for no-egress
workspaces):

```yaml
command: >-
  PYTHONPATH=/databricks/python3/lib/python3.12/site-packages:$PYTHONPATH
  /opt/databricks-environments/databricks-ai/bin/torchrun
  --nnodes=$NUM_NODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR
  --master_port=$MASTER_PORT --nproc_per_node=$LOCAL_WORLD_SIZE --max-restarts=0
  $CODE_SOURCE_PATH/train.py
```

## The `UCVolumeDataset` pattern

```python
from serverless_gpu.data import UCVolumeDataset, DataLoader
from torch.utils.data import IterableDataset

class Decode(IterableDataset):            # docs: wrap it in a second IterableDataset
    def __init__(self, base): self.base = base
    def __iter__(self):
        for path in self.base:            # yields a LOCAL cached path (under /tmp), not /Volumes
            yield decode(path)            # open/decode in the SAME iteration — paths are ephemeral

ds = UCVolumeDataset("/Volumes/<cat>/<schema>/<vol>/images")   # non-recursive: point at the dir
loader = DataLoader(Decode(ds), batch_size=32, num_workers=6)  # SGC defaults: nw=6, prefetch=4
for batch in loader:
    ...
```

- **Auto-sharding**: with `torch.distributed` initialised, files split across ranks, then across
  DataLoader workers — `R×W` non-overlapping slices, exact cover. **No `DistributedSampler`.**
- **Caching**: first access copies each file to a local `/tmp/fslayer_cache_*` dir and yields that
  path; multi-epoch training reads from local disk, not the FUSE mount.

## Gotchas (each cost a run to find)

!!! warning "Build the dataset AFTER the data is visible"
    Constructing `UCVolumeDataset` immediately after writing files to the volume yielded **0 files**
    (run `455033116690072`) — a UC-FUSE write→list lag. If your job stages data onto the volume,
    confirm it's listable (`os.listdir`) before constructing the dataset.

!!! warning "`num_workers=0` is not allowed"
    The SGC `DataLoader` forces `prefetch_factor=4`, which PyTorch rejects at `num_workers=0`. Use
    `num_workers ≥ 1`, and the **same value on every rank** — the global stride is over
    `world_size × num_workers`; a mismatch silently duplicates or drops samples.

!!! warning "Never cancel a run to debug it"
    A cancelled AIR run ships **no logs** (run `157486619679259`). Let a timeout or an in-code
    watchdog fire instead, so the failure is diagnosable.

## Can't add dataloader CPUs to a GPU node

CPU-bound tokenization/decode competes with the GPU on the same node (`getting-started/pick-your-compute.md`).
Push heavy preprocessing upstream into Spark/Delta and land training-ready shards on the Volume.
