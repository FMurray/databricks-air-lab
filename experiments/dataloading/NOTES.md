# Data loading — UCVolumeDataset probe

Verifies the three load-bearing claims the AIR docs make about training-data ingest via
`serverless_gpu.data.UCVolumeDataset` + `serverless_gpu.data.DataLoader`
(docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading). None was verified in this
repo before this probe. Companion cookbook page: `docs/cookbook/load-training-data.md` (written
once the AIR run lands). Report format: the `acceptance-report` skill.

Probe: `ucvolume_probe.py` · Workload: `workloads/ucvolume-dataloader.example.yaml`

## The claims under test (pre-registered — written BEFORE submit)

| # | Sentinel | Claim | How the probe proves it |
|---|---|---|---|
| 1 | `UCVOLUME_SHARDING_OK` | Files are partitioned across ranks × DataLoader workers with no gaps or duplicates (so you drop the hand-written `DistributedSampler`). | Rank 0 writes N files whose payload is their own index; every rank reads the payloads it is handed; `all_gather_object` the per-rank sets; assert the union == `range(N)` exactly once (no dup, no missing, no extra). Non-vacuous whenever `world × num_workers > 1`. |
| 2 | `UCVOLUME_CACHE_OK` | Files are copied to a fast LOCAL cache on first access and yielded from there — multi-epoch training doesn't re-read the volume over FUSE. | Assert the yielded path's real parent is NOT under the `/Volumes` data dir (i.e. it's a local cache path). Non-vacuous at world=1 (a single rank still caches). |
| 3 | `UCVOLUME_NWORKERS_OK` | Every rank must use the same `num_workers` — the partition is a global stride over `world × num_workers`; a mismatch silently duplicates/drops samples. | `all_gather_object` each rank's `num_workers`; assert all equal. (Does NOT deliberately desync ranks — that would hang the collective; it proves the invariant held.) Needs world ≥ 2. |

Completion line `UCVOLUME_DATALOADER_COMPLETE` prints only when all three **PASS** (⇒ requires
world ≥ 2 for Check 3). The A10 world=1 run proves Checks 1+2 via the worker axis and renders
**ACCEPTED WITH CAVEATS** — the individual sentinels carry that proof; the COMPLETE line is
reserved for the full multi-rank receipt.

### Success = for the default A10 run
- `UCVOLUME_SHARDING_OK` and `UCVOLUME_CACHE_OK` both print; Check 3 = N/A (single rank);
  verdict ACCEPTED WITH CAVEATS; exit 0.
- The `serverless_gpu` import succeeds on env v5 and `UCVolumeDataset`/`DataLoader` resolve
  (`UCVOLUME_VERSIONS` line reports the version).
### Success = for the 8xH100 full proof
- All three sentinels + `UCVOLUME_DATALOADER_COMPLETE`; verdict ACCEPTED.

## Local CPU pre-flight (harness only — NOT UCVolumeDataset)

`serverless_gpu.data` exists only on AIR, so `--local` swaps in a stand-in `IterableDataset` that
replicates the documented global-stride partitioning. This pre-flights the **harness** (worker
partitioning + `all_gather_object` + exact-cover assertion + report render), not the product.
Caching (Check 2) has no local leg → N/A under `--local`.

```
$ uv run --with torch --with numpy python ucvolume_probe.py --local --local-world 2 --num-workers 2 --files 40
UCVOLUME_VERSIONS torch=2.13.0 serverless_gpu=n/a(local) world=2 num_workers=2 files=40 local=True
[rank0] wrote 40 files to /tmp/ucvolume_probe_local/dataset
UCVOLUME_SHARDING_OK files=40 world=2 workers=2 union=40 unique=40
UCVOLUME_NWORKERS_OK num_workers=2 all_ranks=[2, 2]
  CHECK 1 ... PASS   CHECK 2 ... N/A (no local cache leg)   CHECK 3 ... PASS
VERDICT: ACCEPTED WITH CAVEATS   Exit: 0
```

Worker-axis cover at world=1 (what the A10 run exercises) also pre-flights green:
```
$ ... --local --local-world 1 --num-workers 2 --files 40
UCVOLUME_SHARDING_OK files=40 world=1 workers=2 units=2 union=40 unique=40
  CHECK 1 ... PASS   CHECK 2 ... N/A   CHECK 3 ... N/A (single rank)
```

The exact-cover checker has teeth (direct test of the inline logic — it must FAIL on a bad
partition, not just PASS on a good one):
```
PASS  correct 2 ranks          {'dup': False, 'missing': [], 'extra': []}
FAIL  duplicate (0 twice)      {'dup': True,  'missing': [1], 'extra': []}
FAIL  gap (1 missing)          {'dup': False, 'missing': [1], 'extra': []}
FAIL  extra (5 not in range)   {'dup': False, 'missing': [], 'extra': [5]}
```

## Claim → evidence (headline results — an empty evidence cell does not ship)

All runs: workspace **fevm-forrest-serverless-stable-2** (profile fevm-forrest-2), 2026-08-26/27, GPU_1xA10.

| Claim | Status | Evidence (run id · quoted line) |
|---|---|---|
| serverless_gpu.data importable on the v5 CLI/torchrun path | ✅ verified **via PYTHONPATH bridge** | 227734973662034: `serverless_gpu=resolved` / `UCVOLUME_SGC_MODULE serverless_gpu.data`. NOT by default: absent from the databricks-ai venv (961855733100717: `serverless_dists=[]`); present on-image in the base python (838281261554839, dist `databricks_serverless_gpu-0.5.22`). Bridge: `PYTHONPATH=/databricks/python3/lib/python3.12/site-packages` onto the torch-having venv. |
| UC volume writable from an AIR job (BR-2) | ✅ verified | 137172830826102: `[rank0] wrote 96 files to /Volumes/…/ucvolume-probe/dataset` |
| Auto-sharding: exact cover across workers, no DistributedSampler | ✅ verified (worker axis, A10) | 227734973662034: `UCVOLUME_SHARDING_OK files=96 world=1 workers=2 units=2 union=96 unique=96` |
| Local caching: yielded path is local /tmp cache, not the /Volumes FUSE dir | ✅ verified | 227734973662034: `UCVOLUME_CACHE_OK sample_cache_parent=/tmp/fslayer_cache_veqzd_wd/Volumes/…` |
| serverless_gpu.data.DataLoader defaults (num_workers=6, prefetch_factor=4, persistent_workers forced) | ✅ verified (from source) | 893478914814084: `DataLoader (self, *args, num_workers:int=6, prefetch_factor:int=4, **kwargs)` + docstring |
| Cross-RANK sharding + num_workers-match invariant (Check 3, COMPLETE sentinel) | ⏳ needs 8xH100 world≥2 | worker axis proven; rank axis is the same `files[global::world*workers]` algorithm (source: run 893478914814084) |

## The interpreter split (and how the PYTHONPATH bridge resolves it) — customer-relevant

On the AIR **CLI/torchrun batch path**, torch and `serverless_gpu` ship in DIFFERENT interpreters,
and neither has both by default:

- **`/opt/databricks-environments/databricks-ai/bin/python`** (what torchrun uses): torch
  2.9.0+cu129, **no** serverless_gpu (`serverless_dists=[]`, run 961855733100717).
- **base python `/databricks/python3`**: `serverless_gpu` v0.5.22 (dist `databricks_serverless_gpu`)
  on-image, **no torch** (run 838281261554839). `serverless_gpu.data` hard-requires torch, so it
  fails to import there.

**Resolution (verified, run 227734973662034):** bridge the base site-packages onto the torch
venv — `PYTHONPATH=/databricks/python3/lib/python3.12/site-packages` before torchrun. The package
is already on the image, so this needs **no pip/egress** — the hermetic-friendly path (matters for
the customer's no-egress posture). The docs' `UCVolumeDataset` examples assume the notebook
`@distributed` runtime where both coexist; this is how you get it on the CLI path the customer uses.

## AIR runs (workspace: fevm-forrest-serverless-stable-2, profile fevm-forrest-2)

- **137172830826102** (A10): crashed at the BR-2 `all_reduce` on a CPU tensor (NCCL can't reduce a
  CPU tensor). Fixed (flag on `cuda`). **Confirmed the volume write works.**
- **157486619679259** (A10): hung ~15 min; **cancelled → shipped NO logs** (AIR skips log flush on
  cancel — never cancel; let the watchdog/timeout fire). Undiagnosable. → added a read watchdog.
- **961855733100717** (A10, nw=2): `serverless_dists=[]` in the databricks-ai venv — serverless_gpu
  not there. ~13 min stall from an unguarded MLflow `log_param` → alarm-guarded `open_mlflow`.
- **838281261554839** (A10, env_probe): mapped the interpreter topology — serverless_gpu v0.5.22 on
  the base python (no torch); databricks-ai venv has torch (no serverless_gpu).
- **893478914814084 / 1105654492853413** (A10, ucvolume_introspect): captured the `UCVolumeDataset`
  signature + `__iter__` source (algorithm `files[rank*nw+wid :: world*nw]`), confirmed direct
  iteration yields 96 + caches to `/tmp/fslayer_cache_*`, and the DataLoader matrix
  (`raw/wrapped × num_workers 1,2 → 96`; `num_workers=0` fails: SGC DataLoader forces
  `prefetch_factor`, needs `num_workers>0`).
- **455033116690072** (A10, nw=2): bridge worked (`serverless_gpu=resolved`) but yielded **0** —
  UCVolumeDataset built immediately after the write hit a UC-FUSE **write→list lag**. → added a
  post-write visibility gate.
- **227734973662034** (A10, nw=2): ✅ **GREEN.** `UCVOLUME_SHARDING_OK … union=96 unique=96` +
  `UCVOLUME_CACHE_OK … /tmp/fslayer_cache_*`. Sharding (worker axis) + caching verified; Check 3
  (cross-rank) N/A at world=1 → ACCEPTED WITH CAVEATS, exit 0. MLflow run
  `3f2d19b472bb49549eb7d01847ec6325`; archived to local store as run
  `ee6d0b194ae74459bbd9d1303072456b` (`experiments/mlflow.db`).

## Still to verify
- **8xH100 world≥2** for cross-RANK exact cover + `UCVOLUME_NWORKERS_OK` + `UCVOLUME_DATALOADER_COMPLETE`
  (needs approval — whole-node H100 spend). Override in `workloads/ucvolume-dataloader.example.yaml`.

## Open questions this probe leaves

- `serverless_gpu.data.DataLoader` documented defaults (num_workers=6, prefetch_factor=4,
  forced `persistent_workers=True`): reported by the run, not yet independently asserted here.
- Cache eviction threshold (`SGC_FSLAYER_MIN_FREE_DISK_BYTES`, docs say ~10% free): untested.
- Delta → Spark Connect → pandas and the Delta→Parquet-on-Volume path are NOT covered by this
  probe (files/unstructured only) — separate follow-up if the customer's tabular path needs it.
