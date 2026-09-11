# Orchestrated delta-resolve: vendor only the wheels the AIR env is missing

Makes dependency resolution **seamless across two environments**: the serverless / AIR env-v5
notebook (knows what's installed, can't reach Artifactory) and a classic MLR 17.3 cluster (can
reach Artifactory, doesn't know the AIR env).

## Why not just `pip download -r requirements.txt` on the classic side?

It resolves the **entire** closure — including packages already baked into the AI v5 image (torch,
CUDA libs, numpy, …). If any baked-in transitive dep has no wheel on Artifactory
(`--only-binary=:all:`), the download fails with *"no matching distribution"* even though the target
already has it. The classic resolver has no idea what serverless provides.

## The fix — resolve against the target's installed set, download only the delta

1. **AIR** `pip freeze`s its installed set → a **constraints file**.
2. **Classic** resolves the full closure *bounded by those constraints*, subtracts what the target
   already has, and downloads **only the delta**. Baked-in packages (incl. any with no Artifactory
   wheel) are in the constraints, so never in the delta, never fetched. A genuine version conflict
   surfaces here as a clean resolver error, not a runtime failure on AIR.
3. **AIR** proves it with an offline `--dry-run` install.

## Files

| File | Runs on | Role |
|---|---|---|
| `air_freeze.py` | serverless / AIR env-v5 | Phase 1 stages `constraints/requirements/target_env`; Phase 2 verifies offline |
| `mlr_download_driver.py` | classic MLR 17.3 | sets `stage_dir`, `%run`s the worker, prints the manifest |
| `resolve_worker.py` | classic MLR 17.3 | the delta resolve + download (dual-mode: `%run` **or** Jobs task) |
| `resolve_worker_uv.py` | classic (any Python) | *uv variant* of the worker — `uv pip compile` resolve + `pip download` fetch |
| `resolve_orchestrator.py` | serverless / AIR | *alternative*: auto-submits the worker as a Jobs run (no `%run`; needs classic cluster id) |

### pip vs uv worker

Both are drop-in for the driver — point `mlr_download_driver`'s `%run` at `./resolve_worker` (pip)
or `./resolve_worker_uv` (uv). They produce the same wheelhouse + manifest, so `air_freeze` Phase 2
verifies either identically.

- **`resolve_worker` (pip):** resolves natively with `pip install --dry-run --report`, so the
  classic host's Python should match AIR's (it warns otherwise). No extra tooling.
- **`resolve_worker_uv` (uv):** resolves with `uv pip compile --python-version --python-platform`,
  which cross-resolves from metadata — **the classic host's Python need not match AIR's.** Faster,
  and bootstraps uv via pip if absent. uv has **no download** command, so the delta *fetch* still
  uses `pip download` (targeted to AIR's tags); a fully-uv, pip-free path is
  `uv pip install --target <dir> --no-deps …` (unpacked + `PYTHONPATH`, matching the parent dir's
  `vendor_deps.sh`) if you don't need a wheelhouse.

## Which tool produces the wheelhouse — `pip download` vs `pip wheel` vs uv

The wheelhouse (the `.whl` files that the offline `pip install --no-index --find-links` consumes) can
be produced three ways. We use **`pip download`** because it's the only one that can target AIR's tags
explicitly from a different host.

| Tool | Cross-target to AIR? | Builds sdist-only pkgs? | Notes |
|---|---|---|---|
| `pip download --only-binary=:all: --platform … --abi … --python-version …` | **yes** — why we use it | no (refuses to build) | fetches official prebuilt wheels for AIR's tags, from any host |
| `pip wheel --no-deps --wheel-dir …` | **no** — has no `--platform`/`--abi`/`--python-version` | **yes** (compiles on the host) | builds/collects for the *running* interpreter only; correct only when host == AIR |
| uv | — | — | **no wheelhouse command exists** (see below) |

**Why not `pip wheel`?** It's the pip docs' canonical wheelhouse builder, but it produces wheels for
the interpreter/platform it *runs on* and has **no** `--platform` / `--abi` / `--python-version`
flags (verified: those exist only on `pip download` / `pip install --target`). So it cannot be told
to target AIR. Since the requirement is "produce wheels for AIR specifically," `pip download
--platform` is the right primitive. `pip wheel` is viable *only* by trusting that the classic host
already equals AIR — verified true here (MLR 17.3 == AIR v5: cp312 / glibc 2.39 / x86_64) — and it
carries one advantage: it **builds sdist-only packages into wheels**, which `--only-binary=:all:`
rejects. But it builds C-extensions with the host's toolchain/system libs, which can differ from the
official manylinux wheels `pip download` fetches, so prefer `pip download` unless you specifically
need to cover an sdist-only dependency.

**uv has no equivalent of `pip wheel`.** There is no `uv pip wheel` and no `uv pip download`.
`uv build` builds a *single project* into a wheel/sdist (like `python -m build`), not a dependency
closure. uv's role in this pipeline is the **resolve** (`uv pip compile`, which cross-targets
cleanly); the wheelhouse itself is always produced by pip. uv's own offline story is different —
`uv pip compile --generate-hashes` + `uv pip sync --offline` against a *pre-warmed uv cache* — which
carries a cache, not a portable directory of wheels.

## Run it (two-step)

Deploy all three notebooks to **one workspace folder** (so `%run ./resolve_worker` resolves).

1. **AIR notebook** `air_freeze` → set `stage_dir` + `requirements`, run **Phase 1**.
2. **Classic notebook** `mlr_download_driver` on the 17.3 cluster → set the same `stage_dir`, run
   all cells. The `%run` executes the worker on this cluster; `wheelhouse/` fills up.
3. Back in `air_freeze`, run **Phase 2** → PASS = the vendored set resolves offline.

## Then use it in a workload

```yaml
environment:
  version: "5"
  dependencies: []
command: |
  pip install --no-index --find-links /Volumes/<cat>/<schema>/<vol>/vendor-stage/wheelhouse \
    -r /Volumes/<cat>/<schema>/<vol>/vendor-stage/requirements.txt \
    -c /Volumes/<cat>/<schema>/<vol>/vendor-stage/constraints.txt
  python $CODE_SOURCE_PATH/your_script.py
```

## Sharp edges

- **Two-workspace setups** (AIR serverless in one workspace, classic MLR in another — the common
  case when your AIR workspace is serverless-only): the `stage_dir` UC Volume must be reachable from
  **both** (same metastore + catalog bound to both workspaces). If UC isn't shared, stage on the
  classic side and copy the wheelhouse back with `databricks fs cp`.
- **Wheels are targeted to AIR explicitly.** The worker pins `--implementation cp --python-version
  --abi --platform` (every x86_64 manylinux baseline) from `target_env.json`, so downloaded wheels
  match AIR's interpreter / ABI / platform regardless of what the classic host runs. The host's
  Python should still match AIR (cp312 today, per `workloads/probes/wheel-tags-probe.yaml`) so the
  *resolve* step picks target-correct versions — the worker warns on mismatch. glibc is a non-issue:
  AIR v5 is glibc 2.39, newer than any classic runtime, so wheels flow forward.
- **`failures` in the manifest = real gaps**: a genuinely-new package with no fetchable wheel. Build
  it from sdist on the classic side, or find a wheel — it isn't something the target already has.
- **The wheelhouse install replaces the PYTHONPATH-unpack approach** in the parent dir's
  `vendor_deps.sh` / cookbook — more robust for full environments (entry points, namespace packages,
  `.pth` files).

## Verification status

- Delta logic (report parse → PEP 503 canonicalization → subtract target → download only new) is
  unit-verified offline against a synthetic pip `--report` payload (2026-09-09).
- AIR env-v5 tags (cp312 / glibc 2.39, both interpreters) verified on `fevm-forrest-2` via
  `workloads/probes/wheel-tags-probe.yaml`, run `433746246158238`.
- **Full loop verified end-to-end** on `fevm-classic-stable-1fe1nl` (profile `f-classic`),
  2026-09-10, classic 17.3 ML cluster `0910-163319-6a6r9khs` (c5d.xlarge, cp312/glibc 2.39 — tags
  identical to AIR v5), staging via the shared UC volume `forrest_fdm.airlab.deps`:
  - freeze (run `195301066191917`): 414-package constraints captured from the classic env.
  - deployed driver `%run` worker (run `872088480209335`): requirements `cowsay`, `numpy-financial`
    → closure of **3**, delta of **2** — `numpy` was in the closure but **skipped** as baked-in,
    exactly the intended behavior; both delta wheels downloaded from PyPI, zero failures.
  - offline verify (run `430984765532323`): `--no-index` install resolves — `numpy` satisfied by
    the env, only `cowsay` + `numpy-financial` would install. **seamless = True.**
- **Cross-env against the REAL AIR v5 target** on the same workspace, 2026-09-10:
  - freeze re-run on the actual AI v5 env via `air run` (job `52417241507965`, interpreter `$AIPY`)
    → 433-package constraints including `torch==2.9.0+cu129` (a CUDA build absent from public PyPI)
    and `numpy==2.1.3`.
  - classic delta-download against those AIR constraints (deployed driver, job `1079255236391640`):
    closure 3 → delta 2, **AIR's `numpy` skipped as baked-in**, `cowsay` + `numpy-financial` fetched.
    (The job was marked INTERNAL_ERROR by a "driver is lost" cluster hiccup *after* the worker had
    already written the wheels + manifest — the work completed; not a code fault.)
  - offline verify back on AIR via `air run` (job `810184916255437`): `numpy` satisfied from
    `/opt/databricks-environments/databricks-ai/.../site-packages (2.1.3)`, only the delta would
    install, **rc 0**.
  - Only remaining stand-in: public **PyPI** in place of Artifactory (the index is interchangeable;
    note a requirement that pulls a non-PyPI baked-in build like `torch==…+cu129` into its closure
    needs that build present on the index — the customer's Artifactory mirror provides it).
- **Platform-targeting proven for compiled wheels**, 2026-09-11: worker now pins AIR's tags
  explicitly. Requirements `rapidfuzz` + `bitarray` (both compiled, both absent from AIR's 433) →
  downloaded on classic as `rapidfuzz-3.14.6-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28…` and
  `bitarray-3.11.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17…manylinux_2_28…` (job
  `332898048582973`), then **installed offline on AIR v5** (`air run`, job `590004138424053`, rc 0).
