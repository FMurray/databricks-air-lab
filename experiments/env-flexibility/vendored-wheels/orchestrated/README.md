# Resolve an AIR-aware wheelhouse through Artifactory

This workflow joins two environments:

- AIR knows its installed Python environment and supported wheel tags, but cannot reach Artifactory.
- A classic MLR cluster can reach Artifactory and downloads wheels for the AIR target.

The AIR freeze is a compatibility baseline. A package remains pinned when its AIR version satisfies
the requested dependency graph. When the graph requires another version, the worker removes that pin
from a build-specific effective constraints file and includes the exact replacement in the delta.

## Algorithm

1. **AIR captures the request.** `pip freeze --all` becomes `constraints.txt`; the requested packages
   become `requirements.txt`; optional operator overrides become `overrides.txt`; the interpreter's
   complete `packaging.tags.sys_tags()` list and marker environment become `target_env.json`.
2. **Classic resolves with AIR preferences.** The uv worker copies the AIR freeze to its compile output,
   then runs `uv pip compile` for AIR's Python and platform without passing the freeze as `-c` constraints.
   Existing output pins are uv preferences: compatible AIR versions are retained, while uv backtracks
   across incompatible direct and transitive versions in one solve.
3. **Classic writes effective constraints.** Every AIR package whose resolved version changed is
   removed from a build-specific constraints file. Explicit overrides use the same mechanism but are
   normally unnecessary.
4. **Classic computes a version-aware delta.** A resolved package is reused only when AIR contains the
   same normalized name **and version**. Every missing or different version is downloaded with
   `pip download --no-deps --only-binary` using AIR's target flags.
5. **Classic validates and publishes.** Every downloaded wheel must have at least one tag in AIR's
   supported-tag list. The worker publishes a request-specific wheelhouse, the effective constraints,
   the full resolution lock, an exact hashed delta lock, and `manifest.json`.
6. **AIR verifies offline.** AIR checks that the manifest fingerprint matches the current request,
   resolves with `--no-index`, and verifies every delta wheel hash before printing the install command.

For the motivating case, uv replaces the preferred AIR OpenAI and jiter versions but retains the
preferred AIR NumPy version because it satisfies the new graph. Older parent metadata is part of the
same global solve, so hidden transitive conflicts do not require a second pass or manual override.

## Files

| File | Runs on | Role |
|---|---|---|
| `air_freeze.py` | AIR env-v5 | Captures the request and verifies the finished build offline |
| `mlr_download_driver.py` | Classic MLR 17.3 | `%run`s the canonical resolver with Artifactory access |
| `resolve_worker.py` | Classic MLR 17.3 | Canonical uv/pip engines plus shared download, validation, and publishing |
| `resolve_orchestrator.py` | AIR env-v5 | Performs the full AIR → classic → AIR flow through a Jobs submit |
| `resolve_worker_uv.py` | Classic MLR 17.3 | Selects the uv preference engine in the canonical worker |
| `test_resolve_worker.py` | Local | Unit coverage for conflict traversal and name+version delta logic |
| `test_resolve_worker_integration.py` | Local | Offline freeze → resolve → wheelhouse → install integration test |

There is one wheelhouse implementation. `resolve_worker_uv` selects uv and delegates shared delta,
download, tag validation, locking, and publishing to `resolve_worker`. The classic driver defaults to
uv; select `pip` only when testing the conservative conflict-walk fallback.

## Local offline verification

Run the resolver logic and the full worker flow without public PyPI access:

```bash
python3 -B -m unittest \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker.py \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker_integration.py
```

The integration test creates a temporary private package source and baseline virtual environment. Its
uv case combines the OpenAI/jiter conflict with an older compatible-looking parent whose dependency
metadata conflicts transitively. One uv solve must replace both conflict chains, retain the compatible
array pin, build the exact wheelhouse delta, and resolve it offline. The test deletes all generated
environments, indexes, caches, and wheelhouses when it exits.

## Two-step workflow

Deploy `air_freeze.py`, `mlr_download_driver.py`, and `resolve_worker.py` to one workspace folder so
the driver's `%run ./resolve_worker` resolves.

1. Run Phase 1 of `air_freeze` on the exact AIR interpreter the workload will use.
2. Run `mlr_download_driver` on a classic MLR 17.3 cluster with the same `stage_dir`.
3. Run Phase 2 of `air_freeze`. It refuses stale or failed manifests.

`overrides.txt` is an escape hatch for a proven dependency conflict, not a native-package safety
classification. The manifest records explicit and detected overrides with their dependency reasons.

## One-notebook workflow

Run `resolve_orchestrator.py` on AIR with the target requirements, classic cluster id, and workspace
path of `resolve_worker`. It captures AIR, submits the classic worker, validates its returned manifest,
and runs the same offline checks.

## Runtime installation

Use the exact paths printed by the AIR verification phase. They are scoped to the request fingerprint:

```bash
/opt/databricks-environments/databricks-ai/bin/python -m pip install \
  --no-index --no-deps --require-hashes \
  --find-links /Volumes/<cat>/<schema>/<vol>/vendor-stage/builds/<request-id>/wheelhouse \
  -r /Volumes/<cat>/<schema>/<vol>/vendor-stage/builds/<request-id>/delta.lock
/opt/databricks-environments/databricks-ai/bin/python -m pip check
```

`--no-deps` is deliberate: the classic worker already produced the complete resolution, and
`delta.lock` contains every package AIR must add or replace. Runtime pip installs that exact delta
without making a second dependency choice.

## Boundaries

- uv cross-resolves using AIR's Python version and manylinux platform, so the classic interpreter may
  differ. The pip fallback still requires matching Python and dependency-marker environments.
- Wheel tags prove Python ABI and platform compatibility. They do not prove compatibility with an
  external Torch, CUDA, MPI, or system-library ABI. Native replacements still need an AIR import or
  workload smoke test.
- uv must be available from the classic environment or Artifactory. The worker installs it with pip
  when absent, using the same configured index.
- An explicit `index_url` is passed as a process argument and never printed. Leaving it blank uses the
  classic cluster's pip configuration.
- Builds live under `builds/<request-id>/`; Phase 2 validates the request fingerprint before consuming
  a manifest, preventing a prior effective constraints file or accumulated wheelhouse from being used.

## Existing evidence

The following receipts validate the wheelhouse transport and AIR tag targeting. They predate the
current uv preference rewrite, which still needs an end-to-end Artifactory/AIR verification round.

- AIR env-v5 tags were captured on `fevm-forrest-2` by run `433746246158238`.
- The additive `cowsay` + `numpy-financial` loop resolved on classic run `872088480209335` and verified
  offline on run `430984765532323`.
- The real AIR v5 baseline was captured by job `52417241507965`; classic job `1079255236391640`
  downloaded its delta, and AIR job `810184916255437` resolved it offline.
- Compiled-wheel targeting downloaded `rapidfuzz` and `bitarray` on classic job `332898048582973` and
  installed them offline on AIR v5 job `590004138424053`.
