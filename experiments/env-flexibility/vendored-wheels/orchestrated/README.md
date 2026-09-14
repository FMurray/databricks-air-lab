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
2. **Classic tries the AIR baseline.** Pip resolves the target requirements with the AIR constraints
   and AIR's Python/ABI/platform flags. A successful resolve means no baseline pin conflicts.
3. **Classic detects required replacements after a failure.** It resolves the requested graph without
   the AIR constraints to obtain candidate metadata, then walks only packages that must be added or
   replaced. An AIR pin is removed only when its version fails an actual incoming requirement
   specifier. A newer unconstrained preference alone is not a conflict.
4. **Classic resolves the effective environment.** It retries with the AIR freeze minus directly
   detected and explicitly named overrides. If an older retained AIR parent still makes that graph
   fail, the worker temporarily relaxes every AIR pin that differs from the valid unconstrained graph,
   then restores compatible pins in groups. Pins that fail restoration become automatic overrides.
5. **Classic computes a version-aware delta.** A resolved package is reused only when AIR contains the
   same normalized name **and version**. Every missing or different version is downloaded with
   `pip download --no-deps --only-binary` using AIR's target flags.
6. **Classic validates and publishes.** Every downloaded wheel must have at least one tag in AIR's
   supported-tag list. The worker publishes a request-specific wheelhouse, the effective constraints,
   the full resolution lock, an exact hashed delta lock, and `manifest.json`.
7. **AIR verifies offline.** AIR checks that the manifest fingerprint matches the current request,
   resolves with `--no-index`, and verifies every delta wheel hash before printing the install command.

For the motivating case, requesting a newer OpenAI version marks OpenAI as an override. Walking that
candidate's metadata finds its `jiter` requirement; when AIR's installed `jiter` does not satisfy the
specifier, `jiter` also becomes an override and enters the name+version delta. A compatible NumPy
version remains pinned even when unconstrained pip happens to prefer a newer NumPy.

## Files

| File | Runs on | Role |
|---|---|---|
| `air_freeze.py` | AIR env-v5 | Captures the request and verifies the finished build offline |
| `mlr_download_driver.py` | Classic MLR 17.3 | `%run`s the canonical resolver with Artifactory access |
| `resolve_worker.py` | Classic MLR 17.3 | Detects conflicts, resolves, downloads, validates, and publishes |
| `resolve_orchestrator.py` | AIR env-v5 | Performs the full AIR → classic → AIR flow through a Jobs submit |
| `resolve_worker_uv.py` | Classic MLR 17.3 | Compatibility alias to `resolve_worker`; retained for old notebooks |
| `test_resolve_worker.py` | Local | Unit coverage for conflict traversal and name+version delta logic |
| `test_resolve_worker_integration.py` | Local | Offline freeze → resolve → wheelhouse → install integration test |

There is one resolver implementation. The earlier uv variant duplicated the algorithm and retained
name-only delta behavior, so its path now delegates to `resolve_worker`.

## Local offline verification

Run the resolver logic and the full worker flow without public PyPI access:

```bash
python3 -B -m unittest \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker.py \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker_integration.py
```

The integration test creates a temporary `file://` package index and a temporary baseline virtual
environment. It installs and freezes `airlab-openai==1.0.0`, `airlab-jiter==0.8.0`, and
`airlab-array==2.1.3`, then requests `airlab-openai==2.0.0`. The new parent requires
`airlab-jiter>=0.10,<1`; the worker must remove the OpenAI and jiter pins, download their exact
replacement wheels, keep the compatible array pin, install the hashed delta offline, and pass
`pip check`. The test deletes all generated environments, indexes, and wheelhouses when it exits.

## Two-step workflow

Deploy `air_freeze.py`, `mlr_download_driver.py`, and `resolve_worker.py` to one workspace folder so
the driver's `%run ./resolve_worker` resolves.

1. Run Phase 1 of `air_freeze` on the exact AIR interpreter the workload will use.
2. Run `mlr_download_driver` on a classic MLR 17.3 cluster with the same `stage_dir`.
3. If the worker cannot infer a remaining conflict, enter package **names only** in the driver's
   `overrides` widget and rerun. The driver updates the staged `overrides.txt`.
4. Run Phase 2 of `air_freeze`. It refuses stale or failed manifests.

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

- The classic resolver's Python major/minor must match AIR. Pip's target flags control wheel and
  `Requires-Python` selection, while some environment-marker behavior still follows the resolver
  interpreter. The verified MLR 17.3/AIR v5 pairing is CPython 3.12 on Linux x86_64.
- Wheel tags prove Python ABI and platform compatibility. They do not prove compatibility with an
  external Torch, CUDA, MPI, or system-library ABI. Native replacements still need an AIR import or
  workload smoke test.
- Requirements-file options and nested `-r` files are passed to pip, but automatic conflict traversal
  handles ordinary PEP 508 requirement lines. Use explicit overrides when the manifest reports skipped
  requirement lines.
- An explicit `index_url` is passed as a process argument and never printed. Leaving it blank uses the
  classic cluster's pip configuration.
- Builds live under `builds/<request-id>/`; Phase 2 validates the request fingerprint before consuming
  a manifest, preventing a prior effective constraints file or accumulated wheelhouse from being used.

## Existing evidence

The following receipts validate the wheelhouse transport and AIR tag targeting. They predate the
current conflict-walk rewrite, which still needs an end-to-end OpenAI/jiter verification round.

- AIR env-v5 tags were captured on `fevm-forrest-2` by run `433746246158238`.
- The additive `cowsay` + `numpy-financial` loop resolved on classic run `872088480209335` and verified
  offline on run `430984765532323`.
- The real AIR v5 baseline was captured by job `52417241507965`; classic job `1079255236391640`
  downloaded its delta, and AIR job `810184916255437` resolved it offline.
- Compiled-wheel targeting downloaded `rapidfuzz` and `bitarray` on classic job `332898048582973` and
  installed them offline on AIR v5 job `590004138424053`.
