# Build a serverless AI wheelhouse on classic compute

The developer supplies two values:

1. A `requirements.txt` path.
2. A UC Volume directory for the wheelhouse builds.

Run `build_wheelhouse.py` on a classic cluster that can reach Artifactory. The notebook uses the
checked-in package baseline and platform metadata for the selected serverless AI environment. It
does not run a freeze or resolver notebook on AIR.

## Use it

Deploy this directory to the workspace, attach `build_wheelhouse` to classic compute, and set:

| Widget | Required | Value |
|---|---:|---|
| `requirements_file` | yes | Workspace or Volume path to the developer's file |
| `wheelhouse_volume` | yes | `/Volumes/<catalog>/<schema>/<volume>/<directory>` |
| `air_environment` | no | `databricks_ai_v5` by default |
| `index_url` | no | Explicit Artifactory index; blank uses the cluster's pip configuration |

The notebook prints the generated `environment.yaml`. Apply that path as the custom serverless base
environment. The file uses `air_environment` and `environment_version` from the selected static
profile; for v5 it begins with:

```yaml
base_environment: databricks_ai_v5
environment_version: "5"
```

## Resolution algorithm

1. Start uv with the selected environment's static package pins as preferences.
2. Resolve the developer requirements for that environment's Python version and Linux platform.
3. Keep every compatible baseline version. Let uv replace a pin when the requested graph requires
   another version, including transitive conflicts such as OpenAI requiring a newer jiter.
4. Compare the complete resolved closure with the baseline by normalized package name and version.
5. Download one compatible wheel for every new or changed package from Artifactory.
6. Write the full lock, hashed wheel delta, manifest, and serverless environment YAML to a build
   directory addressed by the resolved lock's hash.

There are no overrides and no package protection list. The baseline influences version selection;
it does not overconstrain the solve. A baseline package appears in the wheelhouse only when the
resolved graph proves that its installed version must change.

## Output

Each resolution produces:

```text
<wheelhouse-volume>/
  builds/<lock-id>/
    requirements.txt       # original developer input
    resolved.lock          # complete resolved dependency closure
    delta.lock             # exact changed/new packages with wheel hashes
    wheelhouse/*.whl       # wheels named by delta.lock
    environment.yaml       # apply this as the custom serverless base environment
    manifest.json          # counts, target, paths, wheel tags, and hashes
  requests/<request-id>.json
```

`environment.yaml` selects the serverless AI base and matching environment version, then replays the
verified offline install against the hashed delta lock: `--no-index --no-deps --require-hashes
--find-links <wheelhouse> -r <delta.lock>`. The classic build has already resolved the complete graph:
new and changed transitive packages are in the delta, while unchanged packages such as Typer come
from the selected AI base. Serverless startup therefore makes no Artifactory or public PyPI request.
`delta.lock` carries the same package versions with SHA-256 hashes for audit and offline verification.

`lock-id` is derived from the selected AIR environment and normalized resolution, so the same
resolution has the same output path even if it came from a different input spelling. `request-id`
also includes the exact requirements and profile files and provides a stable success or failure
receipt for the submitted request.

## Static environment profiles

Profiles live under `profiles/<environment>/`:

- `constraints.txt` is the exact managed environment package baseline.
- `target_env.json` records the Python, ABI, manylinux platforms, and marker environment used for
  cross-resolution and wheel validation.

The checked-in `databricks_ai_v5` profile contains 433 exact package pins from the verified AIR v5
baseline and targets CPython 3.12 on x86_64 manylinux. Add a new profile when the managed serverless
environment changes; application developers should not regenerate it per build.

The v5 baseline was captured on `fevm-forrest-2` on 2026-09-10 by AIR run `52417241507965`.

## Local verification

The integration test creates a private local wheel source. It covers the OpenAI/jiter conflict, a
compatible baseline dependency that uv must retain, a hidden parent conflict that requires
backtracking, hashed offline installation, and `pip check`. It does not contact PyPI.

```bash
python3 -B -m unittest \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker.py \
  experiments/env-flexibility/vendored-wheels/orchestrated/test_resolve_worker_integration.py
```

Wheel tags establish Python ABI and platform compatibility. Packages coupled to external native
libraries such as CUDA, Torch, or MPI still need an import or workload smoke test on the target
runtime.
