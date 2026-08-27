# Model I/O — base-model load from a UC Volume (offline) + UC registry round-trip

Verifies the two model-lifecycle patterns the customer needs on the AIR **CLI path** (no
`serverless_gpu`, no distributed): loading base weights from a UC Volume with no hub egress, and
registering a trained model to the Unity Catalog registry. Companion cookbook: none yet (candidate
`docs/cookbook/`). Probe: `model_io_probe.py` · Workload: `workloads/model-io.example.yaml`.

## Claims under test (pre-registered) → evidence

All runs: workspace **fevm-forrest-serverless-stable-2** (profile fevm-forrest-2), 2026-08-27,
GPU_1xA10, single process on `/opt/databricks-environments/databricks-ai/bin/python`.

| # | Sentinel | Claim | Status | Evidence |
|---|---|---|---|---|
| 0 | `MODELIO_VERSIONS` | torch + transformers preinstalled in the databricks-ai env | ✅ verified | 554427571181818: `torch 2.9.0+cu129, transformers 4.57.1` — no `dependencies` needed |
| 1 | `MODELIO_VOLUME_LOAD_OK` | `from_pretrained` loads a base model off a UC Volume with `local_files_only=True` + `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` (no hub egress), logits bit-close to the saved model | ✅ verified | 554427571181818: `max_logit_diff=0.00e+00` from `/Volumes/…/modelio/base-model` |
| 2 | `MODELIO_TMP_STAGE_OK` | same load from a local `/tmp` copy matches (the mmap-over-FUSE-safe staging pattern) | ✅ verified | 554427571181818: `max_logit_diff=0.00e+00` from `/tmp/modelio_staged` |
| 3 | `MODELIO_UC_REGISTER_OK` | log → register to a UC three-level name (`databricks-uc`) → load back via `models:/<name>/<v>`, matching output | ✅ verified | 554427571181818: `model=…airlab_modelio_probe version=1 max_out_diff=0.00e+00` |

Completion `MODELIO_COMPLETE` (all three PASS) → **VERDICT: ACCEPTED**, exit 0.

## Sharp edge (cost a run to find): UC registry REQUIRES a model signature

Run **921050926151845**: Checks 1+2 PASS but Check 3 FAILED —
`MlflowException: Model passed for registration did not contain any signature metadata. All models
in the Unity Catalog must be logged with a model signature containing both input and output type
specifications.` **Fix:** pass `signature=infer_signature(x, y)` (or `input_example=`) to
`log_model`. With the signature, registration + `models:/` load-back succeed (run 554427571181818).
This is a customer-relevant requirement for the governed hand-off.

## Local CPU pre-flight (Checks 1/2; Check 3 skipped without a UC registry)

```
$ uv run --with torch --with transformers python model_io_probe.py --local --volume-dir /tmp/modelio_local
MODELIO_VERSIONS torch 2.13.0, transformers 5.15.1 local=True
MODELIO_VOLUME_LOAD_OK max_logit_diff=0.00e+00 dir=/tmp/modelio_local/base-model
MODELIO_TMP_STAGE_OK   max_logit_diff=0.00e+00 staged=/tmp/modelio_staged
CHECK 1 PASS · CHECK 2 PASS · CHECK 3 SKIPPED (local) · VERDICT ACCEPTED WITH CAVEATS
```

## AIR runs
- **921050926151845** (A10): Checks 1+2 PASS; Check 3 FAIL — UC registry needs a model signature
  (finding above). transformers 4.57.1 confirmed preinstalled.
- **554427571181818** (A10): ✅ all three PASS → `MODELIO_COMPLETE`, ACCEPTED. MLflow run
  `ee71730e6523435db56e0d13f113d226`; archived to the local store as run
  `a1337eb307a644768a3b85e1849c644c` (`experiments/mlflow.db`).

## Caveats / open
- The offline Volume load (Check 1) passed on a TINY safetensors file; the mmap-over-FUSE failure
  mode the docs warn about is size/driver-dependent, so Check 2 (`/tmp` stage) remains the
  recommended pattern for large real base models (Llama/Mistral-class) — not yet stress-tested at
  scale here.
- TabICL-class models load base weights via their own `--checkpoint_dir`, not `from_pretrained` —
  same stage-on-Volume story, different API (see foundation-models/tabicl).
