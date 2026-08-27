# Classic ML on AIR

## XGBoost single-GPU training (docs sgc-xgboost, hardened for AIR)

End-to-end GPU training example for the enablement session, modeled on the docs tutorial
`.../ai-runtime/examples/tutorials/sgc-xgboost` (California Housing regression, `tree_method="hist"`,
`device="cuda"`, `TrainingCheckPoint` → UC Volume). Hardened for serverless GPU: reads the dataset
from Unity Catalog (`air_lab.california_housing`) instead of `fetch_california_housing()` (external
egress is blocked on GPU jobs), and prints pass-gated sentinels.

- **Notebook:** `experiments/classic-ml/xgboost_gpu_train.py` (Databricks source; workspace copy
  `/Users/forrest.murray@databricks.com/air-enablement/xgboost_gpu_train`)
- **Run path:** one-time serverless-GPU notebook job via `/api/2.2/jobs/runs/submit`
  (`compute.hardware_accelerator=GPU_1xA10`, env v5, deps `xgboost`+`scikit-learn`).
- **Dataset:** `forrest_serverless_stable_2_catalog.air_lab.california_housing` (20,640 rows,
  8 features + `MedHouseVal`), staged from sklearn/StatLib.
- **Shape:** `GPU_1xA10` — the docs notebook is known to **hang on H100, complete on A10**
  (`workloads/xgboost-gpu.example.yaml`); A10 only here.

### Success criteria (pre-registered, before submit)
- `XGB_GPU_TRAIN_OK` prints (pass-gated: requires `cuda_available` + `USE_CUDA` +
  `test_rmse < 0.6` + ≥1 checkpoint written).
- `RESULT ... USE_CUDA=True cuda_available=True device=<A10>` — proves GPU actually used.
- `PHASE fit_done` with checkpoints `model_50/100/150.ubj` present in the Volume.
- Checkpoint reload of ~round 150 scores like a mid-training model.
- Notebook exits JSON with `verdict=ACCEPTED`.

### Local CPU pre-flight (harness, device=cpu) — 2026-08-27
`uv run --with xgboost --with scikit-learn --with pandas --with pyarrow python preflight_xgb.py`
(local; DYLD_LIBRARY_PATH=libomp). Validates the code path before GPU spend:
```
xgboost 3.4.1 | USE_CUDA False
[199]	train-rmse:0.30451	eval-rmse:0.45643
checkpoints written: ['model_100.ubj', 'model_150.ubj', 'model_50.ubj']
test_rmse=0.4564  fit_s=0.5
reloaded model_150.ubj resumed_rmse=0.4624
PREFLIGHT_OK
```

### Observed (AIR GPU run)
✅ **ACCEPTED** 2026-08-27 · Job Run ID `280502658322853` · MLflow run `3691bc2e6b974e42b535a40b9393c41a`
· fevm-forrest-serverless-stable-2 · GPU_1xA10 · via `air run -f workloads/xgboost-gpu-ca.yaml`.
Archived to local store (`experiments/mlflow.db`).

```
RESULT xgboost_version=3.4.1 USE_CUDA=True gpu_count=1 device=NVIDIA A10G
PHASE data_ready rows=20640 features=8 train=16512 test=4128 src=/Volumes/.../california_housing.parquet
[199]	train-rmse:0.30159	eval-rmse:0.46050
PHASE fit_done seconds=1.5 checkpoints=['model_100.ubj', 'model_150.ubj', 'model_50.ubj']
RESULT test_rmse=0.4605
RESULT checkpoint=model_150.ubj round=150 resumed_rmse=0.4665
XGB_GPU_TRAIN_OK
VERDICT: ACCEPTED rmse=0.4605 gpu=NVIDIA A10G checkpoints=3 fit_s=1.5
```
Checkpoints independently confirmed in the Volume: `xgb_checkpoints/california_housing/model_{50,100,150}.ubj`.
GPU training (device="cuda") of 20,640 rows on an A10 completes in **1.5 s** — the GPU barely warms;
the acceleration story needs large/wide data (a talking point, not a defect).

### Scheduled-job path — VERIFIED on fevm (this is the deployment path)
✅ 2026-08-27 · job `982054610302014` run `1002087777249789` (task run 947257110132349) · TERMINATED
SUCCESS. Config = the **Standard** serverless env + a GPU (exactly how a working user job here is set
up): `environments[].spec.environment_version:"5"` + task `compute.hardware_accelerator:"GPU_1xA10"` —
**no `base_environment` / AI env needed.** Notebook exit JSON:
`{"verdict":"ACCEPTED","device":"NVIDIA A10G","use_cuda":true,"test_rmse":0.4605,"resume_rmse":0.4665,`
`"fit_seconds":0.9,"checkpoints":["model_50/100/150.ubj"],"data_source":"spark.read.table(…california_housing)"}`

Settles two things: (a) **`spark.read.table` works** on the Standard serverless GPU job — it read the
Delta table directly, not the parquet fallback; (b) the ONLY fixes needed were a `%pip install xgboost
scikit-learn` cell (Standard env ships neither, and the Environments panel is unsupported for scheduled
jobs) + a **torch-free GPU gate** (`nvidia-smi` — Standard env has no torch; xgboost trains on the A10
without it). The earlier "can't run as a job / AI-env gated" framing was a distraction — the GPU
attached the whole time.

### Findings — how to (and how NOT to) run a GPU workload from a repo checkout
Two runners failed before the CLI path worked; both are reusable findings:
1. **AI Runtime scheduled jobs ARE a documented feature — but the managed AI base env is gated on
   THIS workspace.** The docs (`/ai-runtime/connecting`) prescribe, for a scheduled AI v5 job:
   `environments[].spec.base_environment: databricks_ai_v5` (in place of `environment_version`) +
   task `compute.hardware_accelerator: GPU_1xA10`, with **dependencies installed via `%pip` in the
   notebook** (the Environments panel is explicitly *not* supported for AI Runtime scheduled jobs).
   On `fevm-forrest-serverless-stable-2` (2026-08-27) every form of selecting that managed env for a
   `runs/submit` job is refused — so a job here lands on the **bare** serverless interpreter (a GPU
   attaches, but no torch / `serverless_gpu`):

   | environments.spec | result (receipt) |
   |---|---|
   | `environment_version:"5"` (+ pip deps) | standard serverless env + GPU; xgboost via deps, **no torch** — job 867436942982753 (my notebook's `torch.cuda` gate tripped) |
   | `base_environment:"databricks_ai_v5"` (inline, the documented form) | ❌ rejected at submit: *"Only custom base environments (…​.yaml/.yml paths)…"* |
   | `base_environment:"/Workspace/…/base-env-ai-v5.yaml"` (→ `databricks_ai_v5`) | ❌ accepted at submit, **fails at env build**: *"missing or invalid environment version"* — job 672789687485610 |

   ⇒ **But you don't need the AI env for this workload.** The working scheduled-job recipe (verified
   above) is `environment_version:"5"` + `hardware_accelerator:"GPU_1xA10"` + `%pip install` in the
   notebook + a torch-free GPU gate. The managed AI base env only matters if the code needs torch /
   `serverless_gpu` directly; for that, on this ws, use the `air` CLI or interactive notebook until the
   AI-base-env-for-jobs gate is lifted. Earlier claim "serverless-GPU = air CLI only / interactive-only"
   was **wrong** — retracted.
2. **No torch on the CLI default interpreter.** First CLI run (868984212366707) printed
   `USE_CUDA=True cuda_available=False device=torch-unavailable` and failed: plain `command: python`
   uses the default interpreter, which has `databricks.serverless_gpu` but **not torch** (torch is in
   the `databricks-ai` venv — see [[serverless-gpu-cli-pythonpath-bridge]]). xgboost bundles its own
   CUDA, so the fix was to gate the GPU on `nvidia-smi` + `xgb.build_info()["USE_CUDA"]`, not
   `torch.cuda`. The `xgboost_gpu_train` **notebook** keeps the `torch.cuda` check — valid there
   because the interactive AI runtime has torch; the CLI **script twin** uses the nvidia-smi check.

## Related
- `xgboost_gpu_repro.py` — the H100-hang UAT repro (synthetic `make_classification`, CLI path).
- `load_training_data_demo` notebook — the Delta vs UC-Volume data-loading companion.
