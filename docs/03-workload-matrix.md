# Heterogeneous Workload × AIR Fit Matrix

How each workload family maps onto AIR's surface, what to test, and where the sharp edges are.

## 1. Foundation model training / fine-tuning

**Fit:** core use case. LoRA/QLoRA/full FT; HF Transformers, DeepSpeed, Composer preinstalled or installable.

| Path | When |
|---|---|
| Notebook + `@distributed(gpus=8)` on 8xH100 | dev/debug, single-node FSDP/DDP/DeepSpeed |
| `air` CLI, `num_accelerators: 16+` | multi-node (only path); torchrun via `command` |
| Managed Model Training product | when custom code isn't needed — know the boundary |

**Sharp edges:** 7-day max runtime → checkpoint/restart discipline; capacity/cross-region fallback;
`gpus=8` required for decorator (no 2- or 4-GPU distributed on a node).

**Customer angle:** the customer's tabular FM (TabICL-class) is *memory-bound* not compute-bound — whole
dataset in GPU memory, attention blow-up OOMs. Test: activation checkpointing / FlashAttention / sequence
sharding on H100 before conceding "needs B300". Also Mistral-class rapid fine-tuning for an ops team.

**Experiments:** `experiments/foundation-models/`

## 2. Reinforcement learning

**Fit:** explicitly recommended; Ray supported; DBRX-AI-research used AIR for RL (KARL).

**Nuances to probe:**
- RL is heterogeneous *within a job*: rollout workers (CPU-heavy or inference GPUs) + learner (GPU).
  AIR nodes are uniform GPU nodes — how do you place a CPU-heavy actor fleet? Likely answer: everything
  on the 8xH100 node or hybrid with classic compute; test what Ray cluster topology is actually possible.
- Long-running training + 7-day cap → checkpoint policies for replay buffers/optimizer state.
- Env simulators with native deps → custom Docker path.

**Experiments:** `experiments/rl/` (start: Ray PPO on single 8xH100 node)

## 3. Classic ML (XGBoost, forecasting, sklearn-scale)

**Fit:** supported and documented (A10 examples: XGBoost, time-series). Real question is *should*:
dev-on-CPU-first discipline is a customer operating-model theme; A10 is the right-size default.

**Sharp edges:**
- **Known bug:** docs sgc-xgboost notebook hangs on H100 (works on A10). Repro + track. (`#ai-runtime-oncall`)
- Spark Connect → pandas conversion is the data path; no Photon/SparkML story here — that stays on classic/serverless CPU.
- Cost discipline: teams default to H100s; utilities should surface "you used 3% of an H100" signals.

**Experiments:** `experiments/classic-ml/`

## 4. Multi-language (Java/JVM, non-Python)

**Fit: not first-class.** Python-only env model and distributed API. Three escape hatches, all CLI-only:

1. `command` is arbitrary bash → anything the base env can run.
2. Custom Docker image (Beta) → bring a JVM + your jar. Constraints: Docker Hub only, <20 GB, no
   `dependencies`/`version` alongside, `WORKDIR` ignored. **Beta status rules this out for
   customers that can't adopt pre-GA features.**
3. **Prebuilt self-contained artifacts via snapshot/UC volume** (GA surface only): portable
   jlink JRE + fat jar, GraalVM native-image binaries, or `zig cc` cross-compiled tools —
   executed from `command`. Depends on exec permissions of the snapshot mount (probe first);
   security tradeoff: unscanned binaries. See `experiments/multi-language/NOTES.md` (the ladder).

**What's lost outside Python:** `@distributed`, `serverless_gpu.data.UCVolumeDataset`, automatic MLflow
integration, Spark Connect ergonomics. A Java DL4J/Spark-based trainer gets raw GPUs + bash, nothing more.

**Test plan (`experiments/multi-language/`):**
- [x] Exec probe ✅ 2026-07-22, run 93215537511850, e2-demo-field-eng: snapshot mount rw+exec,
      +x preserved, glibc 2.39/Ubuntu 24.04, Maven+Adoptium egress 200 (`workloads/exec-probe.example.yaml`;
      receipts in `experiments/multi-language/NOTES.md`)
- [x] JRE present on Standard env (`/usr/bin/java`, same run; version capture added to probe) —
      `java -jar` viable without shipping a JRE, pending version check
- [ ] Portable Temurin JRE + DJL fat jar via snapshot; verify GPU visibility from JVM
      (`workloads/djl-train.example.yaml`)
- [ ] GraalVM native-image of the DJL trainer (stretch — JNI metadata for runtime libtorch loading)
- [ ] UC data access from Java: Spark Connect Scala/Java client? UC volume FUSE path visibility?
- [ ] MLflow tracking from Java client against workspace tracking server
- [ ] Multi-node coordination without `@distributed` — env vars available to `command` for rank/world-size?

**Customer angle:** a customer architect raised Java; use case still unknown — *qualify before engineering*. Plausible honest
answer: "Python-first; JVM via container escape hatch; here's the demo and here's what you give up."

## 5. Batch inference / embedding jobs

Internal positioning includes batch inference on SGC. Same mechanics as training paths; test only if a
customer team asks (they currently serve on edge).

## Cross-cutting test dimensions (apply to every family)

- Data: Delta via Spark Connect vs UC Volumes (+ caching) vs external (Snowflake! — use case 2)
- Submission: notebook / job / DAB / CLI parity
- Observability: MLflow, logs, system tables — what an admin/chargeback view can actually see
- Failure modes: OOM, capacity-unavailable, cross-region fallback, timeout, retry semantics
- Cost: DBU emission per accelerator type; usage_policy attribution granularity
