# Multi-language (polyglot) training on AIR — experiment notes

Goal: JVM-based deep-learning training on AIR **without custom Docker** (Beta — out of scope
for regulated customers). Strategy: ship self-contained artifacts through the GA surface
(`command` + code snapshot / UC volumes) instead of container images.

Status 2026-07-22: scaffolded, nothing launched.

## Package choice: DJL (Deep Java Library)

- Actively maintained (AWS), PyTorch engine via JNI → trains on CUDA like the Python stack.
- DL4J: aging, community slowed. Tribuo: classic ML only, no GPU DL. XGBoost4J: later, once
  the JVM path itself is proven.
- Versions in `djl-train/build.gradle` are from memory — **verify latest DJL BOM before building**.

## The ladder

0. **Exec probe** (`workloads/exec-probe.example.yaml` → `probe/probe.sh`) — answers, on a real
   AIR node: snapshot mount exec permissions (noexec?), whether snapshot preserves the +x bit,
   glibc version, JRE/gcc presence on PATH (open-q #6), egress to Maven Central / Adoptium,
   writable+exec status of /tmp. Run this FIRST; everything below branches on it.
1. **Portable JRE + fat jar** (`workloads/djl-train.example.yaml` → `djl-train/run_djl.sh`) —
   fetch Temurin JRE at runtime (or stage in a UC volume if egress fails), run the DJL MNIST
   trainer on GPU. Success = `Engine.getGpuCount() > 0` and a converging training log.
2. **GraalVM native-image** (stretch) — compile the trainer to a single native binary
   (jlink JRE no longer needed). Hard part: JNI reachability metadata for DJL's runtime
   libtorch loading. If it works: the cleanest possible "drop one binary, train" story.
   If not: documented honestly, ladder step 1 remains the recommendation.
3. **Platform integration** — MLflow tracking from the Java client; UC volume data access
   from JVM code; multi-node coordination via injected env vars (no `@distributed` outside
   Python — see docs/03 §4 for what's lost).

## What this is NOT

- Not a supported product pattern — it's the bash escape hatch used deliberately. Before
  recommending to a customer: their security team must weigh unscanned binaries vs Beta Docker.
- Not a Python replacement — docs/03 §4 lists everything the JVM path gives up.

## Findings

(none yet — fill in per run, verified facts move to docs/)
