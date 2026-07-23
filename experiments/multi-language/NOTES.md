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

### Exec probe ✅ VERIFIED 2026-07-22, run 93215537511850, e2-demo-field-eng
1xA10, env image "4", air CLI v0.1.0, snapshot code source. Raw log: `run-93215537511850.log`.
Scope caveat: measured on e2-demo-field-eng only — the fevm run's output was unretrievable
(see log-capture finding below), so none of these claims are established for fevm nodes yet.

| Claim | Evidence (run-93215537511850.log) | Status |
|---|---|---|
| Snapshot preserves git +x bit; scripts exec in place | `PROBE:snapshot_script_exec=ok` (no chmod needed) | measured |
| ELF binaries exec from snapshot mount (rw overlay, no noexec) | `PROBE:snapshot_binary_exec=ok`, `PROBE:snapshot_mount=rw,relatime,…overlaybd…` | measured |
| /tmp allows binary exec | `PROBE:tmp_binary_exec=ok` | measured |
| **JRE already present on standard env** (open-q #6) | `PROBE:java=/usr/bin/java` — version not captured, probe updated to grab it next run | measured (presence only) |
| gcc present (open-q #6) | `PROBE:gcc=/usr/bin/gcc` | measured (presence only) |
| Target ABI: Ubuntu 24.04.4, glibc 2.39 | `PROBE:os=…24.04.4 LTS`, `PROBE:glibc=…2.39` | measured |
| Egress to Maven Central + Adoptium | `PROBE:egress:…maven…=200`, `…adoptium…=200` | measured |
| 16 vCPU on the 1xA10 node shape | `PROBE:nproc=16` | measured |

Ladder consequence: step 0 fully green on field-eng — step 1 (DJL) is unblocked, and a system
JRE may make the portable-JRE download unnecessary (branch on `PROBE:java_version` next run).

### Log-capture gap on fevm-forrest-aws-stable — run 938962751074433, 2026-07-22
Identical YAML, same day, air v0.1.0. Job `SUCCESS`, 62s execution, MLflow **system metrics
delivered** (CPU/disk/GPU gauges present on run fdbe1f21401a403ba65eb72eadfe3c08) — but stdout
is unretrievable by all three documented channels: `air logs` streaming ("No logs available"),
`air logs --download-to` (hangs), MLflow artifacts (empty ~40 min post-run), Jobs API
(metadata only for gen_ai_compute_task). Raw submission log: `run-938962751074433.log`.
Control: the same YAML on e2-demo-field-eng streamed live and landed logs (run above), and the
2026-07-16 Docker-path baseline (run 37776040541298) still has its `logs/` artifacts.
**Inferred**: workspace-specific log-capture defect on fevm (snapshot path exonerated by the
control; alternative not excluded: a fevm↔field-eng rollout-version difference that will also
hit other new workspaces — worth #ai-runtime-oncall either way, [the customer] UAT depends on log
delivery). Status: reported to nobody yet — escalate.
