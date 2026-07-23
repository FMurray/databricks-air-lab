# Run non-Python workloads

Goal: run JVM (or any non-Python) training on AIR — and be honest about what that costs you.

AIR is **Python-first, full stop**: the env model is pip/uv, the distributed API is Python-only.
There is no first-class Java/Scala surface. But `command` is arbitrary bash, and that's a real
escape hatch.

!!! note "Status: scaffolded, not yet verified"
    Unlike most of this cookbook, this ladder hasn't produced receipts yet (step 0 not yet run as
    of 2026-07-22). Treat it as a tested *plan*, not tested *facts* — and run the probe before
    believing anything below it. Progress: `experiments/multi-language/NOTES.md`.

## Three escape hatches, in order of preference

| Path | Surface | Catch |
|---|---|---|
| 1. Self-contained artifacts via snapshot/UC volume + bash `command` | **GA** | unscanned binaries — security must sign off |
| 2. [Custom Docker image](use-a-custom-docker-image.md) with a JVM | Beta | pre-GA; Docker Hub only |
| 3. Whatever the managed env has on PATH | GA | probably nothing you need (probe answers this) |

## Step 0: probe the node first

Everything branches on facts about the node that only a run can answer — snapshot mount exec
permissions, whether +x survives snapshot, glibc version, JRE/gcc on PATH, egress to Maven
Central/Adoptium:

```bash
cp workloads/exec-probe.example.yaml workloads/exec-probe.yaml
air run -f workloads/exec-probe.yaml -p <profile> --watch    # A10, minutes, cheap by design
```

## Step 1: portable JRE + fat jar

Ship a Temurin jlink JRE and a DJL fat jar through the snapshot, run
`java -jar` from `command`. Template: `workloads/djl-train.example.yaml`. Success criterion:
`Engine.getGpuCount() > 0` plus a converging training log — not just exit 0.

## What you give up outside Python

No `@distributed`, no `UCVolumeDataset`, no automatic MLflow integration, no Spark Connect
ergonomics. A JVM trainer gets **raw GPUs + bash + injected env vars** — multi-node coordination
means reading `NODE_RANK`/`MASTER_ADDR` yourself
([the injected contract](run-multi-node-training.md#what-the-platform-injects-so-torchrun-just-works)).

Before recommending this to any customer: it is a deliberate use of the bash escape hatch, not a
supported product pattern. Their security team must weigh unscanned binaries against Beta Docker.
