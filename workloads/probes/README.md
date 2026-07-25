# Diagnostic probes (AIR CLI workloads)

Tiny, cheap (1×A10 or less), single-purpose workloads used to isolate platform behavior.
Run: `air run --file workloads/probes/<probe>.yaml -p <profile>`. Every probe here backs a
receipt in `docs/06-uat-suite.md` — keep them runnable; they're the re-verification suite
after any workspace-side fix.

| Probe | Question it answers | Healthy signal |
|---|---|---|
| `fail-probe.yaml` | what does user-code `exit 1` look like in run state? | INTERNAL_ERROR (same as platform errors — that's the finding) |
| `python-probe.yaml` | does python + torch + CUDA work at all? | SUCCESS |
| `envvar-probe.yaml` | do `env_variables` reach the container? | SUCCESS |
| `snapshot-python-probe.yaml` | does the code snapshot mount + $CODE_SOURCE_PATH resolve? | SUCCESS |
| `pip-probe.yaml` | can env build install one trivial pip dep? | SUCCESS (fails on no-PyPI workspaces — by design there) |
| `artifact-probe.yaml` | can the GPU node upload an MLflow artifact? (receipts via params) | param `artifact_upload=OK` |
| `cli-egress-probe.example.yaml` | full egress matrix from a CLI Gen-AI task: TCP/upload/PyPI (params) | all params OK; ⚠️ cancel run after `probe_done=yes` — hung log-shipping can outlive timeout_minutes |
| `burn-debug.yaml` | gpu-burn wrapped in self-logging (stdout → MLflow artifact) | SUCCESS + burn-out.log artifact |
| `gpu-burn-nodeps.yaml` | gpu-burn without pip deps (deps-vs-code failure isolation) | SUCCESS |

NB: `root_path` in `code_source` is relative to the YAML's own directory → probes in this
subdir use `../..` where `workloads/*.yaml` use `..`.
