"""Model I/O probe — base-model load from a UC Volume (offline) + UC model-registry round-trip.

Two customer-relevant patterns on the AIR CLI path (single process; no serverless_gpu, no
distributed). Each behind a sentinel unreachable unless its assertion held:

  1. MODELIO_VOLUME_LOAD_OK  — transformers `from_pretrained` loads a base model staged on a UC
     Volume with `local_files_only=True` + HF offline env (NO hub egress), producing logits
     bit-close to the saved model. This is the no-egress base-model pattern for the customer
     (stage weights on a Volume; the docs' hub `from_pretrained` isn't reachable there).
  2. MODELIO_TMP_STAGE_OK    — the SAME load from a local /tmp copy of the Volume model matches.
     This is the mmap-over-FUSE-safe staging pattern (safetensors mmap on a FUSE mount is
     unreliable; stage to local disk once, then load).
  3. MODELIO_UC_REGISTER_OK  — log a model to MLflow, register it under a UC three-level name via
     the `databricks-uc` registry, load it back via `models:/<cat.sch.name>/<version>`, matching
     output. The governed hand-off path.

Self-contained + egress-free: builds a tiny GPT2 (Check 1/2) and a tiny torch model (Check 3)
in-code; no downloads. Runs on the databricks-ai python (torch + transformers + mlflow
preinstalled). transformers absent ⇒ Check 1/2 record BLOCKED (not a crash); Check 3 still runs.

Pre-flight (local, CPU, transformers only — Check 3 skipped without a UC registry):
    python3 model_io_probe.py --local --volume-dir /tmp/modelio_local
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import textwrap
import traceback as _tb
from dataclasses import dataclass
from datetime import datetime, timezone

import torch

# ==========================================================================================
# Acceptance report — COPIED VERBATIM from .claude/skills/acceptance-report/references/renderer.py.
# Do NOT import (each AIR YAML snapshots only its own dir). Only WORKLOAD, the checks, and the call
# site are workload-specific.
# ==========================================================================================
WORKLOAD = "MODEL I/O (load + register)"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
SKIPPED = "SKIPPED"
NA = "N/A-at-this-scale"


@dataclass
class Check:
    name: str
    status: str
    measured: str
    threshold: str
    what_why: str
    sufficient: str
    likely_means: str = ""
    traceback: str = ""


def _fail_from_exc(name, threshold, what_why, likely_means, exc) -> Check:
    return Check(name=name, status=FAIL, measured=f"raised {type(exc).__name__}: {exc}",
                 threshold=threshold, what_why=what_why,
                 sufficient="A raised exception means the property could not be established.",
                 likely_means=likely_means, traceback="".join(_tb.format_exception(exc)))


def _wrap(text: str, indent: str = "               ") -> str:
    return textwrap.fill(text, width=96, initial_indent="", subsequent_indent=indent)


def _receipt(checks: "list[Check]", verdict: str, exit_code: int, test_id: str) -> None:
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    signal.alarm(120)
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.log_param(run_id, "acceptance_verdict", verdict)
        client.log_param(run_id, "acceptance_exit", exit_code)
        if test_id:
            client.log_param(run_id, "acceptance_test_id", test_id)
        for i, c in enumerate(checks, 1):
            client.log_param(run_id, f"acceptance_check_{i}", f"{c.status} — {c.name}"[:490])
    except Exception as e:                                 # noqa: BLE001 — receipt is best-effort
        print(f"acceptance receipt logging FAILED: {e}", flush=True)
    finally:
        signal.alarm(0)


def render_report(checks: "list[Check]", run_id: str, profile: str, shape: str,
                  scope: str, runtime: str, sentinels: str, test_id: str = "") -> int:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    W = 70
    out = []
    out.append("=" * 20 + f" {WORKLOAD} ACCEPTANCE REPORT " + "=" * 20)
    out.append(f"Run {run_id}   Profile {profile}   Shape {shape} ( {scope} )")
    out.append(f"Runtime {runtime}   When {when}")
    out.append("")
    out.append(_wrap("Attests to what rank 0 observed. On multi-node the CLI streams node 0 "
                     "only (`air logs <id> --node N`). If this report is absent, treat it as a "
                     "failure.", indent="  "))
    out.append("-" * W)

    has_fail = False
    for i, c in enumerate(checks, 1):
        if c.status == FAIL:
            has_fail = True
        out.append(f"CHECK {i} — {c.name}")
        out.append(f"  Status ....... {c.status}")
        out.append(f"  Measured ..... {c.measured}   Threshold: {c.threshold}")
        out.append(f"  What & why ... {_wrap(c.what_why)}")
        out.append(f"  Sufficient ... {_wrap(c.sufficient)}")
        out.append("-" * W)

    softs = [c for c in checks if c.status in (BLOCKED, SKIPPED, NA)]
    if has_fail:
        verdict, exit_code = "NOT ACCEPTED", 1
        vline = "One or more checks did not clear their threshold at this shape."
    elif scope == "smoke" or softs:
        verdict, exit_code = "ACCEPTED WITH CAVEATS", 0
        capped = "smoke scope (single-process): distributed properties are vacuous here" \
            if scope == "smoke" else \
            "some checks were blocked / skipped / not applicable at this scale"
        vline = f"Every check that ran passed, but {capped} — see the rows above."
    else:
        verdict, exit_code = "ACCEPTED", 0
        vline = f"All checks passed at {shape}."
    out.append(f"VERDICT: {verdict}")
    out.append(f"  {vline}   Sentinels: {sentinels}   Test-id: {test_id or '-'}   "
               f"Exit: {exit_code}")

    if has_fail:
        out.append("")
        out.append("WHAT THIS LIKELY MEANS")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL:
                out.append(_wrap(f"CHECK {i} failed: {c.measured} did not meet "
                                 f"{c.threshold}. {c.likely_means}", indent="  "))
        out.append("")
        out.append("FOR SUPPORT — raw traceback")
        for i, c in enumerate(checks, 1):
            if c.status == FAIL and c.traceback:
                out.append(f"  [CHECK {i} — {c.name}]")
                out.append(c.traceback.rstrip())

    print("\n" + "\n".join(out), flush=True)
    _receipt(checks, verdict, exit_code, test_id)
    return exit_code


# ==========================================================================================
# Fixtures — a tiny GPT2 (Check 1/2) and a tiny torch model (Check 3), both deterministic.
# ==========================================================================================
FIXED_IDS = torch.arange(8, dtype=torch.long).unsqueeze(0) % 32   # (1, 8) token ids for GPT2
TOL = 1e-5


def build_tiny_gpt2():
    """Deterministic tiny GPT2 (transformers). Raises ImportError if transformers absent."""
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=32, n_positions=16, n_embd=32, n_layer=1, n_head=2)
    m = GPT2LMHeadModel(cfg)
    m.eval()
    return m


def gpt2_logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model(FIXED_IDS).logits


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


# ==========================================================================================
# CHECK 1 + 2 — base model load from a UC Volume (offline) and from a /tmp stage.
# ==========================================================================================
def check_volume_load(volume_dir: str) -> "tuple[Check, Check]":
    """Save a tiny GPT2 into the volume, then load it back offline (a) directly from the volume
    and (b) from a /tmp copy, asserting logits match the saved model both ways."""
    load_name = "Base model loads from a UC Volume offline (no hub egress)"
    stage_name = "Base model loads from a local /tmp stage (mmap-over-FUSE-safe path)"
    load_th = f"logits bit-close (atol {TOL}) to the saved model; no network"
    stage_th = f"logits bit-close (atol {TOL}) to the saved model, loaded from /tmp"
    load_what = ("The customer's workspace has no internet egress, so the docs' hub "
                 "`from_pretrained('org/model')` is unreachable. Staging weights onto a UC Volume "
                 "and loading with local_files_only=True + HF_HUB_OFFLINE is the offline path.")
    stage_what = ("safetensors memory-maps weights; mmap over a FUSE mount (a UC Volume) is "
                  "unreliable/slow. Copying the model to local disk once and loading from there is "
                  "the robust pattern (and what the field does).")

    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:                                 # noqa: BLE001 — transformers absent
        blocked = Check(name=load_name, status=BLOCKED,
                        measured=f"transformers import failed: {type(e).__name__}: {e}",
                        threshold=load_th, what_why=load_what,
                        sufficient="Blocked by a missing dependency (transformers), not a fault in "
                                   "the load pattern. Add it to the env or verify it is preinstalled.",
                        likely_means="transformers is not importable in this env — add it to "
                                     "environment.dependencies or vendor it.")
        return blocked, Check(name=stage_name, status=BLOCKED, measured="transformers absent",
                              threshold=stage_th, what_why=stage_what,
                              sufficient="Same dependency block as Check 1.",
                              likely_means="")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_dir = os.path.join(volume_dir, "base-model")

    # Save the reference model into the volume + capture reference logits.
    ref_model = build_tiny_gpt2()
    ref = gpt2_logits(ref_model)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    ref_model.save_pretrained(model_dir)

    # CHECK 1 — load offline directly from the volume.
    try:
        vm = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        vm.eval()
        diff_vol = (gpt2_logits(vm) - ref).abs().max().item()
        ok_vol = diff_vol < TOL
        if ok_vol:
            print(f"MODELIO_VOLUME_LOAD_OK max_logit_diff={diff_vol:.2e} dir={model_dir}", flush=True)
        c1 = Check(name=load_name, status=(PASS if ok_vol else FAIL),
                   measured=f"max logit diff vs saved model = {diff_vol:.2e} (loaded offline from the volume)",
                   threshold=load_th, what_why=load_what,
                   sufficient="Matching logits with local_files_only=True + HF_HUB_OFFLINE proves "
                              "the base model loads from the Volume with no hub access. A failure "
                              "shows as a logit mismatch or a network/hub error.",
                   likely_means="Either the offline load reached for the hub (env not honored) or "
                                "the weights didn't round-trip — check the traceback for a network "
                                "call or a safetensors/mmap error on the FUSE mount.")
    except Exception as e:                                 # noqa: BLE001
        c1 = _fail_from_exc(load_name, load_th, load_what,
                            "Offline load from the volume raised — often a hub reach (offline env "
                            "not honored) or a safetensors mmap error on the FUSE mount; Check 2 "
                            "(/tmp stage) is the mitigation.", e)

    # CHECK 2 — stage to /tmp, load from there.
    try:
        staged = "/tmp/modelio_staged"
        if os.path.isdir(staged):
            shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(model_dir, staged)
        sm = AutoModelForCausalLM.from_pretrained(staged, local_files_only=True)
        sm.eval()
        diff_tmp = (gpt2_logits(sm) - ref).abs().max().item()
        ok_tmp = diff_tmp < TOL
        if ok_tmp:
            print(f"MODELIO_TMP_STAGE_OK max_logit_diff={diff_tmp:.2e} staged={staged}", flush=True)
        c2 = Check(name=stage_name, status=(PASS if ok_tmp else FAIL),
                   measured=f"max logit diff vs saved model = {diff_tmp:.2e} (loaded from /tmp stage)",
                   threshold=stage_th, what_why=stage_what,
                   sufficient="Matching logits from a local /tmp copy proves the stage-then-load "
                              "pattern works — the recommended path when FUSE mmap is a concern.",
                   likely_means="The /tmp-staged load didn't match — unexpected; check the copy "
                                "completed and the traceback.")
    except Exception as e:                                 # noqa: BLE001
        c2 = _fail_from_exc(stage_name, stage_th, stage_what,
                            "Staging to /tmp or loading from it raised — check disk space and the "
                            "traceback.", e)
    return c1, c2


# ==========================================================================================
# CHECK 3 — UC model registry round-trip.
# ==========================================================================================
def check_uc_register(uc_model: str, local: bool) -> Check:
    name = "Model registers to Unity Catalog and loads back via models:/"
    th = f"registered version loads and reproduces outputs (atol {TOL})"
    what = ("Trained models are handed off through the governed UC registry (three-level "
            "catalog.schema.model name). The run must log the model, register a version, and be "
            "loadable back by URI for downstream serving/inference.")
    if local:
        return Check(name=name, status=SKIPPED,
                     measured="no databricks-uc registry under --local",
                     threshold=th, what_why=what,
                     sufficient="Deliberately not run locally: the UC registry needs a workspace. "
                                "Verified on AIR only.", likely_means="")
    try:
        import mlflow
        from mlflow.models import infer_signature
        mlflow.set_registry_uri("databricks-uc")
        net = TinyNet()
        net.eval()
        x = torch.arange(8, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ref_out = net(x)
        # UC registry REQUIRES a model signature (input + output type specs) — log_model without
        # one is rejected at registration (verified: run 921050926151845). Infer it from examples.
        sig = infer_signature(x.numpy(), ref_out.numpy())
        with mlflow.start_run(run_name="modelio-register") as run:
            info = mlflow.pytorch.log_model(net, name="model", registered_model_name=uc_model,
                                            signature=sig)
        version = getattr(info, "registered_model_version", None)
        if version is None:                                # older mlflow: query the registry
            from mlflow.tracking import MlflowClient
            versions = MlflowClient().search_model_versions(f"name='{uc_model}'")
            version = max((int(v.version) for v in versions), default=None)
        loaded = mlflow.pytorch.load_model(f"models:/{uc_model}/{version}")
        loaded.eval()
        with torch.no_grad():
            got = loaded(x)
        diff = (got - ref_out).abs().max().item()
        ok = diff < TOL
        if ok:
            print(f"MODELIO_UC_REGISTER_OK model={uc_model} version={version} "
                  f"max_out_diff={diff:.2e}", flush=True)
        return Check(name=name, status=(PASS if ok else FAIL),
                     measured=f"registered {uc_model} v{version}; reload max output diff = {diff:.2e}",
                     threshold=th, what_why=what,
                     sufficient="A registered version that reloads and reproduces outputs proves "
                                "the UC three-level registration + models:/ load-back works. A "
                                "failure shows as a registration error or an output mismatch.",
                     likely_means="Registration or reload failed — common causes: registry URI not "
                                  "databricks-uc, no CREATE MODEL on the schema, or an mlflow "
                                  "log_model API mismatch (name= vs artifact_path=).")
    except Exception as e:                                 # noqa: BLE001
        return _fail_from_exc(name, th, what,
                              "The UC registry round-trip raised — check registry URI, schema "
                              "privileges (CREATE MODEL), and the mlflow log_model signature.", e)


# ==========================================================================================
# Entry point.
# ==========================================================================================
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local", action="store_true", help="CPU pre-flight; /tmp volume, skip UC register")
    p.add_argument("--volume-dir", default=None,
                   help="writable dir for the staged base model: a UC volume path on AIR")
    p.add_argument("--uc-model", default=None,
                   help="three-level UC model name for Check 3 (catalog.schema.model)")
    args = p.parse_args()

    if args.volume_dir is None:
        if not args.local:
            print("ERROR: --volume-dir (a UC volume path) is required on AIR", file=sys.stderr)
            return 2
        args.volume_dir = "/tmp/modelio_local"
    os.makedirs(args.volume_dir, exist_ok=True)

    runtime = f"torch {torch.__version__}"
    try:
        import transformers
        runtime += f", transformers {transformers.__version__}"
    except Exception:                                      # noqa: BLE001
        runtime += ", transformers ABSENT"
    print(f"MODELIO_VERSIONS {runtime} local={args.local}", flush=True)

    c1, c2 = check_volume_load(args.volume_dir)
    if not args.local and args.uc_model is None:
        c3 = Check(name="Model registers to Unity Catalog and loads back via models:/",
                   status=SKIPPED, measured="no --uc-model provided",
                   threshold="registered version loads and reproduces outputs",
                   what_why="Governed hand-off via the UC registry.",
                   sufficient="Deliberately not run: --uc-model (catalog.schema.model) not given.",
                   likely_means="")
    else:
        c3 = check_uc_register(args.uc_model, args.local)
    checks = [c1, c2, c3]

    sentinels = []
    if all(c.status == PASS for c in checks):
        print("MODELIO_COMPLETE proofs=1,2,3 (volume-load + tmp-stage + uc-register)", flush=True)
        sentinels = ["MODELIO_COMPLETE"]
    else:
        sentinels = ["MODELIO_INCOMPLETE"]

    scope = "smoke" if args.local else "acceptance"
    exit_code = 0
    try:
        run_id = (os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_RUN_NAME") or "local")
        exit_code = render_report(checks, run_id=run_id,
                                  profile=("local-cpu" if args.local else "air"),
                                  shape=f"single-process, {('cpu' if args.local else 'a10')}",
                                  scope=scope, runtime=runtime,
                                  sentinels=" ".join(sentinels), test_id="model-io")
    except Exception:                                      # noqa: BLE001
        _tb.print_exc()
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
