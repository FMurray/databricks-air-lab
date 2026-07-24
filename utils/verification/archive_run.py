"""Archive workspace MLflow runs into the repo-local store (experiments/mlruns).

Evidence receipts for the experiment-verification standard: instead of loose log files,
each finding cites a local MLflow run that is a faithful copy of the platform-recorded run
(params, metric histories, tags, artifacts) plus provenance tags linking it to the source
workspace/job. The local store is a plain MLflow file store — committed to the repo, so
receipts travel with `git clone` and open with `mlflow ui`.

Usage:
  uv run --with mlflow python -m utils.verification.archive_run \\
      --profile fevm-forrest --run-id <mlflow_run_id> [--extra path ...]
  uv run --with mlflow python -m utils.verification.archive_run \\
      --profile e2-demo-field-eng --job-run-id <id> --experiment <name> [--extra path ...]

--extra attaches local files (e.g. the air CLI submission log) under client_logs/ on the
archived run, so client-side evidence is linked to the platform run identity in one place.
"""

from __future__ import annotations

import argparse
import configparser
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[2]
# SQLite backend (mlflow 3.x deprecates the plain file store): one committable db file,
# artifacts alongside it. Open with: mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
LOCAL_STORE = f"sqlite:///{REPO_ROOT}/experiments/mlflow.db"
ARTIFACT_ROOT = f"file://{REPO_ROOT}/experiments/mlartifacts"


def _workspace_host(profile: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(Path.home() / ".databrickscfg")
    return cfg.get(profile, "host", fallback="unknown")


def _resolve_run(src: MlflowClient, job_run_id: str, experiment: str) -> str:
    exp = src.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"experiment not found: {experiment}")
    runs = src.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.`mlflow.databricks.jobRunID` = '{job_run_id}'",
    )
    if not runs:
        raise SystemExit(f"no MLflow run tagged with job run {job_run_id} in {experiment}")
    return runs[0].info.run_id


def archive(profile: str, run_id: str, extras: list[str]) -> str:
    src = MlflowClient(tracking_uri=f"databricks://{profile}")
    dst = MlflowClient(tracking_uri=LOCAL_STORE)

    run = src.get_run(run_id)
    exp_name = src.get_experiment(run.info.experiment_id).name
    dst_exp = dst.get_experiment_by_name(exp_name)
    dst_exp_id = (
        dst_exp.experiment_id
        if dst_exp
        else dst.create_experiment(exp_name, artifact_location=ARTIFACT_ROOT)
    )

    tags = dict(run.data.tags)
    tags.update(
        {
            "archive.source_run_id": run_id,
            "archive.source_workspace": _workspace_host(profile),
            "archive.source_profile": profile,
            "archive.archived_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    dst_run = dst.create_run(dst_exp_id, run_name=run.info.run_name, tags=tags)
    dst_id = dst_run.info.run_id

    for key, value in run.data.params.items():
        dst.log_param(dst_id, key, value)
    for key in run.data.metrics:
        for m in src.get_metric_history(run_id, key):
            dst.log_metric(dst_id, key, m.value, timestamp=m.timestamp, step=m.step)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, dst_path=tmp, tracking_uri=f"databricks://{profile}"
            )
            if any(Path(local).iterdir()):
                dst.log_artifacts(dst_id, local)
        except Exception as e:  # absent artifacts are themselves evidence — record, don't die
            dst.set_tag(dst_id, "archive.artifacts_error", str(e)[:250])

    for extra in extras:
        dst.log_artifact(dst_id, extra, artifact_path="client_logs")

    dst.set_terminated(dst_id, status=run.info.status)
    print(f"archived {run_id} ({_workspace_host(profile)})")
    print(f"  -> local run {dst_id} in experiment '{exp_name}'")
    print(f"  view: mlflow ui --backend-store-uri {LOCAL_STORE}")
    return dst_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", required=True, help="databricks profile of the source workspace")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="source MLflow run id")
    group.add_argument("--job-run-id", help="job run id (requires --experiment)")
    ap.add_argument("--experiment", help="source experiment name (with --job-run-id)")
    ap.add_argument("--extra", action="append", default=[], help="local file to attach under client_logs/")
    args = ap.parse_args()

    run_id = args.run_id
    if args.job_run_id:
        if not args.experiment:
            ap.error("--job-run-id requires --experiment")
        src = MlflowClient(tracking_uri=f"databricks://{args.profile}")
        run_id = _resolve_run(src, args.job_run_id, args.experiment)

    archive(args.profile, run_id, args.extra)


if __name__ == "__main__":
    main()
