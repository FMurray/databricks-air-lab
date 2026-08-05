"""MLflow experiment loggers for TabICL training on Databricks.

``MLflowLogger`` is the base: it owns tracking-URI resolution (Databricks managed
MLflow by default), run attach/create semantics, and param/metric plumbing.
Subclasses adapt it to a specific training loop's logging interface —
``MLflowFinetuningLogger`` satisfies tabicl's ``FinetuningLogger`` protocol
(``setup`` / ``log_step`` / ``log_epoch`` / ``finish``); the pretrain path reuses
the base via ``wandb_mlflow_shim``.

Run-attach precedence (so metrics land in the AIR harness's run, not a stray one):
  1. an in-process active run (``mlflow.active_run()``) — attach, never end it
  2. an explicit ``run_id`` (resume, e.g. restart semantics) — resume, never end it
  3. ``MLFLOW_RUN_ID`` in the environment — resume, never end it
  4. otherwise start a fresh run — this is the only run ``finish()`` will end

mlflow is imported lazily (in ``setup``) so this module imports cleanly in
environments without it, mirroring upstream's lazy-wandb pattern.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

_METRIC_KEY_BAD_CHARS = re.compile(r"[^A-Za-z0-9_\-. /]")


def _flatten_config(config: Any) -> dict[str, Any]:
    """Accept a dict, argparse Namespace, or any object with __dict__."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    return dict(vars(config))


class MLflowLogger:
    """Base MLflow logger with Databricks managed MLflow setup.

    Parameters
    ----------
    experiment : str or None
        MLflow experiment to log into. On Databricks this must be a workspace
        path (``/Users/...`` or ``/Shared/...``). Leave ``None`` to inherit the
        ambient experiment (AIR sets it from the workload YAML's
        ``experiment_name``).
    run_name : str or None
        Name for the run — only used when this logger has to start a fresh run.
    run_id : str or None
        Existing run to resume (wins over starting a fresh run; see module
        docstring for full precedence).
    tracking_uri : str or None
        Explicit tracking URI. Default resolution: honor ``MLFLOW_TRACKING_URI``
        if set (AIR containers preconfigure it), else ``"databricks"`` so local
        invocations hit managed MLflow via the ambient profile
        (``DATABRICKS_CONFIG_PROFILE``).
    tags : dict or None
        Tags applied when this logger starts a fresh run.
    """

    def __init__(
        self,
        experiment: Optional[str] = None,
        run_name: Optional[str] = None,
        run_id: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ):
        self.experiment = experiment
        self.run_name = run_name
        self.run_id = run_id
        self.tracking_uri = tracking_uri
        self.tags = tags
        self._mlflow = None
        self._owns_run = False

    def setup(self, config: Any = None) -> None:
        try:
            import mlflow  # noqa: PLC0415
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "MLflowLogger requires the 'mlflow' package. Install it with: pip install mlflow"
            ) from None
        self._mlflow = mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        elif not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("databricks")

        if self.experiment:
            mlflow.set_experiment(self.experiment)

        active = mlflow.active_run()
        if active is not None:
            self.run_id = active.info.run_id
        elif self.run_id:
            run = mlflow.start_run(run_id=self.run_id)
            self.run_id = run.info.run_id
        elif os.environ.get("MLFLOW_RUN_ID"):
            run = mlflow.start_run()  # attaches to MLFLOW_RUN_ID
            self.run_id = run.info.run_id
        else:
            run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
            self.run_id = run.info.run_id
            self._owns_run = True

        self.log_params(_flatten_config(config))

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is None or not params:
            return
        try:
            from mlflow.utils.validation import (  # noqa: PLC0415
                MAX_PARAM_VAL_LENGTH,
                MAX_PARAMS_TAGS_PER_BATCH,
            )
        except ImportError:
            MAX_PARAM_VAL_LENGTH, MAX_PARAMS_TAGS_PER_BATCH = 500, 100

        clean = {
            _METRIC_KEY_BAD_CHARS.sub("_", str(k)): str(v)[:MAX_PARAM_VAL_LENGTH]
            for k, v in params.items()
        }
        items = list(clean.items())
        for i in range(0, len(items), MAX_PARAMS_TAGS_PER_BATCH):
            self._mlflow.log_params(dict(items[i : i + MAX_PARAMS_TAGS_PER_BATCH]))

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self._mlflow is None:
            return
        numeric = {
            _METRIC_KEY_BAD_CHARS.sub("_", str(k)): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def finish(self) -> None:
        """End the run only if this logger started it (never the AIR harness's run)."""
        if self._mlflow is not None and self._owns_run:
            self._mlflow.end_run()
            self._owns_run = False


class MLflowFinetuningLogger(MLflowLogger):
    """MLflowLogger adapted to tabicl's ``FinetuningLogger`` protocol.

    Inject via the instance override (upstream only constructs ``WandbLogger``):

        clf = FinetunedTabICLClassifier(...)
        clf._make_experiment_logger = lambda: MLflowFinetuningLogger()
    """

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        self.log_metrics(metrics, step=step)

    def log_epoch(self, metrics: dict[str, float], step: int) -> None:
        self.log_metrics(metrics, step=step)
