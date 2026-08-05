"""Redirect tabicl's wandb calls to MLflow without touching upstream code.

``tabicl.train._run`` does a module-level ``import wandb`` and uses exactly three
symbols (verified against soda-inria/tabicl@main, 2026-08-05):

  - ``wandb.init(dir=, project=, name=, id=, config=, resume=, mode=)``
  - the returned run's ``.id`` (persisted to ``checkpoint_dir/wand_id.txt`` and
    passed back as ``id=`` on restart — which this shim turns into MLflow
    run-resume for free)
  - ``wandb.log(results, step=)``

``install()`` plants a fake ``wandb`` module in ``sys.modules`` backed by
``MLflowLogger``, so it must run before anything imports tabicl — see
``train_with_mlflow.py``. The real wandb package is never imported and need not
be installed. Only rank 0 calls ``wandb.init`` upstream, so this is DDP-safe.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

from mlflow_loggers import MLflowLogger


class _ShimRun:
    def __init__(self, logger: Optional[MLflowLogger]):
        self._logger = logger

    @property
    def id(self) -> str:
        return self._logger.run_id if self._logger else "wandb-disabled"

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self._logger:
            self._logger.log_metrics(metrics, step=step)

    def finish(self) -> None:
        if self._logger:
            self._logger.finish()


def install() -> None:
    """Install the fake ``wandb`` module. Idempotent."""
    if isinstance(sys.modules.get("wandb"), types.ModuleType) and getattr(
        sys.modules.get("wandb"), "__mlflow_shim__", False
    ):
        return

    mod = types.ModuleType("wandb")
    mod.__mlflow_shim__ = True
    state: dict[str, Optional[_ShimRun]] = {"run": None}

    def init(
        project: Optional[str] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002 — wandb API shape
        config: Any = None,
        mode: Optional[str] = None,
        **_ignored: Any,
    ) -> _ShimRun:
        if mode == "disabled":
            state["run"] = _ShimRun(None)
            return state["run"]
        # wandb "project" is not a Databricks experiment path; pass it through
        # only when it looks like one, otherwise inherit the ambient experiment.
        experiment = project if project and project.startswith("/") else None
        logger = MLflowLogger(experiment=experiment, run_name=name, run_id=id)
        logger.setup(config)
        state["run"] = _ShimRun(logger)
        return state["run"]

    def log(metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if state["run"]:
            state["run"].log(metrics, step=step)

    def finish() -> None:
        if state["run"]:
            state["run"].finish()
            state["run"] = None

    mod.init = init
    mod.log = log
    mod.finish = finish
    mod.define_metric = lambda *a, **k: None  # no-op; MLflow steps cover this
    mod.run = None
    sys.modules["wandb"] = mod
