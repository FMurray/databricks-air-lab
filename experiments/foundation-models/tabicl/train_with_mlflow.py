"""torchrun entrypoint: ``tabicl.train`` with wandb redirected to MLflow.

Drop-in replacement for ``torchrun ... -m tabicl.train`` — all tabicl.train CLI
args pass through unchanged (run with ``--wandb_log True`` so upstream actually
emits metrics; the shim sends them to MLflow, the real wandb is never imported):

    torchrun --standalone --nproc_per_node=N train_with_mlflow.py --wandb_log True ...
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wandb_mlflow_shim import install

install()
runpy.run_module("tabicl.train", run_name="__main__", alter_sys=True)
