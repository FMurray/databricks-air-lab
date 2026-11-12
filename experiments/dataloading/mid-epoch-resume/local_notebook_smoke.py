"""Off-cluster smoke test of mid_epoch_resume_walkthrough.py: executes every cell top to bottom
with a stub ``dbutils``, ``/Volumes`` redirected to a scratch dir, and the real AIR SDK code
(AIR_LAB_LOCAL_FS=1). Catches broken cells before anyone opens the notebook on AIR; it does not
replace running it there (FUSE, GPUs, env v5/v6 interpreter layout are not exercised).

    AIR_LAB_LOCAL_FS=1 PYTHONPATH=~/universe/ai-compute/databricks-sdk-air \\
      uv run --no-project --with torch --with mlflow-skinny --with numpy \\
      python local_notebook_smoke.py --scratch /tmp/mer-nb
"""

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "mid_epoch_resume_walkthrough.py")


class _Widgets:
    def __init__(self, values):
        self.values = values

    def text(self, name, default, label=None):  # noqa: ARG002
        self.values.setdefault(name, default)

    dropdown = None

    def get(self, name):
        return self.values[name]


class _DBUtils:
    def __init__(self, values):
        self.widgets = _Widgets(values)

        class _NB:
            @property
            def entry_point(self):
                raise RuntimeError("no notebook context off-cluster")
        self.notebook = _NB()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/tmp/mer-nb")
    a = ap.parse_args()
    shutil.rmtree(a.scratch, ignore_errors=True)
    os.makedirs(a.scratch)
    src = open(NB).read().replace('f"/Volumes/', f'f"{a.scratch}/Volumes/')
    cells = [c for c in re.split(r"^# COMMAND -+$", src, flags=re.M)]
    code_cells = [c for c in cells if any(l.strip() and not l.startswith("# MAGIC") and
                                          not l.startswith("# Databricks notebook source")
                                          for l in c.splitlines())]
    ns = {"dbutils": _DBUtils({"catalog": "cat", "schema": "sch", "volume": "vol"}),
          "__name__": "__main__"}
    os.chdir(HERE)
    for i, cell in enumerate(code_cells, 1):
        print(f"\n######## code cell {i}/{len(code_cells)}", flush=True)
        exec(compile(cell, f"<cell {i}>", "exec"), ns)  # noqa: S102 - our own notebook source
    print("\nNOTEBOOK_SMOKE_OK all code cells ran", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
