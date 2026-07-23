"""GPU spend attributed to teams via config.

SQL lives in utils/billing/queries.py (single source of truth, shared with notebooks
and the CLI runner). Local dev imports it from the repo checkout; for app deployment
copy it into the app dir first (`cp -r ../../utils .` — see README).

Attribution caveat (open-q #5/#16): reserved pools bill aggregate records today, so
principal-level rows may undercount pool usage. Anything we can't attribute is kept
and labeled 'unmapped' rather than dropped — the admin needs to see that gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from utils.billing import queries
except ImportError:  # running from apps/training-hub in the repo checkout
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from utils.billing import queries

from .config import HubConfig

workspace_host = queries.workspace_host


def fetch_gpu_usage(days: int = 30) -> pd.DataFrame:
    return queries.run(queries.air_usage_daily(days))


def usage_by_team(cfg: HubConfig, days: int = 30) -> pd.DataFrame:
    df = queries.run(queries.by_principal(days))
    if df.empty:
        return pd.DataFrame(
            columns=["team", "dbus", "est_list_cost_usd", "principals", "runs"]
        )
    df["team"] = df["principal"].map(cfg.team_of)
    return (
        df.groupby("team")
        .agg(
            dbus=("dbus", "sum"),
            est_list_cost_usd=("est_list_cost_usd", "sum"),
            principals=("principal", "nunique"),
            runs=("jobs", "sum"),
        )
        .sort_values("dbus", ascending=False)
        .reset_index()
    )
