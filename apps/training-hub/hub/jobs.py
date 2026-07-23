"""Currently-active runs via the Jobs API.

AIR CLI workloads surface as job runs (verified: smoke runs appear with job IDs).
TODO(open-q): confirm the reliable discriminator for AIR/serverless-GPU runs vs other
jobs — until then this lists all active runs and the UI says so.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from databricks.sdk import WorkspaceClient


def active_runs(max_runs: int = 100) -> pd.DataFrame:
    w = WorkspaceClient()
    rows = []
    for run in w.jobs.list_runs(active_only=True):
        started = (
            datetime.fromtimestamp(run.start_time / 1000, tz=timezone.utc)
            if run.start_time
            else None
        )
        rows.append(
            {
                "run_name": run.run_name,
                "creator": run.creator_user_name,
                "state": run.state.life_cycle_state.value if run.state else None,
                "started_utc": started,
                "run_page": run.run_page_url,
            }
        )
        if len(rows) >= max_runs:
            break
    return pd.DataFrame(rows)
