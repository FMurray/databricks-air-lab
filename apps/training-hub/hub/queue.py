"""Submission broker: the queue the platform doesn't have.

The platform fail-fasts over-quota submits (no queue, no arbitration — docs/06 receipt), so
the hub holds requests in QUEUED and dispatches only when the capacity model says a slot is
admittable. Gating (team membership, shape allowlist) and attribution (team/user/tag ledger)
happen at enqueue — every brokered node-hour is attributable, which is the chargeback story
the platform can't give yet.

State: SQLite (app-local, prototype). Swap `Store` for a Delta-backed twin when catalog
storage is available; the interface is deliberately tiny.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from . import capacity as cap
from . import recipes

DB_PATH = Path(__file__).resolve().parents[1] / "broker.db"

TERMINAL = {"SUCCESS", "FAILED", "CANCELED", "REJECTED"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc REAL NOT NULL,
  team TEXT NOT NULL,
  user TEXT NOT NULL,
  use_case TEXT DEFAULT '',
  kind TEXT NOT NULL,              -- 'notebook' | 'air_yaml'
  ref TEXT NOT NULL,               -- notebook workspace path | workload yaml repo path
  shape TEXT NOT NULL,             -- e.g. GPU_1xA10, GPU_8xH100
  nodes INTEGER NOT NULL DEFAULT 1,
  needs_torch INTEGER DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'QUEUED',
  run_id TEXT DEFAULT '',
  detail TEXT DEFAULT '',
  submitted_utc REAL,
  finished_utc REAL
);
"""


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def insert(self, **kw) -> int:
        cols = ",".join(kw)
        q = f"INSERT INTO requests ({cols}) VALUES ({','.join('?'*len(kw))})"
        cur = self.conn.execute(q, tuple(kw.values()))
        self.conn.commit()
        return cur.lastrowid

    def update(self, req_id: int, **kw):
        sets = ",".join(f"{k}=?" for k in kw)
        self.conn.execute(f"UPDATE requests SET {sets} WHERE id=?", (*kw.values(), req_id))
        self.conn.commit()

    def rows(self, where: str = "1=1", args=()) -> list[sqlite3.Row]:
        return self.conn.execute(
            f"SELECT * FROM requests WHERE {where} ORDER BY id", args).fetchall()


@dataclass
class Broker:
    cfg: object                 # hub.config.HubConfig
    ws: object                  # databricks.sdk WorkspaceClient (or None for dry-run)
    store: Store = None

    def __post_init__(self):
        self.store = self.store or Store()

    # ---------- gate + enqueue (items 6, 7) ----------
    def enqueue(self, *, user: str, team: str, kind: str, ref: str, shape: str,
                nodes: int = 1, use_case: str = "", needs_torch: bool = False) -> tuple[int, str]:
        t = next((t for t in self.cfg.teams if t.name == team), None)
        if t is None:
            return self._reject(user, team, kind, ref, shape, nodes, use_case, "unknown team")
        if t.members and user not in t.members:
            return self._reject(user, team, kind, ref, shape, nodes, use_case,
                                f"{user} is not a member of team {team}")
        allowed = getattr(t, "allowed_shapes", None)
        if allowed and shape not in allowed:
            return self._reject(user, team, kind, ref, shape, nodes, use_case,
                                f"shape {shape} not allowed for team {team}")
        rid = self.store.insert(created_utc=time.time(), team=team, user=user,
                                use_case=use_case, kind=kind, ref=ref, shape=shape,
                                nodes=nodes, needs_torch=int(needs_torch))
        return rid, "QUEUED"

    def _reject(self, user, team, kind, ref, shape, nodes, use_case, why) -> tuple[int, str]:
        rid = self.store.insert(created_utc=time.time(), team=team, user=user,
                                use_case=use_case, kind=kind, ref=ref, shape=shape,
                                nodes=nodes, state="REJECTED", detail=why,
                                finished_utc=time.time())
        return rid, f"REJECTED: {why}"

    # ---------- capacity view (item 1) ----------
    def in_flight(self, shape: str, team: str | None = None) -> int:
        where = "state IN ('SUBMITTED','RUNNING') AND shape=?"
        args = [shape]
        if team:
            where += " AND team=?"
            args.append(team)
        return sum(r["nodes"] for r in self.store.rows(where, tuple(args)))

    def capacity(self, shape: str) -> cap.ShapeCapacity:
        return cap.shape_capacity(self.cfg, shape, self.in_flight(shape))

    # ---------- dispatch (item 2) ----------
    def tick(self) -> list[str]:
        """One dispatcher pass: sync run states, then admit queue heads while slots exist."""
        events = []
        for r in self.store.rows("state IN ('SUBMITTED','RUNNING')"):
            events += self._sync(r)
        for r in self.store.rows("state='QUEUED'"):
            shape_ok = self.capacity(r["shape"]).admittable >= r["nodes"]
            team_ok = cap.team_headroom(self.cfg, r["team"],
                                        self.in_flight(r["shape"], r["team"])) >= r["nodes"]
            if shape_ok and team_ok:
                events.append(self._submit(r))
        return events

    def _submit(self, r) -> str:
        if self.ws is None:  # dry-run mode (tests / no workspace)
            self.store.update(r["id"], state="SUBMITTED", run_id=f"dryrun-{r['id']}",
                              submitted_utc=time.time())
            return f"req {r['id']}: SUBMITTED (dry-run)"
        if r["kind"] != "notebook":
            # AIR CLI workloads: shell the (vendored) air CLI — the only proven path for
            # Gen-AI tasks; persistent-job wrapping is receipt-dead (docs/06).
            self.store.update(r["id"], state="FAILED", detail="air_yaml dispatch not wired yet",
                              finished_utc=time.time())
            return f"req {r['id']}: FAILED (air_yaml dispatch not wired yet)"
        task = {"task_key": "work",
                "notebook_task": {"notebook_path": r["ref"]},
                "environment_key": "hub", "timeout_seconds": 3600 * 6}
        if r["shape"] != "CPU":
            task["compute"] = {"hardware_accelerator": r["shape"]}
        body = {"run_name": f"hub-{r['team']}-{r['id']}",
                "tasks": [task],
                "environments": [recipes.job_environment()]}
        try:
            resp = self.ws.api_client.do("POST", "/api/2.2/jobs/runs/submit", body=body)
            self.store.update(r["id"], state="SUBMITTED", run_id=str(resp["run_id"]),
                              submitted_utc=time.time())
            return f"req {r['id']}: SUBMITTED run {resp['run_id']}"
        except Exception as e:
            self.store.update(r["id"], state="FAILED", detail=str(e)[:400],
                              finished_utc=time.time())
            return f"req {r['id']}: submit FAILED {e}"

    def _sync(self, r) -> list[str]:
        if self.ws is None or str(r["run_id"]).startswith("dryrun-"):
            return []
        try:
            run = self.ws.api_client.do("GET", f"/api/2.2/jobs/runs/get?run_id={r['run_id']}")
        except Exception:
            return []
        life = run["state"]["life_cycle_state"]
        result = run["state"].get("result_state", "")
        if life == "RUNNING" and r["state"] != "RUNNING":
            self.store.update(r["id"], state="RUNNING")
            return [f"req {r['id']}: RUNNING"]
        if life in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            final = {"SUCCESS": "SUCCESS", "CANCELED": "CANCELED"}.get(result, "FAILED")
            self.store.update(r["id"], state=final, finished_utc=time.time(),
                              detail=run["state"].get("state_message", "")[:400])
            return [f"req {r['id']}: {final}"]
        return []

    # ---------- attribution ledger (item 7) ----------
    def ledger(self) -> list[dict]:
        return [dict(r) for r in self.store.rows()]

    def ledger_json(self) -> str:
        return json.dumps(self.ledger(), indent=1, default=str)
