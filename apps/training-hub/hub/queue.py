"""Run admission and tracking.

Domain: Team -> Use case -> Workload -> Run.
- A *workload* is a registered, reusable definition (notebook or AIR YAML + shape + nodes),
  owned by a use case.
- A *run* is one execution of a workload. Every run takes the same path: created QUEUED,
  admitted when capacity allows, then tracked to a terminal state. There is no second path.
- Team is derived from the authenticated user (config membership), never chosen in a form.
- Dispatch order is FIFO within fair-share: teams furthest under their quota share go first.

State: SQLite (app-local, prototype). The Store interface is small on purpose so a
Delta-backed twin can replace it.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import capacity as cap
from . import catalog
from . import recipes

DB_PATH = Path(__file__).resolve().parents[1] / "broker.db"

TERMINAL = {"SUCCESS", "FAILED", "CANCELED"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS workloads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc REAL NOT NULL,
  created_by TEXT NOT NULL,
  team TEXT NOT NULL,
  use_case TEXT NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,              -- 'notebook' | 'air_yaml'
  ref TEXT NOT NULL,               -- notebook workspace path | workload yaml repo path
  shape TEXT NOT NULL,
  nodes INTEGER NOT NULL DEFAULT 1,
  needs_torch INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workload_id INTEGER NOT NULL REFERENCES workloads(id),
  requested_by TEXT NOT NULL,
  created_utc REAL NOT NULL,
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
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert(self, table: str, **kw) -> int:
        q = f"INSERT INTO {table} ({','.join(kw)}) VALUES ({','.join('?' * len(kw))})"
        cur = self.conn.execute(q, tuple(kw.values()))
        self.conn.commit()
        return cur.lastrowid

    def update(self, table: str, row_id: int, **kw):
        sets = ",".join(f"{k}=?" for k in kw)
        self.conn.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*kw.values(), row_id))
        self.conn.commit()

    def rows(self, sql: str, args=()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, args).fetchall()


class GateError(Exception):
    """Raised when the requesting user is not allowed to do the thing."""


@dataclass
class Broker:
    cfg: object                 # hub.config.HubConfig
    ws: object                  # databricks.sdk WorkspaceClient (None = dry-run)
    store: Store = None

    def __post_init__(self):
        self.store = self.store or Store()
        self._sync_repo_catalog()

    def _sync_repo_catalog(self):
        """Repo workloads re-sync on every start; git is the source of truth for them."""
        team = getattr(self.cfg, "catalog_team", "") or (
            self.cfg.teams[0].name if self.cfg.teams else "")
        if not team:
            return
        existing = {r["name"]: r["id"] for r in self.store.rows(
            "SELECT id, name FROM workloads WHERE created_by='repo-sync'")}
        for w in catalog.repo_workloads(team):
            fields = dict(team=w["team"], use_case=w["use_case"], name=w["name"],
                          kind=w["kind"], ref=w["ref"], shape=w["shape"],
                          nodes=w["nodes"], needs_torch=w["needs_torch"])
            if w["name"] in existing:
                self.store.update("workloads", existing[w["name"]], **fields)
            else:
                self.store.insert("workloads", created_utc=time.time(),
                                  created_by="repo-sync", **fields)

    # ---------- gates ----------
    def _team_for(self, user: str, team_name: str):
        teams = self.cfg.teams_of(user)
        if not teams:
            raise GateError("You are not a member of any team; access is read-only. "
                            "Ask your platform admin to add you to a team in the hub config.")
        team = next((t for t in teams if t.name == team_name), None)
        if team is None:
            raise GateError(f"You are not a member of team '{team_name}'.")
        return team

    # ---------- workloads ----------
    def register_workload(self, *, user: str, team: str, use_case: str, name: str,
                          kind: str, ref: str, shape: str, nodes: int = 1,
                          needs_torch: bool = False) -> int:
        t = self._team_for(user, team)
        if use_case not in [u.name for u in t.use_cases]:
            raise GateError(f"'{use_case}' is not a use case of team {team}.")
        return self.store.insert("workloads", created_utc=time.time(), created_by=user,
                                 team=team, use_case=use_case, name=name, kind=kind,
                                 ref=ref, shape=shape, nodes=nodes,
                                 needs_torch=int(needs_torch))

    def workloads(self, teams: list[str] | None = None) -> list[sqlite3.Row]:
        if teams:
            marks = ",".join("?" * len(teams))
            return self.store.rows(
                f"SELECT * FROM workloads WHERE team IN ({marks}) ORDER BY team, use_case, id",
                tuple(teams))
        return self.store.rows("SELECT * FROM workloads ORDER BY team, use_case, id")

    # ---------- runs ----------
    def request_run(self, *, user: str, workload_id: int) -> int:
        w = self.store.rows("SELECT * FROM workloads WHERE id=?", (workload_id,))
        if not w:
            raise GateError(f"workload {workload_id} not found")
        self._team_for(user, w[0]["team"])   # must be a member of the owning team
        return self.store.insert("runs", workload_id=workload_id, requested_by=user,
                                 created_utc=time.time())

    def runs(self, states: set[str] | None = None) -> list[sqlite3.Row]:
        q = ("SELECT r.*, w.team, w.use_case, w.name, w.shape, w.nodes, w.kind, w.ref,"
             " w.needs_torch FROM runs r JOIN workloads w ON w.id = r.workload_id")
        rows = self.store.rows(q + " ORDER BY r.id")
        return [r for r in rows if states is None or r["state"] in states]

    # ---------- capacity ----------
    def in_flight(self, shape: str, team: str | None = None) -> int:
        return sum(r["nodes"] for r in self.runs({"SUBMITTED", "RUNNING"})
                   if r["shape"] == shape and (team is None or r["team"] == team))

    def capacity(self, shape: str) -> cap.ShapeCapacity:
        return cap.shape_capacity(self.cfg, shape, self.in_flight(shape))

    # ---------- dispatch: FIFO within fair-share ----------
    def _share_ratio(self, team_name: str) -> float:
        team = next((t for t in self.cfg.teams if t.name == team_name), None)
        if team is None or team.quota_nodes <= 0:
            return 1e9
        used = sum(r["nodes"] for r in self.runs({"SUBMITTED", "RUNNING"})
                   if r["team"] == team_name)
        return used / team.quota_nodes

    def tick(self) -> list[str]:
        events = []
        for r in self.runs({"SUBMITTED", "RUNNING"}):
            events += self._sync(r)
        queued = sorted(self.runs({"QUEUED"}),
                        key=lambda r: (self._share_ratio(r["team"]), r["id"]))
        for r in queued:
            team = next((t for t in self.cfg.teams if t.name == r["team"]), None)
            team_room = (team.quota_nodes - self.in_flight(r["shape"], r["team"])
                         if team else 0)
            if self.capacity(r["shape"]).admittable >= r["nodes"] and team_room >= r["nodes"]:
                events.append(self._submit(r))
        return events

    def _submit(self, r) -> str:
        if r["kind"] != "notebook":
            return self._submit_air_yaml(r)   # air CLI path needs no workspace client
        if self.ws is None:
            self.store.update("runs", r["id"], state="SUBMITTED",
                              run_id=f"dryrun-{r['id']}", submitted_utc=time.time())
            return f"run {r['id']} ({r['name']}): submitted (dry-run)"
        task = {"task_key": "work",
                "notebook_task": {"notebook_path": r["ref"]},
                "environment_key": "hub", "timeout_seconds": 3600 * 6}
        if r["shape"] != "CPU":
            task["compute"] = {"hardware_accelerator": r["shape"]}
        body = {"run_name": f"hub-{r['team']}-{r['use_case']}-run{r['id']}",
                "tasks": [task], "environments": [recipes.job_environment()]}
        try:
            resp = self.ws.api_client.do("POST", "/api/2.2/jobs/runs/submit", body=body)
            self.store.update("runs", r["id"], state="SUBMITTED",
                              run_id=str(resp["run_id"]), submitted_utc=time.time())
            return f"run {r['id']} ({r['name']}): submitted as {resp['run_id']}"
        except Exception as e:
            self.store.update("runs", r["id"], state="FAILED", detail=str(e)[:400],
                              finished_utc=time.time())
            return f"run {r['id']}: submit failed — {e}"

    def _submit_air_yaml(self, r) -> str:
        """Repo workloads submit through the air CLI — the only supported path for
        Gen-AI tasks (persistent-job wrapping is receipt-dead, docs/06)."""
        repo_root = Path(__file__).resolve().parents[3]
        yaml_path = repo_root / r["ref"]
        if not yaml_path.exists():
            self.store.update("runs", r["id"], state="FAILED", finished_utc=time.time(),
                              detail=f"{r['ref']} not found in repo")
            return f"run {r['id']}: {r['ref']} not found"
        profile = os.environ.get("HUB_PROFILE", "mkazia-lw2")
        try:
            out = subprocess.run(
                ["air", "run", "--file", str(yaml_path), "-p", profile],
                capture_output=True, text=True, timeout=300, cwd=repo_root)
            text = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout + out.stderr)
            m = re.search(r"Job Run ID:\s*(\d+)", text)
            if m:
                self.store.update("runs", r["id"], state="SUBMITTED", run_id=m.group(1),
                                  submitted_utc=time.time())
                return f"run {r['id']} ({r['name']}): submitted as {m.group(1)}"
            self.store.update("runs", r["id"], state="FAILED", finished_utc=time.time(),
                              detail=text[-400:])
            return f"run {r['id']}: air run gave no Job Run ID"
        except Exception as e:
            self.store.update("runs", r["id"], state="FAILED", finished_utc=time.time(),
                              detail=str(e)[:400])
            return f"run {r['id']}: air run failed — {e}"

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
            self.store.update("runs", r["id"], state="RUNNING")
            return [f"run {r['id']}: running"]
        if life in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
            final = {"SUCCESS": "SUCCESS", "CANCELED": "CANCELED"}.get(result, "FAILED")
            self.store.update("runs", r["id"], state=final, finished_utc=time.time(),
                              detail=run["state"].get("state_message", "")[:400])
            return [f"run {r['id']}: {final.lower()}"]
        return []
