/**
 * Run/workload state in SQLite (app-local). One submission path: every run is created
 * QUEUED and admitted by the dispatcher; FIFO within fair-share.
 * NB for hosted deploys: the Apps filesystem is ephemeral — swap for Lakebase before
 * anything durable matters. The interface is intentionally small.
 */
import Database from "better-sqlite3";
import path from "node:path";

export interface RunRow {
  id: number;
  workload_id: number;
  requested_by: string;
  created_utc: number;
  state: string;
  run_id: string;
  detail: string;
  // joined from workloads:
  team: string;
  use_case: string;
  name: string;
  shape: string;
  nodes: number;
  kind: string;
  ref: string;
  needs_torch: number;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS workloads (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc REAL NOT NULL,
  created_by TEXT NOT NULL,
  team TEXT NOT NULL,
  use_case TEXT NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  ref TEXT NOT NULL,
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
`;

export class Store {
  db: Database.Database;

  constructor(appRoot: string) {
    this.db = new Database(path.join(appRoot, "broker.db"));
    this.db.exec(SCHEMA);
  }

  upsertRepoWorkload(w: {
    team: string;
    use_case: string;
    name: string;
    kind: string;
    ref: string;
    shape: string;
    nodes: number;
    needs_torch: boolean;
  }) {
    const existing = this.db
      .prepare("SELECT id FROM workloads WHERE name=? AND created_by='repo-sync'")
      .get(w.name) as { id: number } | undefined;
    if (existing) {
      this.db
        .prepare(
          `UPDATE workloads SET team=?, use_case=?, kind=?, ref=?, shape=?, nodes=?,
           needs_torch=? WHERE id=?`,
        )
        .run(w.team, w.use_case, w.kind, w.ref, w.shape, w.nodes, w.needs_torch ? 1 : 0, existing.id);
      return existing.id;
    }
    return Number(
      this.db
        .prepare(
          `INSERT INTO workloads (created_utc, created_by, team, use_case, name, kind, ref,
           shape, nodes, needs_torch) VALUES (?, 'repo-sync', ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(Date.now() / 1000, w.team, w.use_case, w.name, w.kind, w.ref, w.shape, w.nodes, w.needs_torch ? 1 : 0).lastInsertRowid,
    );
  }

  workloads(teams?: string[]) {
    if (teams?.length) {
      const marks = teams.map(() => "?").join(",");
      return this.db
        .prepare(`SELECT * FROM workloads WHERE team IN (${marks}) ORDER BY use_case, name`)
        .all(...teams);
    }
    return this.db.prepare("SELECT * FROM workloads ORDER BY use_case, name").all();
  }

  workload(id: number) {
    return this.db.prepare("SELECT * FROM workloads WHERE id=?").get(id) as
      | Record<string, unknown>
      | undefined;
  }

  insertRun(workloadId: number, requestedBy: string): number {
    return Number(
      this.db
        .prepare("INSERT INTO runs (workload_id, requested_by, created_utc) VALUES (?, ?, ?)")
        .run(workloadId, requestedBy, Date.now() / 1000).lastInsertRowid,
    );
  }

  updateRun(id: number, fields: Record<string, unknown>) {
    const keys = Object.keys(fields);
    this.db
      .prepare(`UPDATE runs SET ${keys.map((k) => `${k}=?`).join(",")} WHERE id=?`)
      .run(...keys.map((k) => fields[k]), id);
  }

  runs(states?: string[]): RunRow[] {
    const rows = this.db
      .prepare(
        `SELECT r.*, w.team, w.use_case, w.name, w.shape, w.nodes, w.kind, w.ref, w.needs_torch
         FROM runs r JOIN workloads w ON w.id = r.workload_id ORDER BY r.id`,
      )
      .all() as RunRow[];
    return states ? rows.filter((r) => states.includes(r.state)) : rows;
  }
}
