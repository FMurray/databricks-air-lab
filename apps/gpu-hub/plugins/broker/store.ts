/**
 * Run/workload state in Lakebase (Postgres with OAuth-refreshed pool). One submission
 * path: every run is created QUEUED and admitted by the dispatcher; FIFO within
 * fair-share. Durable across app restarts/redeploys — this replaced the SQLite
 * prototype store.
 *
 * Config: standard PG* env vars (auto-injected when the app has a `postgres` resource;
 * locally set PGHOST to the instance's read_write_dns — OAuth tokens are handled by the
 * pool helper).
 */
import { createLakebasePool } from "@databricks/appkit";
import type { Pool } from "pg";

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
  title: string;
  shape: string;
  nodes: number;
  kind: string;
  ref: string;
  needs_torch: number;
}

const SCHEMA = `
CREATE SCHEMA IF NOT EXISTS broker;
CREATE TABLE IF NOT EXISTS broker.workloads (
  id SERIAL PRIMARY KEY,
  created_utc DOUBLE PRECISION NOT NULL,
  created_by TEXT NOT NULL,
  team TEXT NOT NULL,
  use_case TEXT NOT NULL,
  name TEXT NOT NULL UNIQUE,
  title TEXT DEFAULT '',
  description TEXT DEFAULT '',
  kind TEXT NOT NULL,
  ref TEXT NOT NULL,
  shape TEXT NOT NULL,
  nodes INTEGER NOT NULL DEFAULT 1,
  needs_torch INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS broker.config (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL,
  updated_utc DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS broker.runs (
  id SERIAL PRIMARY KEY,
  workload_id INTEGER NOT NULL REFERENCES broker.workloads(id),
  requested_by TEXT NOT NULL,
  created_utc DOUBLE PRECISION NOT NULL,
  state TEXT NOT NULL DEFAULT 'QUEUED',
  run_id TEXT DEFAULT '',
  detail TEXT DEFAULT '',
  submitted_utc DOUBLE PRECISION,
  finished_utc DOUBLE PRECISION
);
`;

export class Store {
  private pool: Pool;

  constructor() {
    this.pool = createLakebasePool();
  }

  async init() {
    await this.pool.query(SCHEMA);
    await this.pool.query(
      "ALTER TABLE broker.workloads ADD COLUMN IF NOT EXISTS title TEXT DEFAULT ''",
    );
    await this.pool.query(
      "ALTER TABLE broker.workloads ADD COLUMN IF NOT EXISTS description TEXT DEFAULT ''",
    );
  }

  async getConfig(): Promise<Record<string, unknown> | undefined> {
    const res = await this.pool.query("SELECT value FROM broker.config WHERE key = 'broker'");
    return res.rows[0]?.value as Record<string, unknown> | undefined;
  }

  async setConfig(value: Record<string, unknown>): Promise<void> {
    await this.pool.query(
      `INSERT INTO broker.config (key, value, updated_utc) VALUES ('broker', $1, $2)
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_utc = EXCLUDED.updated_utc`,
      [JSON.stringify(value), Date.now() / 1000],
    );
  }

  async upsertRepoWorkload(w: {
    team: string;
    use_case: string;
    name: string;
    title?: string;
    description?: string;
    kind: string;
    ref: string;
    shape: string;
    nodes: number;
    needs_torch: boolean;
  }): Promise<void> {
    await this.pool.query(
      `INSERT INTO broker.workloads
         (created_utc, created_by, team, use_case, name, title, description, kind, ref,
          shape, nodes, needs_torch)
       VALUES ($1, 'repo-sync', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
       ON CONFLICT (name) DO UPDATE SET
         team = EXCLUDED.team, use_case = EXCLUDED.use_case, title = EXCLUDED.title,
         description = EXCLUDED.description, kind = EXCLUDED.kind, ref = EXCLUDED.ref,
         shape = EXCLUDED.shape, nodes = EXCLUDED.nodes, needs_torch = EXCLUDED.needs_torch`,
      [Date.now() / 1000, w.team, w.use_case, w.name, w.title ?? "", w.description ?? "",
       w.kind, w.ref, w.shape, w.nodes, w.needs_torch ? 1 : 0],
    );
  }

  async workloads(teams?: string[]): Promise<Record<string, unknown>[]> {
    if (teams?.length) {
      const res = await this.pool.query(
        "SELECT * FROM broker.workloads WHERE team = ANY($1) ORDER BY use_case, name",
        [teams],
      );
      return res.rows;
    }
    const res = await this.pool.query("SELECT * FROM broker.workloads ORDER BY use_case, name");
    return res.rows;
  }

  async workload(id: number): Promise<Record<string, unknown> | undefined> {
    const res = await this.pool.query("SELECT * FROM broker.workloads WHERE id = $1", [id]);
    return res.rows[0];
  }

  async insertRun(workloadId: number, requestedBy: string): Promise<number> {
    const res = await this.pool.query(
      "INSERT INTO broker.runs (workload_id, requested_by, created_utc) VALUES ($1, $2, $3) RETURNING id",
      [workloadId, requestedBy, Date.now() / 1000],
    );
    return Number(res.rows[0].id);
  }

  async updateRun(id: number, fields: Record<string, string | number | null>): Promise<void> {
    const keys = Object.keys(fields);
    const sets = keys.map((k, i) => `${k} = $${i + 1}`).join(", ");
    await this.pool.query(`UPDATE broker.runs SET ${sets} WHERE id = $${keys.length + 1}`, [
      ...keys.map((k) => fields[k]),
      id,
    ]);
  }

  async runs(states?: string[]): Promise<RunRow[]> {
    const res = await this.pool.query(
      `SELECT r.*, w.team, w.use_case, w.name, w.title, w.shape, w.nodes, w.kind, w.ref,
              w.needs_torch
       FROM broker.runs r JOIN broker.workloads w ON w.id = r.workload_id ORDER BY r.id`,
    );
    const rows: RunRow[] = res.rows.map((row: Record<string, unknown>) => ({
      id: Number(row.id),
      workload_id: Number(row.workload_id),
      requested_by: String(row.requested_by ?? ""),
      created_utc: Number(row.created_utc),
      state: String(row.state),
      run_id: String(row.run_id ?? ""),
      detail: String(row.detail ?? ""),
      team: String(row.team),
      use_case: String(row.use_case),
      name: String(row.name),
      title: String(row.title ?? ""),
      shape: String(row.shape),
      nodes: Number(row.nodes),
      kind: String(row.kind),
      ref: String(row.ref),
      needs_torch: Number(row.needs_torch ?? 0),
    }));
    return states ? rows.filter((r) => states.includes(r.state)) : rows;
  }
}
