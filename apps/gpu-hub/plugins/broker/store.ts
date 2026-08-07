/**
 * State in Lakebase (Postgres, OAuth-refreshed pool).
 *
 * Domain (exact language):
 *   Workload       - abstract user need. Groups configurations. No experiment key.
 *   Configuration  - concrete recipe (YAML/notebook + shape + nodes). One experiment each.
 *   Run            - one execution of a configuration. One job run. One MLflow run.
 *                    The requester pays: use_case is selected at request time.
 *   Result         - one receipt from a run (verification-skill conventions).
 *
 * Migration from the v1 schema (concrete rows in broker.workloads) happens in init():
 * the old table renames to broker.configurations; runs.workload_id renames to
 * configuration_id; the new abstract broker.workloads table is created fresh.
 */
import { createLakebasePool } from "@databricks/appkit";
import type { Pool } from "pg";

export interface RunRow {
  id: number;
  configuration_id: number;
  requested_by: string;
  use_case: string;
  project: string;
  options: string;
  params: string;
  created_utc: number;
  state: string;
  run_id: string;
  detail: string;
  // joined from configurations:
  team: string;
  config_use_case: string;
  name: string;
  title: string;
  shape: string;
  nodes: number;
  kind: string;
  ref: string;
  needs_torch: number;
  workload_id: number;
}

const SCHEMA = `
CREATE SCHEMA IF NOT EXISTS broker;
CREATE TABLE IF NOT EXISTS broker.workloads (
  id SERIAL PRIMARY KEY,
  workload_key TEXT NOT NULL UNIQUE,
  team TEXT NOT NULL DEFAULT '',
  use_case TEXT NOT NULL DEFAULT '',
  title TEXT DEFAULT '',
  description TEXT DEFAULT '',
  created_utc DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS broker.configurations (
  id SERIAL PRIMARY KEY,
  created_utc DOUBLE PRECISION NOT NULL,
  created_by TEXT NOT NULL,
  team TEXT NOT NULL,
  use_case TEXT NOT NULL,
  name TEXT NOT NULL UNIQUE,
  title TEXT DEFAULT '',
  description TEXT DEFAULT '',
  author TEXT DEFAULT '',
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
  configuration_id INTEGER NOT NULL,
  requested_by TEXT NOT NULL,
  created_utc DOUBLE PRECISION NOT NULL,
  state TEXT NOT NULL DEFAULT 'QUEUED',
  run_id TEXT DEFAULT '',
  detail TEXT DEFAULT '',
  submitted_utc DOUBLE PRECISION,
  finished_utc DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS broker.projects (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  teams JSONB NOT NULL DEFAULT '[]',
  metadata JSONB NOT NULL DEFAULT '{}',
  created_utc DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS broker.results (
  id SERIAL PRIMARY KEY,
  run_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  value TEXT DEFAULT '',
  label TEXT DEFAULT '',
  evidence TEXT DEFAULT ''
);
`;

const RENAMES = [
  // v1 -> v2: the old concrete table becomes configurations
  `DO $$ BEGIN
     IF EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema='broker' AND table_name='workloads' AND column_name='ref')
        AND NOT EXISTS (SELECT 1 FROM information_schema.tables
                WHERE table_schema='broker' AND table_name='configurations') THEN
       ALTER TABLE broker.workloads RENAME TO configurations;
     END IF;
   END $$;`,
  `DO $$ BEGIN
     IF EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema='broker' AND table_name='runs' AND column_name='workload_id') THEN
       ALTER TABLE broker.runs RENAME COLUMN workload_id TO configuration_id;
     END IF;
   END $$;`,
];

const ADDS = [
  "ALTER TABLE broker.configurations ADD COLUMN IF NOT EXISTS workload_id INTEGER",
  "ALTER TABLE broker.configurations ADD COLUMN IF NOT EXISTS workload_key TEXT DEFAULT ''",
  "ALTER TABLE broker.configurations ADD COLUMN IF NOT EXISTS experiment_name TEXT DEFAULT ''",
  "ALTER TABLE broker.runs ADD COLUMN IF NOT EXISTS use_case TEXT DEFAULT ''",
  "ALTER TABLE broker.runs ADD COLUMN IF NOT EXISTS options TEXT DEFAULT ''",
  "ALTER TABLE broker.runs ADD COLUMN IF NOT EXISTS params TEXT DEFAULT ''",
  "ALTER TABLE broker.runs ADD COLUMN IF NOT EXISTS project TEXT DEFAULT ''",
  "ALTER TABLE broker.workloads ADD COLUMN IF NOT EXISTS project_id INTEGER",
];

export class Store {
  private pool: Pool;

  constructor() {
    this.pool = createLakebasePool();
  }

  async init() {
    // renames run BEFORE the schema, so the fresh abstract 'workloads' table is only
    // created after the old concrete one has moved aside
    await this.pool.query("CREATE SCHEMA IF NOT EXISTS broker");
    for (const m of RENAMES) await this.pool.query(m);
    await this.pool.query(SCHEMA);
    for (const m of ADDS) await this.pool.query(m);
  }

  // ---------- config ----------
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

  // ---------- projects ----------
  async upsertProject(p: {
    name: string;
    teams: string[];
    metadata: Record<string, string>;
  }): Promise<number> {
    const res = await this.pool.query(
      `INSERT INTO broker.projects (name, teams, metadata, created_utc)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (name) DO UPDATE SET teams = EXCLUDED.teams
       RETURNING id`,
      [p.name, JSON.stringify(p.teams), JSON.stringify(p.metadata), Date.now() / 1000],
    );
    return Number(res.rows[0].id);
  }

  async projects(): Promise<Record<string, unknown>[]> {
    const res = await this.pool.query("SELECT * FROM broker.projects ORDER BY name");
    return res.rows;
  }

  // ---------- workloads (abstract) ----------
  async upsertWorkload(w: {
    workload_key: string;
    team: string;
    use_case: string;
    title: string;
    description: string;
  }): Promise<number> {
    const res = await this.pool.query(
      `INSERT INTO broker.workloads (workload_key, team, use_case, title, description, created_utc)
       VALUES ($1, $2, $3, $4, $5, $6)
       ON CONFLICT (workload_key) DO UPDATE SET
         team = EXCLUDED.team, use_case = EXCLUDED.use_case,
         title = EXCLUDED.title, description = EXCLUDED.description
       RETURNING id`,
      [w.workload_key, w.team, w.use_case, w.title, w.description, Date.now() / 1000],
    );
    return Number(res.rows[0].id);
  }

  async workloads(): Promise<Record<string, unknown>[]> {
    const res = await this.pool.query("SELECT * FROM broker.workloads ORDER BY use_case, title");
    return res.rows;
  }

  // ---------- configurations (concrete) ----------
  async upsertConfiguration(c: {
    workload_id: number;
    workload_key: string;
    experiment_name: string;
    team: string;
    use_case: string;
    name: string;
    title?: string;
    description?: string;
    author?: string;
    kind: string;
    ref: string;
    shape: string;
    nodes: number;
    needs_torch: boolean;
  }): Promise<void> {
    await this.pool.query(
      `INSERT INTO broker.configurations
         (created_utc, created_by, workload_id, workload_key, experiment_name, team, use_case,
          name, title, description, author, kind, ref, shape, nodes, needs_torch)
       VALUES ($1, 'repo-sync', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
       ON CONFLICT (name) DO UPDATE SET
         workload_id = EXCLUDED.workload_id, workload_key = EXCLUDED.workload_key,
         experiment_name = EXCLUDED.experiment_name, team = EXCLUDED.team,
         use_case = EXCLUDED.use_case, title = EXCLUDED.title,
         description = EXCLUDED.description, author = EXCLUDED.author,
         kind = EXCLUDED.kind, ref = EXCLUDED.ref, shape = EXCLUDED.shape,
         nodes = EXCLUDED.nodes, needs_torch = EXCLUDED.needs_torch`,
      [Date.now() / 1000, c.workload_id, c.workload_key, c.experiment_name, c.team, c.use_case,
       c.name, c.title ?? "", c.description ?? "", c.author ?? "", c.kind, c.ref, c.shape,
       c.nodes, c.needs_torch ? 1 : 0],
    );
  }

  async configurations(): Promise<Record<string, unknown>[]> {
    const res = await this.pool.query(
      "SELECT * FROM broker.configurations ORDER BY workload_key, name",
    );
    return res.rows;
  }

  async configuration(id: number): Promise<Record<string, unknown> | undefined> {
    const res = await this.pool.query("SELECT * FROM broker.configurations WHERE id = $1", [id]);
    return res.rows[0];
  }

  // ---------- runs ----------
  async insertRun(configurationId: number, requestedBy: string, project: string,
                  options: string[], params: Record<string, string>): Promise<number> {
    const res = await this.pool.query(
      `INSERT INTO broker.runs (configuration_id, requested_by, project, options, params,
                                created_utc)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`,
      [configurationId, requestedBy, project, JSON.stringify(options),
       JSON.stringify(params), Date.now() / 1000],
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
      `SELECT r.*, c.team, c.use_case AS config_use_case, c.name, c.title, c.shape, c.nodes,
              c.kind, c.ref, c.needs_torch, c.workload_id
       FROM broker.runs r JOIN broker.configurations c ON c.id = r.configuration_id
       ORDER BY r.id`,
    );
    const rows: RunRow[] = res.rows.map((row: Record<string, unknown>) => ({
      id: Number(row.id),
      configuration_id: Number(row.configuration_id),
      requested_by: String(row.requested_by ?? ""),
      use_case: String(row.use_case ?? "") || String(row.config_use_case ?? ""),
      project: String(row.project ?? "") || String(row.use_case ?? ""),
      options: String(row.options ?? ""),
      params: String(row.params ?? ""),
      created_utc: Number(row.created_utc),
      state: String(row.state),
      run_id: String(row.run_id ?? ""),
      detail: String(row.detail ?? ""),
      team: String(row.team),
      config_use_case: String(row.config_use_case ?? ""),
      name: String(row.name),
      title: String(row.title ?? ""),
      shape: String(row.shape),
      nodes: Number(row.nodes),
      kind: String(row.kind),
      ref: String(row.ref),
      needs_torch: Number(row.needs_torch ?? 0),
      workload_id: Number(row.workload_id ?? 0),
    }));
    return states ? rows.filter((r) => states.includes(r.state)) : rows;
  }
}
