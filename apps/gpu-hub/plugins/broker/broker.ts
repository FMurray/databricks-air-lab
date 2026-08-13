/**
 * Broker plugin: run admission, capacity, and attribution for a shared GPU pool.
 *
 * Routes (identity = unified auth: x-forwarded-user in production, DATABRICKS_USER or
 * the SDK's resolved user in local dev — users never pick a team):
 *   GET  /me          -> principal + teams
 *   GET  /workloads   -> catalog (repo workloads + registered)
 *   GET  /capacity    -> per-shape capacity from declared config vs in-flight
 *   GET  /runs        -> all runs (also advances the dispatcher)
 *   POST /runs        -> { workloadId } create a run (gated, QUEUED)
 *
 * Dispatch: FIFO within fair-share; air_yaml workloads submit via the air CLI subprocess
 * (local dev; the only supported path for Gen-AI tasks — docs/06). Hosted dispatch for
 * air_yaml goes through a notebook job that shells the vendored CLI (uat/checks pattern).
 */
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";
import {
  Plugin,
  toPlugin,
  getWorkspaceClient,
  type IAppRouter,
  type PluginManifest,
} from "@databricks/appkit";
import manifest from "./manifest.json";
import {
  findAppRoot,
  loadConfig,
  parseConfig,
  repoWorkloads,
  shapeCapacity,
  teamsOf,
  type BrokerConfig,
} from "./domain";
import { Store, type RunRow } from "./store";

const execFileP = promisify(execFile);
const APP_ROOT = findAppRoot(import.meta.url);
const REPO_ROOT = path.resolve(APP_ROOT, "..", "..");

export class Broker extends Plugin {
  static manifest = manifest as PluginManifest<"broker">;
  cfg!: BrokerConfig;
  store!: Store;

  async setup() {
    this.store = new Store();
    await this.store.init();
    // Config lives in Lakebase (files can't ship real team emails through a public repo's
    // deploy sync — receipt: hosted bundle contained only broker.example.json). The file
    // is the bootstrap: first run with a real config seeds the DB.
    const fileCfg = loadConfig(APP_ROOT);
    const dbRaw = await this.store.getConfig();
    const dbCfg = dbRaw ? parseConfig(dbRaw) : undefined;
    const fileIsReal = fileCfg.teams.some((t) =>
      t.members.some((m) => !m.endsWith("@example.com")),
    );
    if (fileIsReal) {
      await this.store.setConfig(JSON.parse(JSON.stringify(fileCfg)) as Record<string, unknown>);
      this.cfg = fileCfg;
    } else {
      this.cfg = dbCfg ?? fileCfg;
    }
    for (const t of this.cfg.teams) {
      for (const u of t.use_cases) {
        await this.store.upsertProject({
          name: u.name,
          teams: [t.name],
          metadata: { business_app: "", description: u.description ?? "" },
        });
      }
    }
    const catalog = repoWorkloads(APP_ROOT, this.cfg.catalog_team);
    const byKey = new Map<string, typeof catalog>();
    for (const c of catalog) {
      const key = c.workload_key ?? c.name;
      if (!byKey.has(key)) byKey.set(key, []);
      byKey.get(key)!.push(c);
    }
    for (const [key, configs] of byKey) {
      // the base configuration (name == key) carries the workload's title/description
      const base = configs.find((c) => c.name.split("/").pop() === key) ?? configs[0];
      const wid = await this.store.upsertWorkload({
        workload_key: key,
        team: base.team,
        use_case: base.use_case,
        title: base.title ?? key,
        description: base.description ?? "",
      });
      for (const c of configs) {
        await this.store.upsertConfiguration({
          workload_id: wid,
          workload_key: key,
          experiment_name: c.experiment_name ?? "",
          team: c.team,
          use_case: c.use_case,
          name: c.name,
          title: c.title,
          description: c.description,
          author: c.author,
          kind: c.kind,
          ref: c.ref,
          shape: c.shape,
          nodes: c.nodes,
          needs_torch: c.needs_torch,
        });
      }
    }
  }

  private principal(req: { headers: Record<string, unknown> }): string {
    // Apps sends the numeric user ID in x-forwarded-user; the EMAIL (what team config
    // uses) rides x-forwarded-email — prefer it. (Receipt: hosted /me showed
    // 3066118190281634@<workspace-id> before this swap.)
    const fwd = req.headers["x-forwarded-email"] ?? req.headers["x-forwarded-user"];
    if (typeof fwd === "string" && fwd) return fwd;
    return process.env.DATABRICKS_USER ?? "";
  }

  private inFlightOf(active: RunRow[], shape: string, team?: string): number {
    return active
      .filter((r) => r.shape === shape && (!team || r.team === team))
      .reduce((n, r) => n + r.nodes, 0);
  }

  private shareRatioOf(active: RunRow[], teamName: string): number {
    const team = this.cfg.teams.find((t) => t.name === teamName);
    if (!team || team.quota_nodes <= 0) return Number.POSITIVE_INFINITY;
    const used = active.filter((r) => r.team === teamName).reduce((n, r) => n + r.nodes, 0);
    return used / team.quota_nodes;
  }

  private async tick(): Promise<string[]> {
    const events: string[] = [];
    for (const r of await this.store.runs(["SUBMITTED", "RUNNING"])) {
      events.push(...(await this.syncRun(r)));
    }
    const active = await this.store.runs(["SUBMITTED", "RUNNING"]);
    const queued = (await this.store.runs(["QUEUED"])).sort(
      (a, b) =>
        this.shareRatioOf(active, a.team) - this.shareRatioOf(active, b.team) || a.id - b.id,
    );
    for (const r of queued) {
      const team = this.cfg.teams.find((t) => t.name === r.team);
      const teamRoom = team
        ? team.quota_nodes - this.inFlightOf(active, r.shape, r.team)
        : 0;
      const cap = shapeCapacity(this.cfg, r.shape, this.inFlightOf(active, r.shape));
      if (cap.admittable >= r.nodes && teamRoom >= r.nodes) {
        events.push(await this.submitRun(r));
        active.push(r); // count it against capacity for the rest of this pass
      }
    }
    return events;
  }

  private async submitRun(r: RunRow): Promise<string> {
    if (r.kind !== "air_yaml") {
      await this.store.updateRun(r.id, {
        state: "FAILED",
        detail: `unsupported kind ${r.kind}`,
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: unsupported kind`;
    }
    if (!this.hasLocalAir()) return this.submitViaNotebookHop(r);
    try {
      let fileToRun = r.ref;
      const options = r.options ? (JSON.parse(r.options) as string[]) : [];
      if (options.length) {
        // compose next to the base file so the relative root_path stays correct
        const composedRel = path.join(path.dirname(r.ref), `.composed-${r.id}.yaml`);
        const params = r.params ? (JSON.parse(r.params) as Record<string, string>) : {};
        const args = ["run", "--with", "pyyaml", "python", "-m", "utils.composition.compose",
                      r.ref, "--options", options.join(","), "--out", composedRel];
        for (const [k, v] of Object.entries(params)) args.push("--param", `${k}=${v}`);
        await execFileP("uv", args, { cwd: REPO_ROOT, timeout: 120_000, env: process.env });
        fileToRun = composedRel;
      }
      const { stdout, stderr } = await execFileP("air", ["run", "--file", fileToRun], {
        cwd: REPO_ROOT,
        timeout: 300_000,
        env: process.env,
      });
      const text = (stdout + stderr).replace(/\x1b\[[0-9;]*m/g, "");
      const m = text.match(/Job Run ID:\s*(\d+)/);
      if (m) {
        await this.store.updateRun(r.id, {
          state: "SUBMITTED",
          run_id: m[1],
          submitted_utc: Date.now() / 1000,
        });
        return `run ${r.id} (${r.name}): submitted as ${m[1]}`;
      }
      await this.store.updateRun(r.id, {
        state: "FAILED",
        detail: text.slice(-400),
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: air run gave no Job Run ID`;
    } catch (e) {
      await this.store.updateRun(r.id, {
        state: "FAILED",
        detail: String(e).slice(0, 400),
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: air run failed`;
    }
  }

  private hasLocalAir(): boolean {
    const paths = (process.env.PATH ?? "").split(":");
    return paths.some((p) => {
      try {
        return Boolean(p) && fs.existsSync(path.join(p, "air"));
      } catch {
        return false;
      }
    });
  }

  /** Hosted mode: no air binary / repo checkout — submit the dispatch notebook (CPU
   *  serverless), which installs the vendored CLI and runs the workload from the
   *  workspace mirror, exiting with the AIR run id. Pattern verified by
   *  uat/checks/air-cli-from-notebook (docs/06). */
  private async submitViaNotebookHop(r: RunRow): Promise<string> {
    try {
      const client = getWorkspaceClient({});
      const submit = (await client.apiClient.request({
        path: "/api/2.2/jobs/runs/submit",
        method: "POST",
        headers: new Headers(),
        raw: false,
        payload: {
          run_name: `gpu-hub-dispatch-${r.id}`,
          tasks: [
            {
              task_key: "dispatch",
              notebook_task: {
                notebook_path: "/Workspace/Shared/databricks-air-lab/uat/dispatch-workload",
                base_parameters: { workload_ref: r.ref },
              },
              timeout_seconds: 1200,
            },
          ],
        },
      })) as { run_id: number };
      // poll the hop to terminal (it only installs the CLI + submits — minutes)
      for (let i = 0; i < 40; i++) {
        await new Promise((ok) => setTimeout(ok, 15_000));
        const hop = (await client.apiClient.request({
          path: "/api/2.2/jobs/runs/get",
          method: "GET",
          query: { run_id: String(submit.run_id) },
          headers: new Headers(),
          raw: false,
        })) as {
          state: { life_cycle_state: string };
          tasks: { run_id: number }[];
        };
        if (["TERMINATED", "INTERNAL_ERROR", "SKIPPED"].includes(hop.state.life_cycle_state)) {
          const out = (await client.apiClient.request({
            path: "/api/2.2/jobs/runs/get-output",
            method: "GET",
            query: { run_id: String(hop.tasks[0].run_id) },
            headers: new Headers(),
            raw: false,
          })) as { notebook_output?: { result?: string } };
          const parsed = JSON.parse(out.notebook_output?.result ?? "{}") as {
            air_run_id?: string;
            error?: string;
          };
          if (parsed.air_run_id) {
            await this.store.updateRun(r.id, {
              state: "SUBMITTED",
              run_id: parsed.air_run_id,
              submitted_utc: Date.now() / 1000,
            });
            return `run ${r.id} (${r.name}): submitted as ${parsed.air_run_id} (via notebook hop)`;
          }
          await this.store.updateRun(r.id, {
            state: "FAILED",
            detail: (parsed.error ?? "dispatch hop gave no run id").slice(0, 400),
            finished_utc: Date.now() / 1000,
          });
          return `run ${r.id}: dispatch hop failed`;
        }
      }
      await this.store.updateRun(r.id, {
        state: "FAILED",
        detail: "dispatch hop did not finish in 10 min",
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: dispatch hop timeout`;
    } catch (e) {
      await this.store.updateRun(r.id, {
        state: "FAILED",
        detail: String(e).slice(0, 400),
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: dispatch hop error`;
    }
  }

  private async syncRun(r: RunRow): Promise<string[]> {
    if (!r.run_id) return [];
    try {
      const client = getWorkspaceClient({});
      const run = (await client.apiClient.request({
        path: "/api/2.2/jobs/runs/get",
        method: "GET",
        query: { run_id: r.run_id },
        headers: new Headers(),
        raw: false,
      })) as { state: { life_cycle_state: string; result_state?: string; state_message?: string } };
      const life = run.state.life_cycle_state;
      const result = run.state.result_state ?? "";
      if (life === "RUNNING" && r.state !== "RUNNING") {
        await this.store.updateRun(r.id, { state: "RUNNING" });
        return [`run ${r.id}: running`];
      }
      if (["TERMINATED", "INTERNAL_ERROR", "SKIPPED"].includes(life)) {
        const final =
          result === "SUCCESS" ? "SUCCESS" : result === "CANCELED" ? "CANCELED" : "FAILED";
        await this.store.updateRun(r.id, {
          state: final,
          detail: (run.state.state_message ?? "").slice(0, 400),
          finished_utc: Date.now() / 1000,
        });
        return [`run ${r.id}: ${final.toLowerCase()}`];
      }
    } catch {
      /* transient; next tick retries */
    }
    return [];
  }

  /**
   * AppKit's route() registers the raw async handler on the Express router with no error
   * boundary; Express 4 does not catch a rejected async handler, so any throw (e.g. a
   * dropped Lakebase connection mid-query) becomes an unhandledRejection and, under Node's
   * default, exits the process. Receipt: the hub crashed on GET /runs when the pooled
   * Postgres connection was terminated after an idle gap (Store.runs -> tick -> handler,
   * "Connection terminated due to connection timeout"). Wrap every handler so a failure
   * returns 500 and the server stays up.
   */
  private safeRoute(router: IAppRouter, config: Parameters<Broker["route"]>[1]): void {
    const handler = config.handler;
    this.route(router, {
      ...config,
      handler: async (req, res) => {
        try {
          await handler(req, res);
        } catch (err) {
          console.error(`[broker] ${config.method.toUpperCase()} ${config.path} failed:`, err);
          if (!res.headersSent) res.status(500).json({ error: "internal error" });
        }
      },
    });
  }

  injectRoutes(router: IAppRouter): void {
    this.safeRoute(router, {
      name: "me",
      method: "get",
      path: "/me",
      handler: async (req, res) => {
        const principal = this.principal(req as never);
        res.json({
          principal,
          teams: teamsOf(this.cfg, principal).map((t) => ({
            name: t.name,
            quota_nodes: t.quota_nodes,
            use_cases: t.use_cases,
          })),
        });
      },
    });

    this.safeRoute(router, {
      name: "teams",
      method: "get",
      path: "/teams",
      handler: async (req, res) => {
        // The team directory. People belong to teams (possibly several); a team's use cases
        // are its billable projects. Attribution (every run carries team + project) and
        // observability (in-flight nodes vs the team's fair-share quota) both key off this,
        // so the page reports each team's live utilization and run history.
        const principal = this.principal(req as never).toLowerCase();
        const allRuns = await this.store.runs();
        res.json(
          this.cfg.teams.map((t) => {
            const teamRuns = allRuns.filter((r) => r.team === t.name);
            const activeNodes = teamRuns
              .filter((r) => r.state === "SUBMITTED" || r.state === "RUNNING")
              .reduce((n, r) => n + r.nodes, 0);
            return {
              name: t.name,
              quota_nodes: t.quota_nodes,
              members: t.members,
              projects: t.use_cases.map((u) => ({
                name: u.name,
                description: u.description ?? "",
                run_count: teamRuns.filter((r) => r.project === u.name).length,
              })),
              run_count: teamRuns.length,
              active_nodes: activeNodes,
              is_mine: t.members.some((m) => m.toLowerCase() === principal),
            };
          }),
        );
      },
    });

    this.safeRoute(router, {
      name: "workloads",
      method: "get",
      path: "/workloads",
      handler: async (req, res) => {
        // workloads with nested configurations, plus run history relative to the caller
        const principal = this.principal(req as never);
        const myTeams = new Set(teamsOf(this.cfg, principal).map((t) => t.name));
        const allRuns = await this.store.runs();
        const stats = new Map<number, { run_count: number; my_run_count: number;
          team_run_count: number }>();
        for (const r of allRuns) {
          const s = stats.get(r.workload_id) ?? {
            run_count: 0, my_run_count: 0, team_run_count: 0 };
          s.run_count++;
          if (r.requested_by === principal) s.my_run_count++;
          if (myTeams.has(r.team)) s.team_run_count++;
          stats.set(r.workload_id, s);
        }
        const configs = await this.store.configurations();
        const workloads = await this.store.workloads();
        res.json(workloads.map((w) => ({
          ...w,
          ...(stats.get(Number(w.id)) ?? { run_count: 0, my_run_count: 0, team_run_count: 0 }),
          configurations: configs.filter((c) => Number(c.workload_id) === Number(w.id)),
        })));
      },
    });

    this.safeRoute(router, {
      name: "projects",
      method: "get",
      path: "/projects",
      handler: async (req, res) => {
        const principal = this.principal(req as never);
        const myTeams = new Set(teamsOf(this.cfg, principal).map((t) => t.name));
        const rows = await this.store.projects();
        res.json(rows.map((p) => ({
          ...p,
          billable_by_me: (p.teams as string[]).some((t) => myTeams.has(t)),
        })));
      },
    });

    this.safeRoute(router, {
      name: "options",
      method: "get",
      path: "/options",
      handler: async (_req, res) => {
        const p = path.join(APP_ROOT, "plugins", "broker", "options.json");
        res.json(fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : []);
      },
    });

    this.safeRoute(router, {
      name: "capacity",
      method: "get",
      path: "/capacity",
      handler: async (_req, res) => {
        const shapes = new Set([this.cfg.reservation.accelerator_type, "GPU_1xA10", "GPU_1xH100"]);
        const active = await this.store.runs(["SUBMITTED", "RUNNING"]);
        res.json(
          [...shapes]
            .sort()
            .map((s) => shapeCapacity(this.cfg, s, this.inFlightOf(active, s))),
        );
      },
    });

    this.safeRoute(router, {
      name: "runs",
      method: "get",
      path: "/runs",
      handler: async (_req, res) => {
        const events = await this.tick();
        res.json({ runs: await this.store.runs(), events });
      },
    });

    this.safeRoute(router, {
      name: "createRun",
      method: "post",
      path: "/runs",
      handler: async (req, res) => {
        const principal = this.principal(req as never);
        const myTeams = teamsOf(this.cfg, principal);
        if (!myTeams.length) {
          res.status(403).json({ error: "not a member of any team — access is read-only" });
          return;
        }
        const body = req.body as { configurationId?: number; project?: string;
          options?: string[]; params?: Record<string, string> };
        const configurationId = Number(body?.configurationId);
        const c = await this.store.configuration(configurationId);
        if (!c) {
          res.status(404).json({ error: `configuration ${configurationId} not found` });
          return;
        }
        if (!myTeams.some((t) => t.name === c.team)) {
          res.status(403).json({ error: `not a member of team ${c.team}` });
          return;
        }
        // the requester pays: the project must be billable by one of the requester's teams
        const projectName = String(body?.project ?? "");
        if (!projectName) {
          res.status(400).json({ error: "a project is required — the run bills to it" });
          return;
        }
        const project = (await this.store.projects()).find((p) => p.name === projectName);
        const myTeamNames = new Set(myTeams.map((t) => t.name));
        if (!project || !(project.teams as string[]).some((t) => myTeamNames.has(t))) {
          res.status(400).json({ error: `project '${projectName}' is not billable by your teams` });
          return;
        }
        const runId = await this.store.insertRun(
          configurationId, principal, projectName, body?.options ?? [], body?.params ?? {});
        const events = await this.tick();
        res.json({ id: runId, events });
      },
    });
  }
}

export const broker = toPlugin(Broker);
