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
import { fileURLToPath } from "node:url";
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
  loadConfig,
  repoWorkloads,
  shapeCapacity,
  teamsOf,
  type BrokerConfig,
} from "./domain";
import { Store, type RunRow } from "./store";

const execFileP = promisify(execFile);
const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const REPO_ROOT = path.resolve(APP_ROOT, "..", "..");

export class Broker extends Plugin {
  static manifest = manifest as PluginManifest<"broker">;
  cfg!: BrokerConfig;
  store!: Store;

  async setup() {
    this.cfg = loadConfig(APP_ROOT);
    this.store = new Store(APP_ROOT);
    for (const w of repoWorkloads(REPO_ROOT, this.cfg.catalog_team)) {
      this.store.upsertRepoWorkload(w);
    }
  }

  private principal(req: { headers: Record<string, unknown> }): string {
    const fwd = req.headers["x-forwarded-user"] ?? req.headers["x-forwarded-email"];
    if (typeof fwd === "string" && fwd) return fwd;
    return process.env.DATABRICKS_USER ?? "";
  }

  private inFlight(shape: string, team?: string): number {
    return this.store
      .runs(["SUBMITTED", "RUNNING"])
      .filter((r) => r.shape === shape && (!team || r.team === team))
      .reduce((n, r) => n + r.nodes, 0);
  }

  private shareRatio(teamName: string): number {
    const team = this.cfg.teams.find((t) => t.name === teamName);
    if (!team || team.quota_nodes <= 0) return Number.POSITIVE_INFINITY;
    const used = this.store
      .runs(["SUBMITTED", "RUNNING"])
      .filter((r) => r.team === teamName)
      .reduce((n, r) => n + r.nodes, 0);
    return used / team.quota_nodes;
  }

  private async tick(): Promise<string[]> {
    const events: string[] = [];
    for (const r of this.store.runs(["SUBMITTED", "RUNNING"])) {
      events.push(...(await this.syncRun(r)));
    }
    const queued = this.store
      .runs(["QUEUED"])
      .sort((a, b) => this.shareRatio(a.team) - this.shareRatio(b.team) || a.id - b.id);
    for (const r of queued) {
      const team = this.cfg.teams.find((t) => t.name === r.team);
      const teamRoom = team ? team.quota_nodes - this.inFlight(r.shape, r.team) : 0;
      const cap = shapeCapacity(this.cfg, r.shape, this.inFlight(r.shape));
      if (cap.admittable >= r.nodes && teamRoom >= r.nodes) {
        events.push(await this.submitRun(r));
      }
    }
    return events;
  }

  private async submitRun(r: RunRow): Promise<string> {
    if (r.kind !== "air_yaml") {
      this.store.updateRun(r.id, {
        state: "FAILED",
        detail: `unsupported kind ${r.kind}`,
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: unsupported kind`;
    }
    if (!this.hasLocalAir()) return this.submitViaNotebookHop(r);
    try {
      const { stdout, stderr } = await execFileP("air", ["run", "--file", r.ref], {
        cwd: REPO_ROOT,
        timeout: 300_000,
        env: process.env,
      });
      const text = (stdout + stderr).replace(/\x1b\[[0-9;]*m/g, "");
      const m = text.match(/Job Run ID:\s*(\d+)/);
      if (m) {
        this.store.updateRun(r.id, {
          state: "SUBMITTED",
          run_id: m[1],
          submitted_utc: Date.now() / 1000,
        });
        return `run ${r.id} (${r.name}): submitted as ${m[1]}`;
      }
      this.store.updateRun(r.id, {
        state: "FAILED",
        detail: text.slice(-400),
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: air run gave no Job Run ID`;
    } catch (e) {
      this.store.updateRun(r.id, {
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
            this.store.updateRun(r.id, {
              state: "SUBMITTED",
              run_id: parsed.air_run_id,
              submitted_utc: Date.now() / 1000,
            });
            return `run ${r.id} (${r.name}): submitted as ${parsed.air_run_id} (via notebook hop)`;
          }
          this.store.updateRun(r.id, {
            state: "FAILED",
            detail: (parsed.error ?? "dispatch hop gave no run id").slice(0, 400),
            finished_utc: Date.now() / 1000,
          });
          return `run ${r.id}: dispatch hop failed`;
        }
      }
      this.store.updateRun(r.id, {
        state: "FAILED",
        detail: "dispatch hop did not finish in 10 min",
        finished_utc: Date.now() / 1000,
      });
      return `run ${r.id}: dispatch hop timeout`;
    } catch (e) {
      this.store.updateRun(r.id, {
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
        this.store.updateRun(r.id, { state: "RUNNING" });
        return [`run ${r.id}: running`];
      }
      if (["TERMINATED", "INTERNAL_ERROR", "SKIPPED"].includes(life)) {
        const final =
          result === "SUCCESS" ? "SUCCESS" : result === "CANCELED" ? "CANCELED" : "FAILED";
        this.store.updateRun(r.id, {
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

  injectRoutes(router: IAppRouter): void {
    this.route(router, {
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

    this.route(router, {
      name: "workloads",
      method: "get",
      path: "/workloads",
      handler: async (_req, res) => {
        res.json(this.store.workloads());
      },
    });

    this.route(router, {
      name: "capacity",
      method: "get",
      path: "/capacity",
      handler: async (_req, res) => {
        const shapes = new Set([this.cfg.reservation.accelerator_type, "GPU_1xA10", "GPU_1xH100"]);
        res.json(
          [...shapes].sort().map((s) => shapeCapacity(this.cfg, s, this.inFlight(s))),
        );
      },
    });

    this.route(router, {
      name: "runs",
      method: "get",
      path: "/runs",
      handler: async (_req, res) => {
        const events = await this.tick();
        res.json({ runs: this.store.runs(), events });
      },
    });

    this.route(router, {
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
        const workloadId = Number((req.body as { workloadId?: number })?.workloadId);
        const w = this.store.workload(workloadId);
        if (!w) {
          res.status(404).json({ error: `workload ${workloadId} not found` });
          return;
        }
        if (!myTeams.some((t) => t.name === w.team)) {
          res.status(403).json({ error: `not a member of team ${w.team}` });
          return;
        }
        const runId = this.store.insertRun(workloadId, principal);
        const events = await this.tick();
        res.json({ id: runId, events });
      },
    });
  }
}

export const broker = toPlugin(Broker);
