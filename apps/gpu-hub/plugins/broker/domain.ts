/**
 * Domain: Team -> Use case -> Workload -> Run.
 * Config (teams/quotas/use cases) comes from config/broker.yaml; the workload catalog is
 * the repo's workloads/ directory (git is the source of truth). Ported from the Python
 * prototype in apps/training-hub — semantics receipt-backed by the UAT suite (docs/06).
 */
import fs from "node:fs";
import path from "node:path";


export interface UseCase {
  name: string;
  description?: string;
}
export interface Team {
  name: string;
  quota_nodes: number;
  members: string[];
  use_cases: UseCase[];
}
export interface BrokerConfig {
  reservation: {
    total_nodes: number;
    gpus_per_node: number;
    accelerator_type: string;
    region?: string;
  };
  platform_quotas: Record<string, number>;
  catalog_team: string;
  teams: Team[];
}

const CONFIG_CANDIDATES = ["config/broker.json", "config/broker.example.json"];

/** App root that works in both layouts: tsx-from-source (module lives under plugins/)
 *  and compiled dist (module lives under dist/plugins/) — anchor on cwd, then walk up
 *  from the module until a config/ dir appears. */
export function findAppRoot(moduleUrl: string): string {
  const candidates = [process.cwd()];
  let d = path.dirname(new URL(moduleUrl).pathname);
  for (let i = 0; i < 5; i++) {
    candidates.push(d);
    d = path.dirname(d);
  }
  for (const c of candidates) {
    if (CONFIG_CANDIDATES.some((rel) => fs.existsSync(path.join(c, rel)))) return c;
  }
  return process.cwd();
}

export function parseConfig(raw: Record<string, unknown>): BrokerConfig {
  const teams: Team[] = ((raw.teams as Record<string, unknown>[]) ?? []).map((t) => ({
    name: String(t.name ?? ""),
    quota_nodes: Number(t.quota_nodes ?? 0),
    members: (t.members as string[]) ?? [],
    use_cases: ((t.use_cases as (UseCase | string)[]) ?? []).map((u) =>
      typeof u === "string" ? { name: u } : u,
    ),
  }));
  return {
    reservation: raw.reservation as BrokerConfig["reservation"],
    platform_quotas: (raw.platform_quotas as Record<string, number>) ?? {},
    catalog_team: (raw.catalog_team as string) ?? teams[0]?.name ?? "",
    teams,
  };
}

export function loadConfig(appRoot: string): BrokerConfig {
  for (const rel of CONFIG_CANDIDATES) {
    const p = path.join(appRoot, rel);
    if (fs.existsSync(p)) {
      return parseConfig(JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, unknown>);
    }
  }
  throw new Error("no broker config found (config/broker.json)");
}

export function teamsOf(cfg: BrokerConfig, principal: string | undefined): Team[] {
  if (!principal) return [];
  const needle = principal.toLowerCase();
  return cfg.teams.filter((t) => t.members.some((m) => m.toLowerCase() === needle));
}

// ---------- capacity ----------
export interface ShapeCapacity {
  shape: string;
  platformQuotaNodes: number;
  reservedNodes: number;
  inFlight: number;
  admittable: number;
}

export function shapeCapacity(cfg: BrokerConfig, shape: string, inFlight: number): ShapeCapacity {
  const reserved = shape === cfg.reservation.accelerator_type ? cfg.reservation.total_nodes : 0;
  const quota = cfg.platform_quotas[shape] ?? 0;
  const caps = [quota, reserved].filter((c) => c > 0);
  // no configured cap = on-demand shape: nothing binds platform-side, team quota still gates
  const bound = caps.length ? Math.min(...caps) : 999;
  return {
    shape,
    platformQuotaNodes: quota,
    reservedNodes: reserved,
    inFlight,
    admittable: Math.max(0, bound - inFlight),
  };
}

// ---------- catalog: the repo's workloads/ directory ----------
export interface CatalogWorkload {
  name: string;
  title?: string;
  description?: string;
  author?: string;
  kind: "air_yaml";
  ref: string;
  team: string;
  use_case: string;
  shape: string;
  nodes: number;
  needs_torch: boolean;
}



export function repoWorkloads(appRoot: string, team: string): CatalogWorkload[] {
  // The catalog is generated at build time from the repo's workloads/ directory
  // (scripts/gen_catalog.py) — no YAML parsing at runtime, no repo checkout needed hosted.
  const p = path.join(appRoot, "plugins", "broker", "catalog.json");
  if (!fs.existsSync(p)) return [];
  const raw = JSON.parse(fs.readFileSync(p, "utf8")) as CatalogWorkload[];
  return raw.map((w) => ({ ...w, team }));
}
