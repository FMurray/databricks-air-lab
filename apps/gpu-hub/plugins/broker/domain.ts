/**
 * Domain: Team -> Use case -> Workload -> Run.
 * Config (teams/quotas/use cases) comes from config/broker.yaml; the workload catalog is
 * the repo's workloads/ directory (git is the source of truth). Ported from the Python
 * prototype in apps/training-hub — semantics receipt-backed by the UAT suite (docs/06).
 */
import fs from "node:fs";
import path from "node:path";
import * as yaml from "js-yaml";

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

const CONFIG_CANDIDATES = ["config/broker.yaml", "config/broker.example.yaml"];

export function loadConfig(appRoot: string): BrokerConfig {
  for (const rel of CONFIG_CANDIDATES) {
    const p = path.join(appRoot, rel);
    if (fs.existsSync(p)) {
      const raw = yaml.load(fs.readFileSync(p, "utf8")) as Record<string, unknown>;
      const teams = ((raw.teams as Record<string, unknown>[]) ?? []).map((t) => ({
        use_cases: [],
        members: [],
        ...(t as object),
      })) as unknown as Team[];
      return {
        reservation: raw.reservation as BrokerConfig["reservation"],
        platform_quotas: (raw.platform_quotas as Record<string, number>) ?? {},
        catalog_team: (raw.catalog_team as string) ?? teams[0]?.name ?? "",
        teams,
      };
    }
  }
  throw new Error("no broker config found (config/broker.yaml)");
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
  kind: "air_yaml";
  ref: string;
  team: string;
  use_case: string;
  shape: string;
  nodes: number;
  needs_torch: boolean;
}

const FAMILY_USE_CASE: Record<string, string> = {
  "node-acceptance": "node-acceptance",
  probes: "env-diagnostics",
  "scheduling-isolation": "scheduling-isolation",
  "vendored-wheels": "dependencies",
  "env-flexibility": "dependencies",
  "docker-otel-zerobus": "telemetry",
  "foundation-models": "foundation-models",
  tabicl: "classic-ml",
  xgboost: "classic-ml",
  "multi-language": "multi-language",
  multinode: "node-acceptance",
  "rdma-stress": "node-acceptance",
};

function useCaseFor(stem: string, raw: Record<string, unknown>): string {
  const snap =
    ((raw.code_source as Record<string, unknown>)?.snapshot as Record<string, unknown>) ?? {};
  const hints = [stem, ...((snap.include_paths as string[]) ?? [])];
  for (const hint of hints) {
    for (const [key, uc] of Object.entries(FAMILY_USE_CASE)) {
      if (String(hint).includes(key)) return uc;
    }
  }
  return "env-diagnostics";
}

export function repoWorkloads(repoRoot: string, team: string): CatalogWorkload[] {
  const dir = path.join(repoRoot, "workloads");
  if (!fs.existsSync(dir)) return [];
  const files = new Map<string, string>();
  const walk = (d: string) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, entry.name);
      if (entry.isDirectory()) walk(p);
      else if (entry.name.endsWith(".yaml")) {
        const stem = entry.name.replace(".example.yaml", "").replace(".yaml", "");
        const key = path.relative(dir, path.join(d, stem));
        const existing = files.get(key);
        if (!existing || existing.includes(".example")) files.set(key, p);
      }
    }
  };
  walk(dir);
  const out: CatalogWorkload[] = [];
  for (const [key, p] of [...files.entries()].sort()) {
    let raw: Record<string, unknown>;
    try {
      raw = yaml.load(fs.readFileSync(p, "utf8")) as Record<string, unknown>;
    } catch {
      continue;
    }
    if (!raw || typeof raw !== "object" || !("command" in raw)) continue;
    const comp = (raw.compute as Record<string, unknown>) ?? {};
    const shape = String(comp.accelerator_type ?? "GPU_1xA10");
    const accels = Number(comp.num_accelerators ?? 1);
    const perNode = shape.includes("8x") ? 8 : 1;
    const cmd = String(raw.command ?? "");
    out.push({
      name: key,
      kind: "air_yaml",
      ref: path.relative(repoRoot, p),
      team,
      use_case: useCaseFor(key, raw),
      shape,
      nodes: Math.max(1, Math.floor(accels / perNode)),
      needs_torch: cmd.includes("torch"),
    });
  }
  return out;
}
