import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, Card, CardContent } from '@databricks/appkit-ui/react';

interface Me {
  principal: string;
  teams: { name: string; quota_nodes: number; use_cases: { name: string }[] }[];
}
interface Configuration {
  id: number;
  name: string;
  title: string;
  shape: string;
  nodes: number;
  ref: string;
  author: string;
  experiment_name: string;
}
interface Workload {
  id: number;
  workload_key: string;
  team: string;
  use_case: string;
  title: string;
  description: string;
  run_count: number;
  my_run_count: number;
  team_run_count: number;
  configurations: Configuration[];
}
interface OptionDef {
  name: string;
  description: string;
  params: Record<string, string>;
}
interface Run {
  id: number;
  name: string;
  title?: string;
  use_case: string;
  options: string;
  shape: string;
  nodes: number;
  state: string;
  run_id: string;
  detail: string;
  requested_by: string;
  team: string;
}

const ACTIVE = ['QUEUED', 'SUBMITTED', 'RUNNING'];

const CATEGORY_ORDER = [
  'foundation-models', 'classic-ml', 'node-acceptance', 'multi-language',
  'dependencies', 'telemetry', 'scheduling-isolation', 'env-diagnostics',
];
const CATEGORY_LABELS: Record<string, string> = {
  'foundation-models': 'Foundation models',
  'classic-ml': 'Classic ML',
  'node-acceptance': 'Node acceptance',
  'multi-language': 'Multi-language',
  dependencies: 'Dependencies',
  telemetry: 'Telemetry',
  'scheduling-isolation': 'Scheduling & isolation',
  'env-diagnostics': 'Diagnostics',
};

export function RunPage() {
  const [me, setMe] = useState<Me | null>(null);
  const [workloads, setWorkloads] = useState<Workload[]>([]);
  const [optionDefs, setOptionDefs] = useState<OptionDef[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [category, setCategory] = useState('all');
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState<'all' | 'mine' | 'team'>('all');
  const [busyKey, setBusyKey] = useState<number | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    const [meR, wR, oR, rR] = await Promise.all([
      fetch('/api/broker/me').then((r) => r.json()),
      fetch('/api/broker/workloads').then((r) => r.json()),
      fetch('/api/broker/options').then((r) => r.json()),
      fetch('/api/broker/runs').then((r) => r.json()),
    ]);
    setMe(meR);
    setWorkloads(wR);
    setOptionDefs(oR);
    setRuns(rR.runs ?? []);
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, [refresh]);

  const myTeamNames = useMemo(() => new Set((me?.teams ?? []).map((t) => t.name)), [me]);
  const myUseCases = useMemo(
    () => (me?.teams ?? []).flatMap((t) => t.use_cases.map((u) => u.name)),
    [me],
  );
  const mine = useMemo(
    () => workloads.filter((w) => myTeamNames.has(w.team) && w.configurations.length > 0),
    [workloads, myTeamNames],
  );

  const categories = useMemo(() => {
    const present = new Set(mine.map((w) => w.use_case));
    return CATEGORY_ORDER.filter((c) => present.has(c));
  }, [mine]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return mine
      .filter((w) => category === 'all' || w.use_case === category)
      .filter((w) =>
        history === 'all' ? true : history === 'mine' ? w.my_run_count > 0 : w.team_run_count > 0,
      )
      .filter(
        (w) =>
          !q ||
          w.title.toLowerCase().includes(q) ||
          w.description.toLowerCase().includes(q) ||
          w.workload_key.toLowerCase().includes(q),
      )
      .sort(
        (a, b) =>
          CATEGORY_ORDER.indexOf(a.use_case) - CATEGORY_ORDER.indexOf(b.use_case) ||
          a.title.localeCompare(b.title),
      );
  }, [mine, category, query, history]);

  const myRuns = runs
    .filter((r) => r.requested_by === me?.principal || myTeamNames.has(r.team))
    .reverse();
  const active = myRuns.filter((r) => ACTIVE.includes(r.state));
  const past = myRuns.filter((r) => !ACTIVE.includes(r.state));

  const request = async (workloadId: number, configurationId: number, useCase: string,
                         options: string[], params: Record<string, string>) => {
    setBusyKey(workloadId);
    setError('');
    const res = await fetch('/api/broker/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ configurationId, useCase, options, params }),
    });
    if (!res.ok) setError((await res.json()).error ?? res.statusText);
    await refresh();
    setBusyKey(null);
  };

  if (me && me.teams.length === 0) {
    return (
      <p className="text-muted-foreground">
        You are not in any team, so you can look but not run. Ask your platform admin to add
        you to a team. Signed in as {me.principal || '(unknown)'}.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="search"
          placeholder="Search workloads…"
          className="border rounded-md px-3 py-1.5 text-sm bg-background w-64"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Chip label="Run by me" active={history === 'mine'}
              onClick={() => setHistory(history === 'mine' ? 'all' : 'mine')} />
        <Chip label="Run by my team" active={history === 'team'}
              onClick={() => setHistory(history === 'team' ? 'all' : 'team')} />
        <span className="mx-1 h-5 w-px bg-border" />
        <Chip label={`All (${mine.length})`} active={category === 'all'}
              onClick={() => setCategory('all')} />
        {categories.map((c) => (
          <Chip key={c}
                label={`${CATEGORY_LABELS[c] ?? c} (${mine.filter((w) => w.use_case === c).length})`}
                active={category === c} onClick={() => setCategory(c)} />
        ))}
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {shown.map((w) => (
          <WorkloadCard key={w.id} workload={w} optionDefs={optionDefs}
                        useCases={myUseCases} busy={busyKey === w.id} onRun={request} />
        ))}
        {shown.length === 0 && (
          <p className="text-sm text-muted-foreground col-span-full">Nothing matches.</p>
        )}
      </div>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Active</h2>
        <RunTable runs={active} empty="Nothing queued or running." />
        <h2 className="text-lg font-semibold">Past runs</h2>
        <RunTable runs={past} empty="No finished runs yet." showDetail />
      </section>
    </div>
  );
}

function WorkloadCard({ workload: w, optionDefs, useCases, busy, onRun }: {
  workload: Workload;
  optionDefs: OptionDef[];
  useCases: string[];
  busy: boolean;
  onRun: (workloadId: number, configurationId: number, useCase: string,
          options: string[], params: Record<string, string>) => void;
}) {
  const [configId, setConfigId] = useState(w.configurations[0]?.id);
  const [useCase, setUseCase] = useState(
    useCases.includes(w.use_case) ? w.use_case : (useCases[0] ?? ''));
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [params, setParams] = useState<Record<string, string>>({});
  const [showOptions, setShowOptions] = useState(false);

  const cfg = w.configurations.find((c) => c.id === configId) ?? w.configurations[0];

  const toggle = (name: string) => {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelected(next);
  };

  return (
    <Card className="flex flex-col">
      <CardContent className="flex flex-col gap-2 p-4 flex-1">
        <h3 className="font-semibold leading-snug">{w.title || w.workload_key}</h3>
        <p className="text-sm text-muted-foreground line-clamp-3">
          {w.description || 'No description yet.'}
        </p>
        <div className="flex flex-wrap gap-1.5">
          <Badge>{CATEGORY_LABELS[w.use_case] ?? w.use_case}</Badge>
          {w.run_count > 0 && <Badge>{w.run_count} runs</Badge>}
        </div>

        <label className="text-xs text-muted-foreground mt-1">Configuration</label>
        <select
          className="border rounded-md px-2 py-1.5 text-sm bg-background"
          value={configId}
          onChange={(e) => setConfigId(Number(e.target.value))}
        >
          {w.configurations.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name.split('/').pop()} — {c.shape.replace('GPU_', '')}
              {c.nodes > 1 ? ` × ${c.nodes} nodes` : ''}
            </option>
          ))}
        </select>

        <button
          className="text-xs text-left text-muted-foreground underline underline-offset-2"
          onClick={() => setShowOptions(!showOptions)}
        >
          {showOptions ? 'Hide options' : `Options (${selected.size} selected)`}
        </button>
        {showOptions && (
          <div className="space-y-2 rounded-md border p-2">
            {optionDefs.map((o) => (
              <div key={o.name}>
                <label className="flex items-start gap-2 text-sm">
                  <input type="checkbox" className="mt-0.5" checked={selected.has(o.name)}
                         onChange={() => toggle(o.name)} />
                  <span>
                    <span className="font-medium">{o.name}</span>
                    <span className="text-muted-foreground"> — {o.description}</span>
                  </span>
                </label>
                {selected.has(o.name) &&
                  Object.entries(o.params ?? {}).map(([k, v]) => (
                    <input key={k}
                           className="ml-6 mt-1 border rounded px-2 py-0.5 text-xs bg-background w-72"
                           placeholder={`${k} (default: ${v || 'required'})`}
                           value={params[`${o.name}.${k}`] ?? ''}
                           onChange={(e) =>
                             setParams({ ...params, [`${o.name}.${k}`]: e.target.value })} />
                  ))}
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between gap-2 pt-1">
          <select
            className="border rounded-md px-2 py-1 text-xs bg-background"
            value={useCase}
            onChange={(e) => setUseCase(e.target.value)}
            title="The use case that pays for this run"
          >
            {useCases.map((u) => (
              <option key={u} value={u}>
                bill to: {u}
              </option>
            ))}
          </select>
          <Button size="sm" disabled={busy || !cfg}
                  onClick={() => cfg && onRun(w.id, cfg.id, useCase, [...selected],
                    Object.fromEntries(Object.entries(params).filter(([, v]) => v)))}>
            {busy ? 'Submitting…' : 'Run'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function Chip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 rounded-full text-sm border transition-colors ${
        active
          ? 'bg-primary text-primary-foreground border-primary'
          : 'bg-background text-muted-foreground hover:bg-muted'
      }`}
    >
      {label}
    </button>
  );
}

function Badge({ children }: { children: React.ReactNode }) {
  return (
    <span className="px-2 py-0.5 rounded-full bg-muted text-xs text-muted-foreground">
      {children}
    </span>
  );
}

function RunTable({ runs, empty, showDetail }: { runs: Run[]; empty: string; showDetail?: boolean }) {
  if (!runs.length) return <p className="text-sm text-muted-foreground">{empty}</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-muted-foreground border-b">
            <th className="py-1 pr-4">#</th>
            <th className="py-1 pr-4">configuration</th>
            <th className="py-1 pr-4">billed to</th>
            <th className="py-1 pr-4">options</th>
            <th className="py-1 pr-4">shape</th>
            <th className="py-1 pr-4">state</th>
            <th className="py-1 pr-4">requested by</th>
            <th className="py-1 pr-4">{showDetail ? 'detail' : 'platform run'}</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="border-b last:border-0">
              <td className="py-1 pr-4">{r.id}</td>
              <td className="py-1 pr-4">{r.title || r.name}</td>
              <td className="py-1 pr-4">{r.use_case}</td>
              <td className="py-1 pr-4">
                {r.options && r.options !== '[]' ? JSON.parse(r.options).join(', ') : '—'}
              </td>
              <td className="py-1 pr-4">
                {r.shape.replace('GPU_', '')}
                {r.nodes > 1 ? `×${r.nodes}` : ''}
              </td>
              <td className="py-1 pr-4">{r.state}</td>
              <td className="py-1 pr-4">{r.requested_by}</td>
              <td className="py-1 pr-4 max-w-md truncate">
                {showDetail ? r.detail : r.run_id}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
