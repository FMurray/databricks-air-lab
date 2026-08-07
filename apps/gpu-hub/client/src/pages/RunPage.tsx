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
interface Project {
  id: number;
  name: string;
  teams: string[];
  metadata: Record<string, string>;
  billable_by_me: boolean;
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
  project: string;
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
  const [projects, setProjects] = useState<Project[]>([]);
  const [optionDefs, setOptionDefs] = useState<OptionDef[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [category, setCategory] = useState('all');
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState<'all' | 'mine' | 'team'>('all');
  const [openWorkload, setOpenWorkload] = useState<Workload | null>(null);

  const refresh = useCallback(async () => {
    const [meR, wR, pR, oR, rR] = await Promise.all([
      fetch('/api/broker/me').then((r) => r.json()),
      fetch('/api/broker/workloads').then((r) => r.json()),
      fetch('/api/broker/projects').then((r) => r.json()),
      fetch('/api/broker/options').then((r) => r.json()),
      fetch('/api/broker/runs').then((r) => r.json()),
    ]);
    setMe(meR);
    setWorkloads(wR);
    setProjects(pR);
    setOptionDefs(oR);
    setRuns(rR.runs ?? []);
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, [refresh]);

  const myTeamNames = useMemo(() => new Set((me?.teams ?? []).map((t) => t.name)), [me]);
  const myProjects = useMemo(() => projects.filter((p) => p.billable_by_me), [projects]);
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

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {shown.map((w) => (
          <Card key={w.id} className="flex flex-col">
            <CardContent className="flex flex-col gap-2 p-4 flex-1">
              <h3 className="font-semibold leading-snug">{w.title || w.workload_key}</h3>
              <p className="text-sm text-muted-foreground line-clamp-3 flex-1">
                {w.description || 'No description yet.'}
              </p>
              <div className="flex flex-wrap gap-1.5">
                <Badge>{CATEGORY_LABELS[w.use_case] ?? w.use_case}</Badge>
                <Badge>{w.configurations.length} configuration{w.configurations.length > 1 ? 's' : ''}</Badge>
                {w.run_count > 0 && <Badge>{w.run_count} runs</Badge>}
              </div>
              <div className="pt-1">
                <Button size="sm" onClick={() => setOpenWorkload(w)}>Run…</Button>
              </div>
            </CardContent>
          </Card>
        ))}
        {shown.length === 0 && (
          <p className="text-sm text-muted-foreground col-span-full">Nothing matches.</p>
        )}
      </div>

      {openWorkload && (
        <RunModal
          workload={openWorkload}
          optionDefs={optionDefs}
          projects={myProjects}
          onClose={() => setOpenWorkload(null)}
          onDone={refresh}
        />
      )}

      <section className="space-y-4">
        <h2 className="text-lg font-semibold">Active</h2>
        <RunTable runs={active} empty="Nothing queued or running." />
        <h2 className="text-lg font-semibold">Past runs</h2>
        <RunTable runs={past} empty="No finished runs yet." showDetail />
      </section>
    </div>
  );
}

function RunModal({ workload: w, optionDefs, projects, onClose, onDone }: {
  workload: Workload;
  optionDefs: OptionDef[];
  projects: Project[];
  onClose: () => void;
  onDone: () => Promise<void>;
}) {
  const [configId, setConfigId] = useState(w.configurations[0]?.id);
  const [project, setProject] = useState(projects[0]?.name ?? '');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [params, setParams] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const cfg = w.configurations.find((c) => c.id === configId) ?? w.configurations[0];
  const proj = projects.find((p) => p.name === project);

  const toggle = (name: string) => {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setSelected(next);
  };

  const submit = async () => {
    if (!cfg) return;
    setBusy(true);
    setError('');
    const res = await fetch('/api/broker/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        configurationId: cfg.id,
        project,
        options: [...selected],
        params: Object.fromEntries(Object.entries(params).filter(([, v]) => v)),
      }),
    });
    if (!res.ok) {
      setError((await res.json()).error ?? res.statusText);
      setBusy(false);
      return;
    }
    await onDone();
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
         onClick={onClose}>
      <div className="w-full max-w-lg max-h-[85vh] overflow-y-auto rounded-lg border bg-background p-5 shadow-lg space-y-4"
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-semibold">{w.title || w.workload_key}</h3>
            <p className="text-sm text-muted-foreground mt-1">{w.description}</p>
          </div>
          <button className="text-muted-foreground hover:text-foreground text-lg leading-none"
                  onClick={onClose} aria-label="Close">×</button>
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">Configuration</label>
          <select
            className="w-full border rounded-md px-2 py-1.5 text-sm bg-background"
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
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">Options</label>
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
                           className="ml-6 mt-1 border rounded px-2 py-0.5 text-xs bg-background w-full max-w-sm"
                           placeholder={`${k} (default: ${v || 'required'})`}
                           value={params[`${o.name}.${k}`] ?? ''}
                           onChange={(e) =>
                             setParams({ ...params, [`${o.name}.${k}`]: e.target.value })} />
                  ))}
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium text-muted-foreground">Project</label>
          <select
            className="w-full border rounded-md px-2 py-1.5 text-sm bg-background"
            value={project}
            onChange={(e) => setProject(e.target.value)}
            title="The project that pays for this run"
          >
            {projects.map((p) => (
              <option key={p.id} value={p.name}>{p.name}</option>
            ))}
          </select>
          {proj && (
            <p className="text-xs text-muted-foreground">
              teams: {proj.teams.join(', ')}
              {proj.metadata.business_app ? ` · business app: ${proj.metadata.business_app}` : ''}
            </p>
          )}
        </div>

        {error && <p className="text-sm text-destructive">{error}</p>}

        <div className="flex justify-end gap-2 pt-1">
          <Button size="sm" variant="outline" onClick={onClose}>Cancel</Button>
          <Button size="sm" disabled={busy || !cfg || !project} onClick={submit}>
            {busy ? 'Submitting…' : 'Run'}
          </Button>
        </div>
      </div>
    </div>
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
            <th className="py-1 pr-4">project</th>
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
              <td className="py-1 pr-4">{r.project}</td>
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
