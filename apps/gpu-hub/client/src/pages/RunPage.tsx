import { useCallback, useEffect, useMemo, useState } from 'react';
import { Button, Card, CardContent } from '@databricks/appkit-ui/react';

interface Me {
  principal: string;
  teams: { name: string; quota_nodes: number }[];
}
interface Workload {
  id: number;
  team: string;
  use_case: string;
  name: string;
  title: string;
  description: string;
  shape: string;
  nodes: number;
  ref: string;
}
interface Run {
  id: number;
  name: string;
  title?: string;
  use_case: string;
  shape: string;
  nodes: number;
  state: string;
  run_id: string;
  detail: string;
  requested_by: string;
  team: string;
}

const ACTIVE = ['QUEUED', 'SUBMITTED', 'RUNNING'];

// shelf order: real training work first, diagnostics last
const CATEGORY_ORDER = [
  'foundation-models',
  'classic-ml',
  'node-acceptance',
  'multi-language',
  'dependencies',
  'telemetry',
  'scheduling-isolation',
  'env-diagnostics',
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
  const [runs, setRuns] = useState<Run[]>([]);
  const [category, setCategory] = useState<string>('all');
  const [query, setQuery] = useState('');
  const [busyId, setBusyId] = useState<number | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    const [meR, wR, rR] = await Promise.all([
      fetch('/api/broker/me').then((r) => r.json()),
      fetch('/api/broker/workloads').then((r) => r.json()),
      fetch('/api/broker/runs').then((r) => r.json()),
    ]);
    setMe(meR);
    setWorkloads(wR);
    setRuns(rR.runs ?? []);
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, [refresh]);

  const myTeamNames = useMemo(() => new Set((me?.teams ?? []).map((t) => t.name)), [me]);
  const mine = useMemo(
    () => workloads.filter((w) => myTeamNames.has(w.team)),
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
      .filter(
        (w) =>
          !q ||
          w.title.toLowerCase().includes(q) ||
          w.description.toLowerCase().includes(q) ||
          w.name.toLowerCase().includes(q),
      )
      .sort(
        (a, b) =>
          CATEGORY_ORDER.indexOf(a.use_case) - CATEGORY_ORDER.indexOf(b.use_case) ||
          a.title.localeCompare(b.title),
      );
  }, [mine, category, query]);

  const myRuns = runs
    .filter((r) => r.requested_by === me?.principal || myTeamNames.has(r.team))
    .reverse();
  const active = myRuns.filter((r) => ACTIVE.includes(r.state));
  const past = myRuns.filter((r) => !ACTIVE.includes(r.state));

  const runWorkload = async (id: number) => {
    setBusyId(id);
    setError('');
    const res = await fetch('/api/broker/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workloadId: id }),
    });
    if (!res.ok) setError((await res.json()).error ?? res.statusText);
    await refresh();
    setBusyId(null);
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
        <Chip label={`All (${mine.length})`} active={category === 'all'} onClick={() => setCategory('all')} />
        {categories.map((c) => (
          <Chip
            key={c}
            label={`${CATEGORY_LABELS[c] ?? c} (${mine.filter((w) => w.use_case === c).length})`}
            active={category === c}
            onClick={() => setCategory(c)}
          />
        ))}
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {shown.map((w) => (
          <Card key={w.id} className="flex flex-col">
            <CardContent className="flex flex-col gap-2 p-4 flex-1">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-semibold leading-snug">{w.title || w.name}</h3>
              </div>
              <p className="text-sm text-muted-foreground line-clamp-3 flex-1">
                {w.description || 'No description yet — add a comment header to the workload YAML.'}
              </p>
              <div className="flex flex-wrap gap-1.5">
                <Badge>{CATEGORY_LABELS[w.use_case] ?? w.use_case}</Badge>
                <Badge>{w.shape.replace('GPU_', '')}</Badge>
                {w.nodes > 1 && <Badge>{w.nodes} nodes</Badge>}
              </div>
              <div className="flex items-center justify-between pt-1">
                <span className="text-xs text-muted-foreground truncate max-w-[60%]">{w.ref}</span>
                <Button size="sm" onClick={() => runWorkload(w.id)} disabled={busyId !== null}>
                  {busyId === w.id ? 'Submitting…' : 'Run'}
                </Button>
              </div>
            </CardContent>
          </Card>
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
            <th className="py-1 pr-4">workload</th>
            <th className="py-1 pr-4">use case</th>
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
              <td className="py-1 pr-4">{CATEGORY_LABELS[r.use_case] ?? r.use_case}</td>
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
