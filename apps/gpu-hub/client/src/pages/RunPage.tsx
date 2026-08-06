import { useCallback, useEffect, useState } from 'react';
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@databricks/appkit-ui/react';

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

export function RunPage() {
  const [me, setMe] = useState<Me | null>(null);
  const [workloads, setWorkloads] = useState<Workload[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
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

  const myTeamNames = new Set((me?.teams ?? []).map((t) => t.name));
  const mine = workloads.filter((w) => myTeamNames.has(w.team));
  const myRuns = runs
    .filter((r) => r.requested_by === me?.principal || myTeamNames.has(r.team))
    .reverse();
  const active = myRuns.filter((r) => ACTIVE.includes(r.state));
  const past = myRuns.filter((r) => !ACTIVE.includes(r.state));

  const submit = async () => {
    if (selected == null) return;
    setBusy(true);
    setError('');
    const res = await fetch('/api/broker/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workloadId: selected }),
    });
    if (!res.ok) setError((await res.json()).error ?? res.statusText);
    await refresh();
    setBusy(false);
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
    <div className="space-y-6 max-w-5xl">
      <Card>
        <CardHeader>
          <CardTitle>Run</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-3 items-center flex-wrap">
            <select
              className="border rounded-md px-3 py-2 text-sm bg-background min-w-[28rem]"
              value={selected ?? ''}
              onChange={(e) => setSelected(Number(e.target.value))}
            >
              <option value="" disabled>
                choose a workload…
              </option>
              {mine.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.title || w.name}
                </option>
              ))}
            </select>
            <Button onClick={submit} disabled={busy || selected == null}>
              {busy ? 'Submitting…' : 'Run'}
            </Button>
            {error && <span className="text-sm text-destructive">{error}</span>}
          </div>
          {selected != null &&
            (() => {
              const w = mine.find((x) => x.id === selected);
              if (!w) return null;
              return (
                <div className="rounded-md border bg-muted/40 p-3 text-sm space-y-1">
                  <p>{w.description || 'No description in the workload header yet.'}</p>
                  <p className="text-muted-foreground">
                    {w.use_case} · {w.shape}
                    {w.nodes > 1 ? ` × ${w.nodes} nodes` : ''} · {w.ref}
                  </p>
                </div>
              );
            })()}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Active</CardTitle>
        </CardHeader>
        <CardContent>
          <RunTable runs={active} empty="Nothing queued or running." />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Past runs</CardTitle>
        </CardHeader>
        <CardContent>
          <RunTable runs={past} empty="No finished runs yet." showDetail />
        </CardContent>
      </Card>
    </div>
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
              <td className="py-1 pr-4">{r.name}</td>
              <td className="py-1 pr-4">{r.use_case}</td>
              <td className="py-1 pr-4">
                {r.shape}×{r.nodes}
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
