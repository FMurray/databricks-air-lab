import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@databricks/appkit-ui/react';

interface Capacity {
  shape: string;
  platformQuotaNodes: number;
  reservedNodes: number;
  inFlight: number;
  admittable: number;
}
interface Run {
  id: number;
  team: string;
  project: string;
  name: string;
  shape: string;
  nodes: number;
  requested_by: string;
  state: string;
  run_id: string;
  detail: string;
}

export function OverviewPage() {
  const [capacity, setCapacity] = useState<Capacity[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    const refresh = async () => {
      const [c, r] = await Promise.all([
        fetch('/api/broker/capacity').then((x) => x.json()),
        fetch('/api/broker/runs').then((x) => x.json()),
      ]);
      setCapacity(c);
      setRuns((r.runs ?? []).reverse());
    };
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {capacity.map((c) => {
          const bound = c.platformQuotaNodes || c.reservedNodes || 0;
          return (
            <Card key={c.shape}>
              <CardHeader>
                <CardTitle className="text-base">{c.shape}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-semibold">{c.admittable} free</p>
                <p className="text-sm text-muted-foreground">
                  {c.inFlight} in flight of {bound || 'no configured cap'}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All runs</CardTitle>
        </CardHeader>
        <CardContent>
          {runs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No runs yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-muted-foreground border-b">
                    <th className="py-1 pr-4">#</th>
                    <th className="py-1 pr-4">team</th>
                    <th className="py-1 pr-4">project</th>
                    <th className="py-1 pr-4">workload</th>
                    <th className="py-1 pr-4">shape</th>
                    <th className="py-1 pr-4">requested by</th>
                    <th className="py-1 pr-4">state</th>
                    <th className="py-1 pr-4">platform run / detail</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.id} className="border-b last:border-0">
                      <td className="py-1 pr-4">{r.id}</td>
                      <td className="py-1 pr-4">{r.team}</td>
                      <td className="py-1 pr-4">{r.project}</td>
                      <td className="py-1 pr-4">{r.name}</td>
                      <td className="py-1 pr-4">
                        {r.shape}×{r.nodes}
                      </td>
                      <td className="py-1 pr-4">{r.requested_by}</td>
                      <td className="py-1 pr-4">{r.state}</td>
                      <td className="py-1 pr-4 max-w-md truncate">{r.run_id || r.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
