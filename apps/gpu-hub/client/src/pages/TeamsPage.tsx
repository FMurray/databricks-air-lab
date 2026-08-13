import { useEffect, useMemo, useState } from 'react';
import {
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Progress,
} from '@databricks/appkit-ui/react';

interface Project {
  name: string;
  description: string;
  run_count: number;
}
interface Team {
  name: string;
  quota_nodes: number;
  members: string[];
  projects: Project[];
  run_count: number;
  active_nodes: number;
  is_mine: boolean;
}

const localPart = (email: string) => email.split('@')[0] || email;

export function TeamsPage() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [principal, setPrincipal] = useState('');

  useEffect(() => {
    const refresh = async () => {
      const [t, me] = await Promise.all([
        fetch('/api/broker/teams').then((x) => x.json()),
        fetch('/api/broker/me').then((x) => x.json()),
      ]);
      setTeams(t ?? []);
      setPrincipal(me?.principal ?? '');
    };
    refresh();
    const timer = setInterval(refresh, 30_000);
    return () => clearInterval(timer);
  }, []);

  // People can belong to more than one team — map each member to every team they're on so
  // the directory can flag shared membership.
  const membership = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const t of teams) {
      for (const email of t.members) {
        const key = email.toLowerCase();
        m.set(key, [...(m.get(key) ?? []), t.name]);
      }
    }
    return m;
  }, [teams]);

  const me = principal.toLowerCase();

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h2 className="text-lg font-semibold text-foreground">Teams</h2>
        <p className="text-sm text-muted-foreground">
          People belong to one or more teams; a team&apos;s projects are what its runs bill to.
          Utilization is in-flight nodes against the team&apos;s fair-share quota.
        </p>
      </div>

      {teams.length === 0 ? (
        <p className="text-sm text-muted-foreground">No teams configured.</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {teams.map((t) => {
            const pct = t.quota_nodes > 0 ? (t.active_nodes / t.quota_nodes) * 100 : 0;
            const overQuota = t.quota_nodes > 0 && t.active_nodes > t.quota_nodes;
            return (
              <Card key={t.name}>
                <CardHeader>
                  <div className="flex items-center justify-between gap-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      {t.name}
                      {t.is_mine && <Badge variant="default">Your team</Badge>}
                    </CardTitle>
                    <Badge variant="secondary">quota {t.quota_nodes} nodes</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-5">
                  {/* Observability: live utilization vs the team's fair-share quota */}
                  <div>
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Utilization</span>
                      <span className={overQuota ? 'text-destructive font-medium' : 'font-medium'}>
                        {t.active_nodes} / {t.quota_nodes} nodes in flight
                      </span>
                    </div>
                    <Progress value={Math.min(100, pct)} className="h-2" />
                    <p className="text-xs text-muted-foreground mt-1">
                      {t.run_count} run{t.run_count === 1 ? '' : 's'} attributed to this team
                    </p>
                  </div>

                  {/* People belong to teams (maybe more than one) */}
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Members ({t.members.length})
                    </p>
                    {t.members.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No members.</p>
                    ) : (
                      <div className="flex flex-wrap gap-1.5">
                        {t.members.map((email) => {
                          const others = (membership.get(email.toLowerCase()) ?? []).filter(
                            (n) => n !== t.name,
                          );
                          const isYou = email.toLowerCase() === me;
                          return (
                            <Badge
                              key={email}
                              variant={isYou ? 'default' : 'secondary'}
                              title={others.length ? `Also on ${others.join(', ')}` : email}
                            >
                              {localPart(email)}
                              {isYou && ' (you)'}
                              {others.length > 0 && (
                                <span className="ml-1 opacity-70">+{others.length}</span>
                              )}
                            </Badge>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Teams have projects — runs bill to these */}
                  <div>
                    <p className="text-sm text-muted-foreground mb-2">
                      Projects ({t.projects.length})
                    </p>
                    {t.projects.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No projects.</p>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="text-left text-muted-foreground border-b">
                              <th className="py-1 pr-4 font-medium">project</th>
                              <th className="py-1 pr-4 font-medium">description</th>
                              <th className="py-1 font-medium text-right">runs</th>
                            </tr>
                          </thead>
                          <tbody>
                            {t.projects.map((p) => (
                              <tr key={p.name} className="border-b last:border-0 align-top">
                                <td className="py-1 pr-4 font-medium whitespace-nowrap">{p.name}</td>
                                <td className="py-1 pr-4 text-muted-foreground">
                                  {p.description || '—'}
                                </td>
                                <td className="py-1 text-right tabular-nums">{p.run_count}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
