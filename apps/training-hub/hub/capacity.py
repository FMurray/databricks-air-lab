"""Capacity model: declared config vs live in-flight — so capacity is never tribal knowledge.

Three nested budgets, all declarative (the platform enforces only its own quota, by
fail-fast — receipt: 20 submits, 4 admitted, docs/06):
  1. platform quota per shape  (admission cap measured/It'd-be-configured, e.g. GPU_8xH100: 4)
  2. reservation               (what the customer pays for, e.g. 20 nodes)
  3. team quota                (hub-declared share of the reservation)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShapeCapacity:
    shape: str
    platform_quota_nodes: int   # measured/known admission cap for this shape (0 = unknown)
    reserved_nodes: int         # from the reservation (0 = on-demand only)
    in_flight: int              # brokered + observed runs currently holding nodes

    @property
    def admittable(self) -> int:
        """Slots the platform would admit right now (the binding constraint).
        Shapes with no configured cap are on-demand: nothing to bind platform-side,
        team quotas still apply at dispatch."""
        caps = [c for c in (self.platform_quota_nodes, self.reserved_nodes) if c > 0]
        if not caps:
            return max(0, 999 - self.in_flight)
        return max(0, min(caps) - self.in_flight)


def shape_capacity(cfg, shape: str, in_flight: int) -> ShapeCapacity:
    """cfg: hub.config.HubConfig. Platform quotas live in cfg.platform_quotas (shape->nodes)."""
    reserved = cfg.reservation.total_nodes if shape == cfg.reservation.accelerator_type else 0
    quota = getattr(cfg, "platform_quotas", {}).get(shape, 0)
    return ShapeCapacity(shape=shape, platform_quota_nodes=quota,
                         reserved_nodes=reserved, in_flight=in_flight)


def team_headroom(cfg, team_name: str, team_in_flight: int) -> int:
    team = next((t for t in cfg.teams if t.name == team_name), None)
    if team is None:
        return 0
    return max(0, team.quota_nodes - team_in_flight)
