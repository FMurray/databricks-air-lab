"""Teams/quota/reservation config. Quotas are declarative — see README 'Known gaps'."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = os.environ.get("HUB_TEAMS_CONFIG", "config/teams.yaml")


@dataclass
class Reservation:
    total_nodes: int
    gpus_per_node: int = 8
    accelerator_type: str = "GPU_8xH100"
    region: str = ""


@dataclass
class Team:
    name: str
    quota_nodes: int
    members: list[str] = field(default_factory=list)


@dataclass
class HubConfig:
    reservation: Reservation
    teams: list[Team]

    def team_of(self, principal: str | None) -> str:
        if principal:
            needle = principal.lower()
            for team in self.teams:
                if needle in (m.lower() for m in team.members):
                    return team.name
        return "unmapped"

    @property
    def allocated_nodes(self) -> int:
        return sum(t.quota_nodes for t in self.teams)


def load(path: str | Path = DEFAULT_CONFIG_PATH) -> HubConfig:
    p = Path(path)
    if not p.exists():
        # Fall back to the example so the app renders before real config exists.
        p = Path(__file__).parent.parent / "config" / "teams.example.yaml"
    raw = yaml.safe_load(p.read_text())
    return HubConfig(
        reservation=Reservation(**raw["reservation"]),
        teams=[Team(**t) for t in raw["teams"]],
    )
