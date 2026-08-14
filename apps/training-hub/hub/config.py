"""Teams/quota/reservation config. Quotas are declarative — see README 'Known gaps'."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = os.environ.get("HUB_TEAMS_CONFIG", "config/teams.yaml")


@dataclass
class Workspace:
    profile: str            # ~/.databrickscfg profile — THE app's one workspace
    host: str = ""          # display only


@dataclass
class Reservation:
    total_nodes: int
    gpus_per_node: int = 8
    accelerator_type: str = "GPU_8xH100"
    region: str = ""


@dataclass
class UseCase:
    name: str
    description: str = ""


@dataclass
class Team:
    name: str
    quota_nodes: int
    members: list[str] = field(default_factory=list)
    use_cases: list[UseCase] = field(default_factory=list)


@dataclass
class HubConfig:
    reservation: Reservation
    teams: list[Team]
    platform_quotas: dict = field(default_factory=dict)  # shape -> admitted nodes cap
    catalog_team: str = ""  # team that owns the repo workload catalog
    workspace: Workspace = field(default_factory=lambda: Workspace(profile="DEFAULT"))

    def team_of(self, principal: str | None) -> str:
        teams = self.teams_of(principal)
        return teams[0].name if teams else "unmapped"

    def teams_of(self, principal: str | None) -> list[Team]:
        """All teams the principal belongs to. Empty = read-only user (the access gate)."""
        if not principal:
            return []
        needle = principal.lower()
        return [t for t in self.teams if needle in (m.lower() for m in t.members)]

    @property
    def allocated_nodes(self) -> int:
        return sum(t.quota_nodes for t in self.teams)


def load(path: str | Path = DEFAULT_CONFIG_PATH) -> HubConfig:
    p = Path(path)
    if not p.exists():
        # Fall back to the example so the app renders before real config exists.
        p = Path(__file__).parent.parent / "config" / "teams.example.yaml"
    raw = yaml.safe_load(p.read_text())
    teams = []
    for t in raw["teams"]:
        ucs = [UseCase(**u) if isinstance(u, dict) else UseCase(name=u)
               for u in t.pop("use_cases", [])]
        teams.append(Team(**t, use_cases=ucs))
    return HubConfig(
        reservation=Reservation(**raw["reservation"]),
        teams=teams,
        platform_quotas=raw.get("platform_quotas", {}),
        catalog_team=raw.get("catalog_team", ""),
        workspace=Workspace(**raw.get("workspace", {"profile": "DEFAULT"})),
    )
