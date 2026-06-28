from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def slugify_team_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"team_{slug}"


class TeamResolver:
    """Resolve team names to canonical team_id using alias mappings."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._aliases = {k.strip().lower(): v for k, v in (aliases or {}).items()}
        self._id_to_name: dict[str, str] = {}

    @classmethod
    def from_csv(cls, path: Path) -> TeamResolver:
        df = pd.read_csv(path)
        aliases = dict(zip(df["alias_name"], df["canonical_team_id"], strict=True))
        resolver = cls(aliases=aliases)
        for alias, team_id in aliases.items():
            resolver._id_to_name[team_id] = alias
        return resolver

    def resolve(self, team_name: str) -> str:
        normalized = team_name.strip()
        key = normalized.lower()
        team_id = self._aliases.get(key, slugify_team_name(normalized))
        if team_id not in self._id_to_name:
            self._id_to_name[team_id] = normalized
        return team_id

    def team_name(self, team_id: str) -> str:
        return self._id_to_name.get(team_id, team_id.replace("team_", "").replace("_", " ").title())
