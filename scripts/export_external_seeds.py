"""Export WC 2018/2022 roster seeds into data/external/seeds/ (committed reference data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.export_sample_data import (
    _match_row,
    export_injuries,
    export_lineups,
    export_odds,
    export_player_stats,
    export_players,
    export_team_aliases,
)
from worldcup.data_ingestion.sources.world_cup_catalog import ALL_WORLD_CUP_MATCHES
from worldcup.utils.paths import project_root


def export_external_seeds(seeds_dir: Path | None = None) -> dict[str, int]:
    root = seeds_dir or project_root() / "data" / "external" / "seeds"
    root.mkdir(parents=True, exist_ok=True)

    matches = pd.DataFrame([_match_row(record) for record in ALL_WORLD_CUP_MATCHES])
    matches = matches.drop_duplicates(subset=["match_id"], keep="last").sort_values("kickoff_ts")

    players = export_players(matches)
    lineups = export_lineups(matches, players)
    stats = export_player_stats(matches, players)
    odds = export_odds(matches)
    injuries = export_injuries()
    aliases = export_team_aliases()

    matches.to_csv(root / "wc_matches.csv", index=False)
    players.to_csv(root / "players.csv", index=False)
    lineups.to_csv(root / "lineups.csv", index=False)
    stats.to_csv(root / "player_match_stats.csv", index=False)
    odds.to_csv(root / "odds.csv", index=False)
    injuries.to_csv(root / "injuries.csv", index=False)
    aliases.to_csv(project_root() / "data" / "external_mappings" / "team_aliases.csv", index=False)

    return {
        "wc_matches": len(matches),
        "players": len(players),
        "lineups": len(lineups),
        "player_match_stats": len(stats),
        "odds": len(odds),
        "injuries": len(injuries),
    }


def main() -> None:
    counts = export_external_seeds()
    print("Exported external seeds:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
