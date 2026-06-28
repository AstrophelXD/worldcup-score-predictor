"""Report ScoreGen player-tensor coverage for World Cup sample matches."""

from __future__ import annotations

import re

import pandas as pd

from worldcup.data_ingestion.sources.world_cup_squads import TEAM_ALIASES
from worldcup.features.player_state import build_team_player_tensors
from worldcup.utils.paths import project_root


def main() -> None:
    root = project_root()
    matches = pd.read_csv(root / "data/samples/matches.csv")
    lineups = pd.read_csv(root / "data/samples/lineups.csv")
    players = pd.read_csv(root / "data/samples/players.csv")
    stats = pd.read_csv(root / "data/samples/player_match_stats.csv")
    injuries = pd.read_csv(root / "data/samples/injuries.csv")
    features = pd.read_parquet(root / "data/feature_mart/match_features.parquet")

    wc = matches.loc[matches["is_world_cup"]]
    full = ge8 = total = 0
    for _, match in wc.iterrows():
        if match["match_id"] == "wc2022_eng_fra_qf":
            continue
        for side in ("home", "away"):
            team = match[f"{side}_team_name"]
            slug = re.sub(r"[^a-z0-9]+", "_", team.strip().lower()).strip("_")
            team_id = TEAM_ALIASES.get(team, f"team_{slug}")
            as_of = pd.Timestamp(match["kickoff_ts"])
            _, mask, _ = build_team_player_tensors(
                match_id=match["match_id"],
                team_id=team_id,
                as_of_time=as_of,
                player_slots=11,
                players=players,
                lineups=lineups,
                stats=stats,
                injuries=injuries,
            )
            count = int(mask.sum())
            total += 1
            if count == 11:
                full += 1
            if count >= 8:
                ge8 += 1

    print(f"ScoreGen tensor coverage: {full}/{total} full 11/11 ({100 * full / total:.1f}%)")
    print(f"ScoreGen tensor coverage: {ge8}/{total} >= 8 players ({100 * ge8 / total:.1f}%)")
    print(
        "Feature mart missing Elo: "
        f"home={int(features['home_elo'].isna().sum())}, "
        f"away={int(features['away_elo'].isna().sum())}"
    )


if __name__ == "__main__":
    main()
