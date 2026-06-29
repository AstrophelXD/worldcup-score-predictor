"""Export complete World Cup sample CSVs into data/samples/."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.sources.world_cup_catalog import (
    ALL_WORLD_CUP_MATCHES,
    FRIENDLY_MATCHES,
    WorldCupMatch,
)
from worldcup.data_ingestion.sources.world_cup_2026_catalog import world_cup_2026_match_rows
from worldcup.data_ingestion.sources.world_cup_squads import (
    MIN_SQUAD_SIZE,
    TEAM_ALIASES,
    WORLD_CUP_SQUADS,
    squad_for,
)
from worldcup.data_ingestion.sources.world_cup_2026_strength import strength_for
from worldcup.utils.paths import project_root

SCORING_POSITIONS = {"ST", "LW", "RW", "CAM"}

PROJECTED_ONLY_MATCHES = {"wc2022_eng_fra_qf"}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _stage_slug(stage: str) -> str:
    mapping = {
        "Group stage": "gs",
        "Round of 32": "r32",
        "Round of 16": "r16",
        "Quarter-finals": "qf",
        "Semi-finals": "sf",
        "Third place": "tp",
        "Final": "final",
        "Friendly": "friendly",
    }
    return mapping.get(stage, _slug(stage))


def _team_slug(team_name: str) -> str:
    team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
    return team_id.removeprefix("team_")


def match_id_for(record: WorldCupMatch) -> str:
    if record.stage_name == "Friendly":
        return f"intl_{_slug(record.home_team)}_{_slug(record.away_team)}_{record.match_date[:4]}"
    return (
        f"wc{record.year}_{_team_slug(record.home_team)}_{_team_slug(record.away_team)}"
        f"_{_stage_slug(record.stage_name)}"
    )


def _match_row(record: WorldCupMatch) -> dict:
    is_wc = record.stage_name != "Friendly"
    is_ko = record.stage_name in {
        "Round of 16",
        "Quarter-finals",
        "Semi-finals",
        "Third place",
        "Final",
    }
    return {
        "match_id": match_id_for(record),
        "competition_name": "FIFA World Cup" if is_wc else "Friendly",
        "season_name": str(record.year),
        "stage_name": record.stage_name,
        "match_date": record.match_date,
        "kickoff_ts": record.kickoff_ts,
        "home_team_name": record.home_team,
        "away_team_name": record.away_team,
        "home_score_ft": record.home_score_ft,
        "away_score_ft": record.away_score_ft,
        "home_score_ht": record.home_score_ht,
        "away_score_ht": record.away_score_ht,
        "aet_score_home": record.aet_score_home,
        "aet_score_away": record.aet_score_away,
        "pen_score_home": record.pen_score_home,
        "pen_score_away": record.pen_score_away,
        "status": "finished",
        "is_world_cup": is_wc,
        "is_knockout": is_ko,
        "venue": record.venue,
        "city": record.city,
        "country": record.country,
    }


def _squad_name_for(team_name: str) -> str:
    team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
    for alias, tid in sorted(TEAM_ALIASES.items()):
        if tid == team_id and alias in WORLD_CUP_SQUADS:
            return alias
    return team_name


def _unique_teams(matches: pd.DataFrame) -> list[str]:
    """One display name per canonical team_id (prefer longer/stable names)."""
    by_id: dict[str, str] = {}
    for team_name in set(matches["home_team_name"]) | set(matches["away_team_name"]):
        team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
        current = by_id.get(team_id)
        if current is None or len(team_name) > len(current):
            by_id[team_id] = team_name
    return sorted(by_id.values())


def export_players(matches: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    teams = _unique_teams(matches)
    for team_name in teams:
        team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
        for player in squad_for(_squad_name_for(team_name)):
            player_id = f"pla_{player.slug}"
            rows[player_id] = {
                "player_id": player_id,
                "full_name": player.full_name,
                "national_team_id": team_id,
                "primary_position": player.position,
                "market_value_eur": player.market_value_eur,
                "player_rating": player.rating,
            }
    return pd.DataFrame(list(rows.values())).sort_values("player_id")


def export_team_aliases() -> pd.DataFrame:
    return pd.DataFrame(
        [{"alias_name": name, "canonical_team_id": team_id, "notes": ""} for name, team_id in sorted(TEAM_ALIASES.items())]
    )


def export_elo_fifa(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = _unique_teams(matches)
    elo_rows: list[dict] = []
    fifa_rows: list[dict] = []
    ratings_2017 = {name: 1490 + (idx * 13) % 580 for idx, name in enumerate(teams)}
    ratings_2018 = {name: 1500 + (idx * 17) % 600 for idx, name in enumerate(teams)}
    ratings_2022 = {name: 1480 + (idx * 19) % 620 for idx, name in enumerate(teams)}
    snapshots = [
        ("2017-06-01", ratings_2017, "2017-06-07"),
        ("2018-06-01", ratings_2018, "2018-06-07"),
        ("2022-11-01", ratings_2022, "2022-10-06"),
        ("2026-06-01", {name: 1470 + (idx * 23) % 640 for idx, name in enumerate(teams)}, "2026-05-29"),
    ]
    for rating_date, ratings, fifa_date in snapshots:
        for name in teams:
            if rating_date.startswith("2026"):
                s = strength_for(
                    name,
                    default_elo=ratings.get(name, 1500),
                    default_rank=5 + (hash(name + rating_date) % 60),
                )
                elo_val = float(s["elo"])
                rank_val = int(s["fifa_rank"])
                points_val = float(s["fifa_points"])
            else:
                elo_val = round(ratings[name], 1)
                rank_val = 5 + (hash(name + rating_date) % 60)
                points_val = round(1100 + ratings[name] / 3, 2)
            elo_rows.append(
                {
                    "team_name": name,
                    "rating": round(elo_val, 1),
                    "rating_date": rating_date,
                    "rating_system": "elo",
                    "rank": None,
                }
            )
            fifa_rows.append(
                {
                    "team_name": name,
                    "ranking_date": fifa_date,
                    "rank": rank_val,
                    "points": round(points_val, 2),
                }
            )
    return pd.DataFrame(elo_rows), pd.DataFrame(fifa_rows)


def export_lineups(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    player_lookup = players.set_index("player_id")
    rows: list[dict] = []
    for _, match in matches.iterrows():
        if not match["is_world_cup"]:
            continue
        if match["match_id"] in PROJECTED_ONLY_MATCHES:
            continue
        if str(match.get("status", "finished")) == "scheduled":
            continue
        for side in ("home", "away"):
            team_name = match[f"{side}_team_name"]
            team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
            squad = squad_for(_squad_name_for(team_name))[:11]
            for idx, squad_player in enumerate(squad):
                player_id = f"pla_{squad_player.slug}"
                if player_id not in player_lookup.index:
                    continue
                rows.append(
                    {
                        "lineup_id": f"lu_{match['match_id']}_{side}_{idx}",
                        "match_id": match["match_id"],
                        "team_id": team_id,
                        "player_id": player_id,
                        "is_starting": True,
                        "bench_order": "",
                        "position_code": squad_player.position,
                        "formation_slot": f"4-3-3-{squad_player.position}",
                        "lineup_status": "historical",
                        "projection_prob": 1.0,
                        "snapshot_ts": "",
                    }
                )

    projected_match = "wc2022_eng_fra_qf"
    snapshot = "2022-12-10T12:00:00Z"
    for side, team_name in (("home", "England"), ("away", "France")):
        team_id = TEAM_ALIASES[team_name]
        for idx, squad_player in enumerate(squad_for(_squad_name_for(team_name))[:11]):
            player_id = f"pla_{squad_player.slug}"
            if player_id not in player_lookup.index:
                continue
            rows.append(
                {
                    "lineup_id": f"lu_proj_{projected_match}_{side}_{idx}",
                    "match_id": projected_match,
                    "team_id": team_id,
                    "player_id": player_id,
                    "is_starting": True,
                    "bench_order": "",
                    "position_code": squad_player.position,
                    "formation_slot": f"4-3-3-{squad_player.position}",
                    "lineup_status": "projected",
                    "projection_prob": round(0.95 - idx * 0.02, 2),
                    "snapshot_ts": snapshot,
                }
            )

    for _, match in matches.iterrows():
        if not match["is_world_cup"] or str(match.get("status", "finished")) != "scheduled":
            continue
        kickoff = pd.Timestamp(match["kickoff_ts"])
        snapshot = (kickoff - pd.Timedelta(hours=12)).isoformat()
        for side in ("home", "away"):
            team_name = match[f"{side}_team_name"]
            team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
            for idx, squad_player in enumerate(squad_for(_squad_name_for(team_name))[:11]):
                player_id = f"pla_{squad_player.slug}"
                if player_id not in player_lookup.index:
                    continue
                rows.append(
                    {
                        "lineup_id": f"lu_proj_{match['match_id']}_{side}_{idx}",
                        "match_id": match["match_id"],
                        "team_id": team_id,
                        "player_id": player_id,
                        "is_starting": True,
                        "bench_order": "",
                        "position_code": squad_player.position,
                        "formation_slot": f"4-3-3-{squad_player.position}",
                        "lineup_status": "projected",
                        "projection_prob": round(0.92 - idx * 0.02, 2),
                        "snapshot_ts": snapshot,
                    }
                )
    return pd.DataFrame(rows)


def _pick_scorers(team_name: str, goals: int, players: pd.DataFrame) -> list[str]:
    team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
    team_players = players.loc[players["national_team_id"] == team_id].copy()
    if team_players.empty or goals <= 0:
        return []
    scorers = team_players.loc[
        team_players["primary_position"].isin(SCORING_POSITIONS)
    ].sort_values("player_rating", ascending=False)
    if scorers.empty:
        scorers = team_players.sort_values("player_rating", ascending=False)
    ids = scorers["player_id"].tolist()
    if not ids:
        return []
    chosen: list[str] = []
    for i in range(goals):
        chosen.append(ids[min(i, len(ids) - 1)])
    return chosen


def _has_final_score(match: pd.Series) -> bool:
    home = match["home_score_ft"]
    away = match["away_score_ft"]
    if str(match.get("status", "finished")) == "scheduled":
        return False
    if home is None or away is None or pd.isna(home) or pd.isna(away):
        return False
    if str(home).strip() == "" or str(away).strip() == "":
        return False
    return True


def export_player_stats(matches: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    stat_idx = 0
    for _, match in matches.iterrows():
        if not _has_final_score(match):
            continue
        for side, goals in (
            ("home", int(match["home_score_ft"])),
            ("away", int(match["away_score_ft"])),
        ):
            team_name = match[f"{side}_team_name"]
            team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
            for player_id in _pick_scorers(team_name, goals, players):
                stat_idx += 1
                seed = hash(f"{match['match_id']}_{player_id}") % 1000
                rows.append(
                    {
                        "stat_id": f"ps_{stat_idx:05d}",
                        "match_id": match["match_id"],
                        "player_id": player_id,
                        "team_id": team_id,
                        "match_date": match["match_date"],
                        "minutes_played": 90,
                        "goals": 1,
                        "assists": 0,
                        "shots": 2 + (seed % 3),
                        "xg": round(0.35 + (seed % 50) / 100.0, 2),
                        "yellow_cards": 1 if seed % 17 == 0 else 0,
                        "red_cards": 0,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_team_side_events(
    match_id: str,
    side: str,
    goals: int,
    goals_against: int,
) -> dict[str, float | int]:
    seed = abs(hash(f"{match_id}_{side}")) % 10_000
    xg = max(0.25, goals * 0.72 + (seed % 90) / 100.0)
    shots = max(3, goals * 3 + (seed % 6) + 2)
    sot = max(1, min(shots, goals * 2 + (seed % 3) + 1))
    yellow = min(5, (seed % 4) + (1 if goals_against >= 3 else 0))
    red = 1 if seed % 113 == 0 else 0
    possession = 44.0 + (seed % 120) / 10.0
    if side == "away":
        possession = 100.0 - possession
    return {
        "possession": round(possession, 1),
        "shots": int(shots),
        "shots_on_target": int(sot),
        "xg": round(xg, 2),
        "passes_completed": 280 + (seed % 180),
        "corners": 2 + (seed % 7),
        "fouls": 8 + (seed % 10),
        "yellow_cards": int(yellow),
        "red_cards": int(red),
        "cards": int(yellow + red),
    }


def export_team_match_stats(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    idx = 0
    for _, match in matches.iterrows():
        if not _has_final_score(match):
            continue
        if not match.get("is_world_cup"):
            continue
        for side in ("home", "away"):
            team_name = match[f"{side}_team_name"]
            team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
            goals = int(match[f"{side}_score_ft"])
            against = int(match[f"{'away' if side == 'home' else 'home'}_score_ft"])
            events = _synthetic_team_side_events(match["match_id"], side, goals, against)
            idx += 1
            rows.append(
                {
                    "team_match_stat_id": f"tms_{idx:05d}",
                    "match_id": match["match_id"],
                    "team_id": team_id,
                    "match_date": match["match_date"],
                    **events,
                }
            )
    return pd.DataFrame(rows)


def export_odds(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, match in matches.iterrows():
        if not match["is_world_cup"]:
            continue
        if str(match["match_id"]).startswith("wc2026_"):
            continue
        kickoff = pd.Timestamp(match["kickoff_ts"])
        snapshot = (kickoff - pd.Timedelta(hours=3)).isoformat()
        stage = str(match["stage_name"])
        if stage == "Group stage":
            home_odds, draw_odds, away_odds = 2.60, 3.20, 2.80
        elif stage == "Round of 16":
            home_odds, draw_odds, away_odds = 2.40, 3.20, 3.00
        elif stage in {"Quarter-finals", "Semi-finals"}:
            home_odds, draw_odds, away_odds = 2.50, 3.15, 2.90
        else:
            home_odds, draw_odds, away_odds = 2.45, 3.25, 2.85
        rows.append(
            {
                "match_id": match["match_id"],
                "snapshot_ts": snapshot,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "over25_odds": 2.05,
                "under25_odds": 1.78,
                "btts_yes_odds": 1.90,
            }
        )
    return pd.DataFrame(rows)


def export_injuries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "injury_id": "inj_fra_kante_wc22",
                "player_id": "pla_kante",
                "team_id": "team_fra",
                "injury_type": "muscle",
                "status": "out",
                "start_date": "2022-11-01",
                "expected_return_date": "2023-03-01",
                "confidence": 0.95,
                "notes": "Did not travel to Qatar",
            },
            {
                "injury_id": "inj_fra_benzema_wc22",
                "player_id": "pla_benzema",
                "team_id": "team_fra",
                "injury_type": "muscle",
                "status": "out",
                "start_date": "2022-11-19",
                "expected_return_date": "",
                "confidence": 0.98,
                "notes": "Left camp before WC",
            },
            {
                "injury_id": "inj_arg_dybala_wc22",
                "player_id": "pla_dybala",
                "team_id": "team_arg",
                "injury_type": "muscle",
                "status": "doubtful",
                "start_date": "2022-11-10",
                "expected_return_date": "2022-12-01",
                "confidence": 0.6,
                "notes": "Limited minutes risk",
            },
            {
                "injury_id": "inj_bra_neymar_wc22",
                "player_id": "pla_neymar",
                "team_id": "team_bra",
                "injury_type": "knee",
                "status": "out",
                "start_date": "2022-11-20",
                "expected_return_date": "",
                "confidence": 0.97,
                "notes": "Missed knockout stage",
            },
            {
                "injury_id": "inj_eng_kane_wc22",
                "player_id": "pla_kane",
                "team_id": "team_eng",
                "injury_type": "ankle",
                "status": "fit",
                "start_date": "2022-11-01",
                "expected_return_date": "",
                "confidence": 1.0,
                "notes": "Cleared for WC",
            },
            {
                "injury_id": "inj_ger_reus_wc18",
                "player_id": "pla_reus",
                "team_id": "team_ger",
                "injury_type": "ankle",
                "status": "out",
                "start_date": "2018-05-01",
                "expected_return_date": "2018-08-01",
                "confidence": 0.99,
                "notes": "Missed 2018 World Cup",
            },
        ]
    )


def _validate_export(
    matches: pd.DataFrame,
    players: pd.DataFrame,
    lineups: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    teams = _unique_teams(matches)
    for team_name in teams:
        squad = squad_for(_squad_name_for(team_name))
        if len(squad) < MIN_SQUAD_SIZE:
            raise ValueError(f"Squad too small for {team_name}: {len(squad)}")
        team_id = TEAM_ALIASES.get(team_name, f"team_{_slug(team_name)}")
        roster = players.loc[players["national_team_id"] == team_id]
        if len(roster) < MIN_SQUAD_SIZE:
            raise ValueError(
                f"players.csv missing roster for {team_name}: {len(roster)} < {MIN_SQUAD_SIZE}"
            )

    wc_matches = matches.loc[matches["is_world_cup"]]
    for _, match in wc_matches.iterrows():
        if match["match_id"] in PROJECTED_ONLY_MATCHES:
            continue
        expected_status = "projected" if str(match.get("status", "finished")) == "scheduled" else "historical"
        for side in ("home", "away"):
            team_id = TEAM_ALIASES.get(
                match[f"{side}_team_name"],
                f"team_{_slug(match[f'{side}_team_name'])}",
            )
            starters = lineups.loc[
                (lineups["match_id"] == match["match_id"])
                & (lineups["team_id"] == team_id)
                & (lineups["is_starting"] == True)  # noqa: E712
                & (lineups["lineup_status"] == expected_status)
            ]
            if len(starters) != 11:
                raise ValueError(
                    f"Lineup incomplete for {match['match_id']} {side}: "
                    f"{len(starters)}/11 ({expected_status})"
                )

    if stats.empty:
        raise ValueError("player_match_stats.csv is empty")


def export_all(samples_dir: Path | None = None) -> dict[str, int]:
    root = samples_dir or project_root() / "data" / "samples"
    mappings_dir = project_root() / "data" / "external_mappings"
    root.mkdir(parents=True, exist_ok=True)

    all_records = ALL_WORLD_CUP_MATCHES + FRIENDLY_MATCHES
    matches = pd.DataFrame([_match_row(record) for record in all_records])
    wc2026 = pd.DataFrame(world_cup_2026_match_rows())
    matches = pd.concat([matches, wc2026], ignore_index=True)
    matches = matches.drop_duplicates(subset=["match_id"], keep="last").sort_values("kickoff_ts")

    players = export_players(matches)
    elo, fifa = export_elo_fifa(matches)
    lineups = export_lineups(matches, players)
    stats = export_player_stats(matches, players)
    team_stats = export_team_match_stats(matches)
    odds = export_odds(matches)
    injuries = export_injuries()
    aliases = export_team_aliases()

    matches.to_csv(root / "matches.csv", index=False)
    players.to_csv(root / "players.csv", index=False)
    elo.to_csv(root / "elo.csv", index=False)
    fifa.to_csv(root / "fifa_rankings.csv", index=False)
    lineups.to_csv(root / "lineups.csv", index=False)
    stats.to_csv(root / "player_match_stats.csv", index=False)
    team_stats.to_csv(root / "team_match_stats.csv", index=False)
    odds.to_csv(root / "odds.csv", index=False)
    injuries.to_csv(root / "injuries.csv", index=False)
    aliases.to_csv(mappings_dir / "team_aliases.csv", index=False)

    _validate_export(matches, players, lineups, stats)

    return {
        "matches": len(matches),
        "world_cup_matches": int(matches["is_world_cup"].sum()),
        "players": len(players),
        "lineups": len(lineups),
        "player_match_stats": len(stats),
        "team_match_stats": len(team_stats),
        "odds": len(odds),
        "injuries": len(injuries),
        "teams": len(aliases),
    }


def main() -> None:
    counts = export_all()
    print("Exported sample data:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
