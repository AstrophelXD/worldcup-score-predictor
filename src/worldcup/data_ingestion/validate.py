from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb


@dataclass
class ValidationIssue:
    level: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, int | float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)


def validate_curated(curated_dir: Path) -> ValidationReport:
    report = ValidationReport()
    con = duckdb.connect(database=":memory:")

    required = {
        "matches": curated_dir / "matches.parquet",
        "teams": curated_dir / "teams.parquet",
        "elo_ratings": curated_dir / "elo_ratings.parquet",
        "fifa_rankings": curated_dir / "fifa_rankings.parquet",
    }

    for name, path in required.items():
        if not path.exists():
            report.issues.append(ValidationIssue("error", f"missing curated table: {name}"))
            continue
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        report.stats[f"{name}_rows"] = int(count)

    if report.stats.get("matches_rows", 0) > 0:
        dup_matches = con.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT match_id) FROM matches"
        ).fetchone()[0]
        if dup_matches:
            report.issues.append(
                ValidationIssue("error", f"duplicate match_id rows: {dup_matches}")
            )

        orphan_home = con.execute(
            """
            SELECT COUNT(*) FROM matches m
            LEFT JOIN teams t ON m.home_team_id = t.team_id
            WHERE t.team_id IS NULL
            """
        ).fetchone()[0]
        if orphan_home:
            report.issues.append(
                ValidationIssue("error", f"matches with unknown home_team_id: {orphan_home}")
            )

        wc_count = con.execute(
            "SELECT COUNT(*) FROM matches WHERE is_world_cup = TRUE"
        ).fetchone()[0]
        report.stats["world_cup_matches"] = int(wc_count)

    if report.stats.get("teams_rows", 0) > 0:
        dup_teams = con.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT team_id) FROM teams"
        ).fetchone()[0]
        if dup_teams:
            report.issues.append(ValidationIssue("error", f"duplicate team_id rows: {dup_teams}"))

    if report.stats.get("elo_ratings_rows", 0) > 0:
        dup_elo = con.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT elo_id) FROM elo_ratings"
        ).fetchone()[0]
        if dup_elo:
            report.issues.append(ValidationIssue("error", f"duplicate elo_id rows: {dup_elo}"))

    if report.stats.get("fifa_rankings_rows", 0) > 0:
        dup_fifa = con.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT fifa_ranking_id) FROM fifa_rankings"
        ).fetchone()[0]
        if dup_fifa:
            report.issues.append(
                ValidationIssue("error", f"duplicate fifa_ranking_id rows: {dup_fifa}")
            )

    optional = {
        "players": curated_dir / "players.parquet",
        "lineups": curated_dir / "lineups.parquet",
        "player_match_stats": curated_dir / "player_match_stats.parquet",
        "injuries": curated_dir / "injuries.parquet",
        "odds": curated_dir / "odds.parquet",
    }
    for name, path in optional.items():
        if not path.exists():
            report.issues.append(
                ValidationIssue("warning", f"optional curated table missing: {name}")
            )
            continue
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        report.stats[f"{name}_rows"] = int(count)

    con.close()
    return report
