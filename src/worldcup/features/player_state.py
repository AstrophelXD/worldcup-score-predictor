from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup.features.point_in_time import filter_before

PLAYER_FEATURE_DIM = 7

POSITION_LINE = {
    "GK": 0.0,
    "CB": 0.25,
    "LB": 0.25,
    "RB": 0.25,
    "CDM": 0.5,
    "CM": 0.55,
    "CAM": 0.65,
    "LW": 0.85,
    "RW": 0.85,
    "ST": 1.0,
}

PRIMARY_MATCHUP_PAIRS = {
    ("ST", "CB"),
    ("ST", "GK"),
    ("LW", "RB"),
    ("RW", "LB"),
    ("CAM", "CDM"),
    ("CM", "CM"),
    ("RW", "LB"),
    ("LW", "RB"),
}

STATUS_RANK = {"official": 0, "historical": 1, "projected": 2}


def position_line(position_code: str | None) -> float:
    if not position_code:
        return 0.5
    return POSITION_LINE.get(str(position_code).upper(), 0.5)


def matchup_weight(home_pos: str | None, away_pos: str | None) -> float:
    if not home_pos or not away_pos:
        return 0.0
    home = str(home_pos).upper()
    away = str(away_pos).upper()
    if (home, away) in PRIMARY_MATCHUP_PAIRS or (away, home) in PRIMARY_MATCHUP_PAIRS:
        return 1.0
    home_line = position_line(home)
    away_line = position_line(away)
    if abs(home_line - away_line) <= 0.2:
        return 0.5
    return 0.0


@dataclass
class PlayerFormWindow:
    minutes_last5: float
    goals_last5: float
    assists_last5: float


def rolling_player_form(
    stats: pd.DataFrame,
    player_id: str,
    as_of_time: pd.Timestamp,
    window: int = 5,
) -> PlayerFormWindow:
    if stats.empty:
        return PlayerFormWindow(0.0, 0.0, 0.0)
    player_stats = stats.loc[stats["player_id"] == player_id].copy()
    if player_stats.empty:
        return PlayerFormWindow(0.0, 0.0, 0.0)
    player_stats["match_date_ts"] = pd.to_datetime(player_stats["match_date"], utc=True)
    eligible = filter_before(player_stats, as_of_time.to_pydatetime(), "match_date_ts")
    recent = eligible.sort_values("match_date_ts").tail(window)
    if recent.empty:
        return PlayerFormWindow(0.0, 0.0, 0.0)
    return PlayerFormWindow(
        minutes_last5=float(recent["minutes_played"].sum()),
        goals_last5=float(recent["goals"].sum()),
        assists_last5=float(recent["assists"].sum()),
    )


def player_availability(
    injuries: pd.DataFrame,
    player_id: str,
    team_id: str,
    as_of_time: pd.Timestamp,
) -> float:
    if injuries.empty:
        return 1.0
    rows = injuries.loc[
        (injuries["player_id"] == player_id) & (injuries["team_id"] == team_id)
    ].copy()
    if rows.empty:
        return 1.0

    as_of_date = as_of_time.date()
    active_rows: list[pd.Series] = []
    for _, row in rows.iterrows():
        start = pd.to_datetime(row["start_date"]).date()
        if pd.isna(start) or start > as_of_date:
            continue
        expected = row.get("expected_return_date")
        status = str(row.get("status", "out")).lower()
        if (
            expected is not None
            and not pd.isna(expected)
            and str(expected).strip()
            and status in {"out", "suspended"}
        ):
            expected_date = pd.to_datetime(expected).date()
            if expected_date <= as_of_date:
                continue
        active_rows.append(row)

    if not active_rows:
        return 1.0

    latest = sorted(active_rows, key=lambda item: item["start_date"])[-1]
    status = str(latest.get("status", "out")).lower()
    if status in {"out", "suspended"}:
        return 0.0
    if status in {"doubtful", "questionable"}:
        confidence = latest.get("confidence")
        if confidence is not None and not pd.isna(confidence):
            return float(max(0.25, min(float(confidence), 0.75)))
        return 0.5
    if status in {"fit", "available"}:
        return 1.0
    return 0.7


def _lineup_row_eligible(row: pd.Series, as_of_time: pd.Timestamp) -> bool:
    status = str(row.get("lineup_status", "historical")).lower()
    if status != "projected":
        return True
    snapshot = row.get("snapshot_ts")
    if snapshot is None or pd.isna(snapshot) or not str(snapshot).strip():
        return True
    snapshot_ts = pd.to_datetime(snapshot, utc=True)
    return snapshot_ts <= as_of_time


def lineup_entries_for_match(
    lineups: pd.DataFrame,
    match_id: str,
    team_id: str,
    as_of_time: pd.Timestamp,
    *,
    prefer_starting: bool = True,
) -> pd.DataFrame:
    if lineups.empty:
        return pd.DataFrame()
    rows = lineups.loc[
        (lineups["match_id"] == match_id) & (lineups["team_id"] == team_id)
    ].copy()
    if rows.empty:
        return rows

    rows = rows.loc[rows.apply(lambda row: _lineup_row_eligible(row, as_of_time), axis=1)]
    if rows.empty:
        return rows

    if prefer_starting:
        rows = rows.loc[rows["is_starting"].astype(bool)]
    if rows.empty:
        return rows

    rows["status_rank"] = rows["lineup_status"].str.lower().map(STATUS_RANK).fillna(9)
    best_rank = rows["status_rank"].min()
    rows = rows.loc[rows["status_rank"] == best_rank]
    prob_col = "projection_prob"
    if prob_col in rows.columns:
        return rows.sort_values(prob_col, ascending=False)
    return rows


def build_player_vector(
    *,
    player_row: pd.Series,
    form: PlayerFormWindow,
    is_starter: bool,
    position_code: str | None,
    availability: float,
    rating_scale: float = 100.0,
) -> np.ndarray:
    rating = float(player_row.get("player_rating") or 70.0) / rating_scale
    rating *= max(0.0, min(availability, 1.0))
    return np.array(
        [
            rating,
            min(form.minutes_last5, 450.0) / 450.0,
            min(form.goals_last5, 5.0) / 5.0,
            min(form.assists_last5, 5.0) / 5.0,
            1.0 if is_starter else 0.0,
            position_line(position_code),
            max(0.0, min(availability, 1.0)),
        ],
        dtype=np.float32,
    )


def build_team_player_tensors(
    *,
    match_id: str,
    team_id: str,
    as_of_time: pd.Timestamp,
    player_slots: int,
    players: pd.DataFrame,
    lineups: pd.DataFrame,
    stats: pd.DataFrame,
    injuries: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    tensor = np.zeros((player_slots, PLAYER_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(player_slots, dtype=bool)
    positions: list[str | None] = [None] * player_slots

    entries = lineup_entries_for_match(lineups, match_id, team_id, as_of_time)
    if entries.empty:
        return tensor, mask, positions

    player_lookup = players.set_index("player_id") if not players.empty else pd.DataFrame()
    slot = 0
    for _, entry in entries.iterrows():
        if slot >= player_slots:
            break
        player_id = str(entry["player_id"])
        if player_lookup.empty or player_id not in player_lookup.index:
            continue
        availability = player_availability(injuries, player_id, team_id, as_of_time)
        if availability <= 0.0:
            continue
        player_row = player_lookup.loc[player_id]
        form = rolling_player_form(stats, player_id, as_of_time)
        pos = entry.get("position_code") or player_row.get("primary_position")
        tensor[slot] = build_player_vector(
            player_row=player_row,
            form=form,
            is_starter=bool(entry.get("is_starting", True)),
            position_code=str(pos) if pos else None,
            availability=availability,
        )
        mask[slot] = True
        positions[slot] = str(pos) if pos else None
        slot += 1
    return tensor, mask, positions
