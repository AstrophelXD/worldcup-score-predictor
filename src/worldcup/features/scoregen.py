from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup.features.matchup_graph import (
    DEFAULT_MAX_MATCHUP_EDGES,
    EDGE_FEATURE_DIM,
    build_matchup_graph,
)
from worldcup.features.player_state import build_team_player_tensors
from worldcup.features.point_in_time import filter_before
from worldcup.features.tabular import TabularFeatureSpec, fit_feature_spec

PLAYER_FEATURE_DIM = 6

SEQ_FEATURE_NAMES = [
    "goals_for",
    "goals_against",
    "is_home",
    "is_world_cup",
    "days_since_prev",
]

ODDS_COLUMNS = [
    "odds_home_implied",
    "odds_draw_implied",
    "odds_away_implied",
    "odds_over25_implied",
    "odds_under25_implied",
    "odds_btts_yes_implied",
]


@dataclass
class PlayerContext:
    players: pd.DataFrame
    lineups: pd.DataFrame
    stats: pd.DataFrame


@dataclass
class ScoreGenFeatureSpec:
    tabular: TabularFeatureSpec
    seq_means: np.ndarray
    seq_stds: np.ndarray
    player_means: np.ndarray
    player_stds: np.ndarray
    seq_len: int
    player_slots: int
    odds_dim: int
    player_dim: int
    max_matchup_edges: int
    edge_dim: int

    def to_dict(self) -> dict:
        return {
            "tabular": self.tabular.to_dict(),
            "seq_means": self.seq_means.tolist(),
            "seq_stds": self.seq_stds.tolist(),
            "player_means": self.player_means.tolist(),
            "player_stds": self.player_stds.tolist(),
            "seq_len": self.seq_len,
            "player_slots": self.player_slots,
            "odds_dim": self.odds_dim,
            "player_dim": self.player_dim,
            "max_matchup_edges": self.max_matchup_edges,
            "edge_dim": self.edge_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ScoreGenFeatureSpec:
        player_dim = int(payload.get("player_dim", PLAYER_FEATURE_DIM))
        return cls(
            tabular=TabularFeatureSpec.from_dict(payload["tabular"]),
            seq_means=np.array(payload["seq_means"], dtype=np.float32),
            seq_stds=np.array(payload["seq_stds"], dtype=np.float32),
            player_means=np.array(
                payload.get("player_means", [0.0] * player_dim),
                dtype=np.float32,
            ),
            player_stds=np.array(
                payload.get("player_stds", [1.0] * player_dim),
                dtype=np.float32,
            ),
            seq_len=int(payload["seq_len"]),
            player_slots=int(payload["player_slots"]),
            odds_dim=int(payload["odds_dim"]),
            player_dim=player_dim,
            max_matchup_edges=int(payload.get("max_matchup_edges", DEFAULT_MAX_MATCHUP_EDGES)),
            edge_dim=int(payload.get("edge_dim", EDGE_FEATURE_DIM)),
        )


def load_player_context(curated_dir: Path | None) -> PlayerContext:
    if curated_dir is None:
        return PlayerContext(players=pd.DataFrame(), lineups=pd.DataFrame(), stats=pd.DataFrame())
    players_path = curated_dir / "players.parquet"
    lineups_path = curated_dir / "lineups.parquet"
    stats_path = curated_dir / "player_match_stats.parquet"
    players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()
    lineups = pd.read_parquet(lineups_path) if lineups_path.exists() else pd.DataFrame()
    stats = pd.read_parquet(stats_path) if stats_path.exists() else pd.DataFrame()
    return PlayerContext(players=players, lineups=lineups, stats=stats)


def _normalize_odds(home: float, draw: float, away: float) -> tuple[float, float, float]:
    inv = np.array([1.0 / max(home, 1.01), 1.0 / max(draw, 1.01), 1.0 / max(away, 1.01)])
    total = inv.sum()
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    inv /= total
    return float(inv[0]), float(inv[1]), float(inv[2])


def load_odds_table(odds_path: Path | None) -> pd.DataFrame:
    if odds_path is None or not odds_path.exists():
        return pd.DataFrame(
            columns=[
                "match_id",
                "snapshot_ts",
                "home_odds",
                "draw_odds",
                "away_odds",
                "over25_odds",
                "under25_odds",
                "btts_yes_odds",
            ]
        )
    return pd.read_csv(odds_path)


def odds_features_for_match(
    odds_df: pd.DataFrame,
    match_id: str,
    as_of_time: pd.Timestamp,
) -> tuple[np.ndarray, bool]:
    if odds_df.empty:
        return np.zeros(len(ODDS_COLUMNS), dtype=np.float32), False

    rows = odds_df.loc[odds_df["match_id"] == match_id].copy()
    if rows.empty:
        return np.zeros(len(ODDS_COLUMNS), dtype=np.float32), False

    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"], utc=True)
    eligible = filter_before(rows, as_of_time.to_pydatetime(), "snapshot_ts")
    if eligible.empty:
        return np.zeros(len(ODDS_COLUMNS), dtype=np.float32), False

    row = eligible.sort_values("snapshot_ts").iloc[-1]
    home_p, draw_p, away_p = _normalize_odds(
        float(row["home_odds"]),
        float(row["draw_odds"]),
        float(row["away_odds"]),
    )
    over_odds = float(row.get("over25_odds", 2.0))
    under_odds = float(row.get("under25_odds", 2.0))
    btts_odds = float(row.get("btts_yes_odds", 2.0))
    over_p = 1.0 / max(over_odds, 1.01)
    under_p = 1.0 / max(under_odds, 1.01)
    ou_total = over_p + under_p
    over_p /= ou_total
    under_p /= ou_total
    btts_p = 1.0 / max(btts_odds, 1.01)

    return (
        np.array(
            [home_p, draw_p, away_p, over_p, under_p, btts_p],
            dtype=np.float32,
        ),
        True,
    )


def _team_sequence_steps(
    matches: pd.DataFrame,
    team_id: str,
    as_of_time: pd.Timestamp,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    past = matches.loc[pd.to_datetime(matches["kickoff_ts"], utc=True) < as_of_time].copy()
    home = past.loc[past["home_team_id"] == team_id].copy()
    away = past.loc[past["away_team_id"] == team_id].copy()
    home["goals_for"] = home["home_score_ft"]
    home["goals_against"] = home["away_score_ft"]
    home["is_home"] = 1.0
    away["goals_for"] = away["away_score_ft"]
    away["goals_against"] = away["home_score_ft"]
    away["is_home"] = 0.0

    history = pd.concat(
        [
            home[
                [
                    "kickoff_ts",
                    "goals_for",
                    "goals_against",
                    "is_home",
                    "is_world_cup",
                ]
            ],
            away[
                [
                    "kickoff_ts",
                    "goals_for",
                    "goals_against",
                    "is_home",
                    "is_world_cup",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values("kickoff_ts")

    seq = np.zeros((seq_len, len(SEQ_FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros(seq_len, dtype=bool)
    if history.empty:
        return seq, mask

    recent = history.tail(seq_len)
    prev_kickoff: pd.Timestamp | None = None
    for idx, (_, row) in enumerate(recent.iterrows()):
        kickoff = pd.to_datetime(row["kickoff_ts"], utc=True)
        if prev_kickoff is None:
            days_since = 7.0
        else:
            days_since = max(1.0, (kickoff - prev_kickoff).total_seconds() / 86400.0)
        prev_kickoff = kickoff
        seq[idx] = np.array(
            [
                float(row["goals_for"]) if pd.notna(row["goals_for"]) else 0.0,
                float(row["goals_against"]) if pd.notna(row["goals_against"]) else 0.0,
                float(row["is_home"]),
                float(row["is_world_cup"]),
                float(np.log1p(days_since)),
            ],
            dtype=np.float32,
        )
        mask[idx] = True
    return seq, mask


def normalize_players(players: np.ndarray, spec: ScoreGenFeatureSpec) -> np.ndarray:
    normalized = (players - spec.player_means) / spec.player_stds
    return np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)


def fit_scoregen_spec(
    features: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seq_len: int,
    player_slots: int,
    odds_path: Path | None = None,
    player_context: PlayerContext | None = None,
    max_matchup_edges: int = DEFAULT_MAX_MATCHUP_EDGES,
) -> ScoreGenFeatureSpec:
    tabular_spec = fit_feature_spec(features)
    player_context = player_context or PlayerContext(
        players=pd.DataFrame(), lineups=pd.DataFrame(), stats=pd.DataFrame()
    )

    seq_rows: list[np.ndarray] = []
    player_rows: list[np.ndarray] = []
    for row in features.itertuples(index=False):
        as_of = pd.to_datetime(row.as_of_time, utc=True)
        home_seq, home_mask = _team_sequence_steps(matches, row.home_team_id, as_of, seq_len)
        away_seq, away_mask = _team_sequence_steps(matches, row.away_team_id, as_of, seq_len)
        if home_mask.any():
            seq_rows.append(home_seq[home_mask])
        if away_mask.any():
            seq_rows.append(away_seq[away_mask])

        home_players, home_player_mask, _ = build_team_player_tensors(
            match_id=str(row.match_id),
            team_id=str(row.home_team_id),
            as_of_time=as_of,
            player_slots=player_slots,
            players=player_context.players,
            lineups=player_context.lineups,
            stats=player_context.stats,
        )
        away_players, away_player_mask, _ = build_team_player_tensors(
            match_id=str(row.match_id),
            team_id=str(row.away_team_id),
            as_of_time=as_of,
            player_slots=player_slots,
            players=player_context.players,
            lineups=player_context.lineups,
            stats=player_context.stats,
        )
        if home_player_mask.any():
            player_rows.append(home_players[home_player_mask])
        if away_player_mask.any():
            player_rows.append(away_players[away_player_mask])

    if seq_rows:
        stacked = np.vstack(seq_rows)
        seq_means = np.nanmean(stacked, axis=0)
        seq_stds = np.nanstd(stacked, axis=0)
    else:
        seq_means = np.zeros(len(SEQ_FEATURE_NAMES), dtype=np.float32)
        seq_stds = np.ones(len(SEQ_FEATURE_NAMES), dtype=np.float32)

    if player_rows:
        player_stack = np.vstack(player_rows)
        player_means = np.nanmean(player_stack, axis=0)
        player_stds = np.nanstd(player_stack, axis=0)
    else:
        player_means = np.zeros(PLAYER_FEATURE_DIM, dtype=np.float32)
        player_stds = np.ones(PLAYER_FEATURE_DIM, dtype=np.float32)

    seq_stds = np.where((seq_stds < 1e-6) | np.isnan(seq_stds), 1.0, seq_stds)
    seq_means = np.where(np.isnan(seq_means), 0.0, seq_means)
    player_stds = np.where((player_stds < 1e-6) | np.isnan(player_stds), 1.0, player_stds)
    player_means = np.where(np.isnan(player_means), 0.0, player_means)

    _ = load_odds_table(odds_path)
    return ScoreGenFeatureSpec(
        tabular=tabular_spec,
        seq_means=seq_means,
        seq_stds=seq_stds,
        player_means=player_means,
        player_stds=player_stds,
        seq_len=seq_len,
        player_slots=player_slots,
        odds_dim=len(ODDS_COLUMNS),
        player_dim=PLAYER_FEATURE_DIM,
        max_matchup_edges=max_matchup_edges,
        edge_dim=EDGE_FEATURE_DIM,
    )


def vectorize_tabular_row(row: pd.Series, spec: ScoreGenFeatureSpec) -> np.ndarray:
    from worldcup.features.tabular import vectorize_features

    frame = pd.DataFrame([row])
    return vectorize_features(frame, spec.tabular)[0]


def normalize_sequence(seq: np.ndarray, spec: ScoreGenFeatureSpec) -> np.ndarray:
    return ((seq - spec.seq_means) / spec.seq_stds).astype(np.float32)


def build_scoregen_batch_item(
    row: pd.Series,
    matches: pd.DataFrame,
    spec: ScoreGenFeatureSpec,
    odds_df: pd.DataFrame,
    player_context: PlayerContext | None = None,
) -> dict[str, np.ndarray | bool]:
    player_context = player_context or PlayerContext(
        players=pd.DataFrame(), lineups=pd.DataFrame(), stats=pd.DataFrame()
    )
    as_of = pd.to_datetime(row["as_of_time"], utc=True)
    tabular = vectorize_tabular_row(row, spec)
    home_seq, home_seq_mask = _team_sequence_steps(
        matches, row["home_team_id"], as_of, spec.seq_len
    )
    away_seq, away_seq_mask = _team_sequence_steps(
        matches, row["away_team_id"], as_of, spec.seq_len
    )

    home_players, home_player_mask, home_positions = build_team_player_tensors(
        match_id=str(row["match_id"]),
        team_id=str(row["home_team_id"]),
        as_of_time=as_of,
        player_slots=spec.player_slots,
        players=player_context.players,
        lineups=player_context.lineups,
        stats=player_context.stats,
    )
    away_players, away_player_mask, away_positions = build_team_player_tensors(
        match_id=str(row["match_id"]),
        team_id=str(row["away_team_id"]),
        as_of_time=as_of,
        player_slots=spec.player_slots,
        players=player_context.players,
        lineups=player_context.lineups,
        stats=player_context.stats,
    )
    home_players = normalize_players(home_players, spec)
    away_players = normalize_players(away_players, spec)

    edge_home, edge_away, edge_feats, edge_mask = build_matchup_graph(
        home_players,
        home_player_mask,
        home_positions,
        away_players,
        away_player_mask,
        away_positions,
        max_edges=spec.max_matchup_edges,
    )
    odds, odds_mask = odds_features_for_match(odds_df, str(row["match_id"]), as_of)

    return {
        "tabular": tabular,
        "home_seq": normalize_sequence(home_seq, spec),
        "home_seq_mask": home_seq_mask,
        "away_seq": normalize_sequence(away_seq, spec),
        "away_seq_mask": away_seq_mask,
        "home_players": home_players,
        "home_player_mask": home_player_mask,
        "away_players": away_players,
        "away_player_mask": away_player_mask,
        "edge_home_idx": edge_home,
        "edge_away_idx": edge_away,
        "edge_feats": edge_feats,
        "edge_mask": edge_mask,
        "odds": odds,
        "odds_mask": odds_mask,
    }
