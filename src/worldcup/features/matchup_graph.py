from __future__ import annotations

import numpy as np

from worldcup.features.player_state import matchup_weight, position_line

EDGE_FEATURE_DIM = 6
DEFAULT_MAX_MATCHUP_EDGES = 24


def build_matchup_graph(
    home_players: np.ndarray,
    home_mask: np.ndarray,
    home_positions: list[str | None],
    away_players: np.ndarray,
    away_mask: np.ndarray,
    away_positions: list[str | None],
    *,
    max_edges: int = DEFAULT_MAX_MATCHUP_EDGES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sparse home-vs-away matchup edges with edge features."""
    home_idx: list[int] = []
    away_idx: list[int] = []
    edge_feats: list[np.ndarray] = []

    slots = home_players.shape[0]
    for h in range(slots):
        if not home_mask[h]:
            continue
        for a in range(slots):
            if not away_mask[a]:
                continue
            weight = matchup_weight(home_positions[h], away_positions[a])
            if weight <= 0:
                continue
            home_rating = float(home_players[h, 0])
            away_rating = float(away_players[a, 0])
            edge_feats.append(
                np.array(
                    [
                        home_rating,
                        away_rating,
                        home_rating - away_rating,
                        position_line(home_positions[h]),
                        position_line(away_positions[a]),
                        weight,
                    ],
                    dtype=np.float32,
                )
            )
            home_idx.append(h)
            away_idx.append(a)

    edge_home = np.zeros(max_edges, dtype=np.int64)
    edge_away = np.zeros(max_edges, dtype=np.int64)
    edges = np.zeros((max_edges, EDGE_FEATURE_DIM), dtype=np.float32)
    edge_mask = np.zeros(max_edges, dtype=bool)

    for idx, (h, a, feat) in enumerate(zip(home_idx, away_idx, edge_feats, strict=False)):
        if idx >= max_edges:
            break
        edge_home[idx] = h
        edge_away[idx] = a
        edges[idx] = feat
        edge_mask[idx] = True

    return edge_home, edge_away, edges, edge_mask
