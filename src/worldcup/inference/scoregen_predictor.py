from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from worldcup.features.scoregen import (
    ScoreGenFeatureSpec,
    build_scoregen_batch_item,
    load_odds_table,
)
from worldcup.inference.decoder import decode_score_matrix
from worldcup.inference.predictor import MatchPrediction
from worldcup.models.advanced.scoregen import ScoreGenFootballTransformer
from worldcup.models.registry import ScoregenCheckpoint, load_scoregen_checkpoint


class ScoregenPredictor:
    PLAYER_DIM = 5

    def __init__(
        self,
        checkpoint: ScoregenCheckpoint,
        model: ScoreGenFootballTransformer,
        matches: pd.DataFrame,
        odds_path: Path | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.model = model
        self.matches = matches
        self.spec = ScoreGenFeatureSpec.from_dict(checkpoint.feature_spec)
        self.odds_df = load_odds_table(odds_path)
        self.model.eval()

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        matches_path: Path | None = None,
        odds_path: Path | None = None,
    ) -> ScoregenPredictor:
        checkpoint, state_dict = load_scoregen_checkpoint(path)
        spec = ScoreGenFeatureSpec.from_dict(checkpoint.feature_spec)
        model = ScoreGenFootballTransformer(
            tabular_dim=len(spec.tabular.feature_columns),
            seq_dim=len(spec.seq_means),
            player_dim=cls.PLAYER_DIM,
            odds_dim=spec.odds_dim,
            d_model=checkpoint.d_model,
            n_heads=checkpoint.n_heads,
            n_layers=checkpoint.n_layers,
            max_players=checkpoint.player_slots,
            grid_max_goal=checkpoint.grid_max_goal,
            n_components=checkpoint.n_components,
            dropout=checkpoint.dropout,
        )
        model.load_state_dict(state_dict)
        model.eval()

        if matches_path is None:
            from worldcup.utils.paths import project_root

            matches_path = project_root() / "data" / "curated" / "matches.parquet"
        matches = pd.read_parquet(matches_path)
        return cls(checkpoint, model, matches, odds_path)

    def predict_row(
        self,
        row: pd.Series,
        temperature: float | None = None,
    ) -> MatchPrediction:
        scale = self.checkpoint.temperature if temperature is None else temperature
        item = build_scoregen_batch_item(row, self.matches, self.spec, self.odds_df)
        matrix, overflow = self.model.predict_matrix(
            torch.from_numpy(item["tabular"]).unsqueeze(0),
            torch.from_numpy(item["home_seq"]).unsqueeze(0),
            torch.from_numpy(item["home_seq_mask"]).unsqueeze(0),
            torch.from_numpy(item["away_seq"]).unsqueeze(0),
            torch.from_numpy(item["away_seq_mask"]).unsqueeze(0),
            torch.from_numpy(item["home_players"]).unsqueeze(0),
            torch.from_numpy(item["home_player_mask"]).unsqueeze(0),
            torch.from_numpy(item["away_players"]).unsqueeze(0),
            torch.from_numpy(item["away_player_mask"]).unsqueeze(0),
            torch.from_numpy(item["odds"]).unsqueeze(0),
            torch.tensor([item["odds_mask"]], dtype=torch.bool),
            temperature=scale,
        )
        numpy_matrix = matrix[0].detach().cpu().numpy()
        decoded = decode_score_matrix(numpy_matrix, overflow_prob=overflow)
        return MatchPrediction(
            match_id=str(row["match_id"]),
            lambda_home=decoded.expected_goals["home"],
            lambda_away=decoded.expected_goals["away"],
            output=decoded,
            lambda_scale=scale,
        )

    def predict_match_id(self, features: pd.DataFrame, match_id: str) -> MatchPrediction:
        rows = features.loc[features["match_id"] == match_id]
        if rows.empty:
            raise KeyError(f"match_id not found in feature mart: {match_id}")
        return self.predict_row(rows.iloc[0])
