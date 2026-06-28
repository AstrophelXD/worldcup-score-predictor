from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from worldcup.features.tabular import TabularFeatureSpec, vectorize_features
from worldcup.inference.decoder import decode_score_matrix
from worldcup.inference.predictor import MatchPrediction
from worldcup.models.midlevel.tabular_score import TabularScoreModel
from worldcup.models.registry import MidlevelCheckpoint, load_midlevel_checkpoint


class MidlevelPredictor:
    def __init__(self, checkpoint: MidlevelCheckpoint, model: TabularScoreModel) -> None:
        self.checkpoint = checkpoint
        self.model = model
        self.feature_spec = TabularFeatureSpec.from_dict(checkpoint.feature_spec)
        self.model.eval()

    @classmethod
    def from_path(cls, path: Path) -> MidlevelPredictor:
        checkpoint, state_dict = load_midlevel_checkpoint(path)
        model = TabularScoreModel(
            input_dim=len(checkpoint.feature_spec["feature_columns"]),
            hidden_dims=checkpoint.hidden_dims,
            grid_max_goal=checkpoint.grid_max_goal,
            dropout=checkpoint.dropout,
        )
        model.load_state_dict(state_dict)
        model.eval()
        return cls(checkpoint, model)

    def predict_row(
        self,
        row: pd.Series,
        temperature: float | None = None,
    ) -> MatchPrediction:
        scale = self.checkpoint.temperature if temperature is None else temperature
        frame = pd.DataFrame([row])
        features = vectorize_features(frame, self.feature_spec)
        tensor = torch.from_numpy(features)
        matrix, overflow_prob = self.model.predict_matrix(tensor, temperature=scale)
        decoded = decode_score_matrix(matrix, overflow_prob=overflow_prob)
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
