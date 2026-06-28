from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worldcup.inference.decoder import PredictionOutput, decode_score_matrix
from worldcup.models.baseline.dixon_coles import DixonColesModel
from worldcup.models.baseline.likelihood import expected_lambdas
from worldcup.models.registry import BaselineCheckpoint, load_checkpoint


@dataclass
class MatchPrediction:
    match_id: str
    lambda_home: float
    lambda_away: float
    output: PredictionOutput
    lambda_scale: float = 1.0

    def to_dict(self) -> dict:
        payload = self.output.to_dict()
        payload["match_id"] = self.match_id
        payload["expected_goals"] = self.output.expected_goals
        payload["lambda_home"] = self.lambda_home
        payload["lambda_away"] = self.lambda_away
        payload["lambda_scale"] = self.lambda_scale
        return payload


class BaselinePredictor:
    def __init__(self, checkpoint: BaselineCheckpoint) -> None:
        self.checkpoint = checkpoint
        self.model = DixonColesModel(
            home_advantage=checkpoint.home_advantage,
            rho=checkpoint.rho,
            grid_max_goal=checkpoint.grid_max_goal,
        )

    @classmethod
    def from_path(cls, path: Path) -> BaselinePredictor:
        return cls(load_checkpoint(path))

    def predict_row(self, row: pd.Series, lambda_scale: float | None = None) -> MatchPrediction:
        scale = self.checkpoint.lambda_scale if lambda_scale is None else lambda_scale
        lambda_home, lambda_away = expected_lambdas(
            row["home_team_id"],
            row["away_team_id"],
            self.checkpoint.attack,
            self.checkpoint.defense,
            self.checkpoint.home_advantage,
        )
        lambda_home *= scale
        lambda_away *= scale
        dc_output = self.model.predict(lambda_home=lambda_home, lambda_away=lambda_away)
        decoded = decode_score_matrix(dc_output.matrix, overflow_prob=dc_output.overflow_prob)
        return MatchPrediction(
            match_id=str(row["match_id"]),
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            output=decoded,
            lambda_scale=scale,
        )

    def predict_match_id(self, features: pd.DataFrame, match_id: str) -> MatchPrediction:
        rows = features.loc[features["match_id"] == match_id]
        if rows.empty:
            raise KeyError(f"match_id not found in feature mart: {match_id}")
        return self.predict_row(rows.iloc[0])
