from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = [
    "is_world_cup",
    "is_knockout",
    "must_win_flag",
    "draw_acceptable_flag",
]

STRENGTH_COLUMNS = [
    "home_elo",
    "away_elo",
    "home_elo_rank",
    "away_elo_rank",
    "home_fifa_rank",
    "away_fifa_rank",
    "home_fifa_points",
    "away_fifa_points",
]

REST_COLUMNS = ["home_rest_days", "away_rest_days"]

FORM_COLUMNS = [
    "home_goals_for_last3",
    "home_goals_against_last3",
    "home_goals_for_last5",
    "home_goals_against_last5",
    "home_goals_for_last10",
    "home_goals_against_last10",
    "away_goals_for_last3",
    "away_goals_against_last3",
    "away_goals_for_last5",
    "away_goals_against_last5",
    "away_goals_for_last10",
    "away_goals_against_last10",
]

DERIVED_COLUMNS = [
    "elo_diff",
    "fifa_rank_diff",
    "fifa_points_diff",
    "form_goals_for_diff_last5",
    "form_goals_against_diff_last5",
]


@dataclass
class TabularFeatureSpec:
    feature_columns: list[str]
    means: np.ndarray
    stds: np.ndarray

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> TabularFeatureSpec:
        return cls(
            feature_columns=list(payload["feature_columns"]),
            means=np.array(payload["means"], dtype=np.float32),
            stds=np.array(payload["stds"], dtype=np.float32),
        )


def enrich_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["elo_diff"] = enriched["home_elo"] - enriched["away_elo"]
    enriched["fifa_rank_diff"] = enriched["away_fifa_rank"] - enriched["home_fifa_rank"]
    enriched["fifa_points_diff"] = enriched["home_fifa_points"] - enriched["away_fifa_points"]
    enriched["form_goals_for_diff_last5"] = (
        enriched["home_goals_for_last5"] - enriched["away_goals_for_last5"]
    )
    enriched["form_goals_against_diff_last5"] = (
        enriched["home_goals_against_last5"] - enriched["away_goals_against_last5"]
    )
    return enriched


def feature_columns() -> list[str]:
    return CONTEXT_COLUMNS + STRENGTH_COLUMNS + REST_COLUMNS + FORM_COLUMNS + DERIVED_COLUMNS


def fit_feature_spec(df: pd.DataFrame) -> TabularFeatureSpec:
    enriched = enrich_tabular_features(df)
    columns = feature_columns()
    matrix = enriched[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds = np.where((stds < 1e-6) | np.isnan(stds), 1.0, stds)
    means = np.where(np.isnan(means), 0.0, means)
    return TabularFeatureSpec(feature_columns=columns, means=means, stds=stds)


def vectorize_features(df: pd.DataFrame, spec: TabularFeatureSpec) -> np.ndarray:
    enriched = enrich_tabular_features(df)
    matrix = enriched[spec.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    matrix = np.where(np.isnan(matrix), spec.means, matrix)
    return ((matrix - spec.means) / spec.stds).astype(np.float32)


def extract_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    home = df["home_score_ft"].astype(int).to_numpy()
    away = df["away_score_ft"].astype(int).to_numpy()
    return home, away
