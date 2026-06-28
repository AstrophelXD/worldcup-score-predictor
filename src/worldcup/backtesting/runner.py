from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from worldcup.backtesting.metrics import MatchMetrics, aggregate_metrics, evaluate_match
from worldcup.inference.predictor import BaselinePredictor
from worldcup.training.baseline_trainer import train_baseline_dixon_coles
from worldcup.utils.paths import ensure_dir


@dataclass
class BacktestResult:
    test_scope: str
    checkpoint_path: Path
    train_cutoff: str
    metrics: dict
    per_match: list[dict]
    report_path: Path


def _world_cup_slice(features: pd.DataFrame, season: int) -> pd.DataFrame:
    kickoffs = pd.to_datetime(features["kickoff_ts"], utc=True)
    mask = features["is_world_cup"].astype(bool) & (kickoffs.dt.year == season)
    subset = features.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"no World Cup {season} matches found in feature mart")
    return subset.sort_values("kickoff_ts")


def run_world_cup_backtest(
    *,
    features: pd.DataFrame,
    season: int,
    val_ratio: float,
    home_advantage_init: float,
    rho: float,
    grid_max_goal: int,
    model_name: str,
    model_version: str,
    checkpoint_dir: Path,
    report_dir: Path,
) -> BacktestResult:
    test_scope = f"world_cup_{season}"
    wc_features = _world_cup_slice(features, season)
    first_kickoff = pd.to_datetime(wc_features["kickoff_ts"], utc=True).min()
    train_cutoff = (first_kickoff - pd.Timedelta(seconds=1)).isoformat()

    train_result = train_baseline_dixon_coles(
        features=features,
        train_cutoff=train_cutoff,
        val_ratio=val_ratio,
        home_advantage_init=home_advantage_init,
        rho=rho,
        grid_max_goal=grid_max_goal,
        model_name=model_name,
        model_version=f"{model_version}_{test_scope}",
        checkpoint_dir=checkpoint_dir,
    )
    predictor = BaselinePredictor.from_path(train_result.checkpoint_path)

    per_match_metrics: list[MatchMetrics] = []
    per_match_rows: list[dict] = []
    confidences: list[float] = []
    top1_hits: list[bool] = []

    for _, row in wc_features.iterrows():
        if pd.isna(row["home_score_ft"]) or pd.isna(row["away_score_ft"]):
            continue
        prediction = predictor.predict_row(row)
        metrics = evaluate_match(
            match_id=str(row["match_id"]),
            home_goals=int(row["home_score_ft"]),
            away_goals=int(row["away_score_ft"]),
            output=prediction.output,
        )
        per_match_metrics.append(metrics)
        per_match_rows.append(
            {
                **asdict(metrics),
                "home_score_ft": int(row["home_score_ft"]),
                "away_score_ft": int(row["away_score_ft"]),
                "as_of_time": str(row["as_of_time"]),
            }
        )
        top1 = prediction.output.top3_scorelines[0]
        top1_hit = (
            top1.home_goals == int(row["home_score_ft"])
            and top1.away_goals == int(row["away_score_ft"])
        )
        confidences.append(prediction.output.uncertainty["confidence"])
        top1_hits.append(top1_hit)

    summary = aggregate_metrics(per_match_metrics, confidences, top1_hits)
    metrics_payload = asdict(summary)
    metrics_payload["train_nll"] = train_result.train_nll
    metrics_payload["val_nll"] = train_result.val_nll

    ensure_dir(report_dir)
    report_path = report_dir / f"backtest_{test_scope}.json"
    payload = {
        "test_scope": test_scope,
        "train_cutoff": train_cutoff,
        "checkpoint_path": str(train_result.checkpoint_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics_payload,
        "per_match": per_match_rows,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return BacktestResult(
        test_scope=test_scope,
        checkpoint_path=train_result.checkpoint_path,
        train_cutoff=train_cutoff,
        metrics=metrics_payload,
        per_match=per_match_rows,
        report_path=report_path,
    )
