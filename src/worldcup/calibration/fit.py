from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup.backtesting.metrics import score_nll
from worldcup.inference.midlevel_predictor import MidlevelPredictor
from worldcup.inference.predictor import BaselinePredictor
from worldcup.inference.scoregen_predictor import ScoregenPredictor
from worldcup.models.registry import (
    BaselineCheckpoint,
    MidlevelCheckpoint,
    ScoregenCheckpoint,
    save_checkpoint,
    save_midlevel_checkpoint,
    save_scoregen_checkpoint,
)
from worldcup.training.baseline_trainer import temporal_train_val_split


def fit_lambda_scale(
    checkpoint: BaselineCheckpoint,
    features: pd.DataFrame,
    val_ratio: float,
) -> tuple[float, float]:
    """Find a global lambda multiplier that minimizes validation score NLL."""
    split = temporal_train_val_split(features, checkpoint.train_cutoff, val_ratio)
    if split.val.empty:
        return 1.0, float("inf")

    predictor = BaselinePredictor(checkpoint)
    candidates = np.linspace(0.75, 1.35, 25)
    best_scale = 1.0
    best_nll = float("inf")

    for scale in candidates:
        total = 0.0
        count = 0
        for _, row in split.val.iterrows():
            if pd.isna(row["home_score_ft"]) or pd.isna(row["away_score_ft"]):
                continue
            prediction = predictor.predict_row(row, lambda_scale=float(scale))
            total += score_nll(
                int(row["home_score_ft"]),
                int(row["away_score_ft"]),
                prediction.output.matrix,
                prediction.output.overflow_prob,
            )
            count += 1
        if count == 0:
            continue
        avg = total / count
        if avg < best_nll:
            best_nll = avg
            best_scale = float(scale)

    return best_scale, best_nll


def calibrate_checkpoint(
    checkpoint: BaselineCheckpoint,
    features: pd.DataFrame,
    val_ratio: float,
    checkpoint_dir: Path,
) -> BaselineCheckpoint:
    scale, val_nll = fit_lambda_scale(checkpoint, features, val_ratio)
    calibrated = replace(
        checkpoint,
        lambda_scale=scale,
        calibrated_at=datetime.now(UTC).isoformat(),
    )
    save_checkpoint(calibrated, checkpoint_dir)
    return calibrated


def fit_temperature(
    checkpoint: MidlevelCheckpoint,
    predictor: MidlevelPredictor,
    features: pd.DataFrame,
    val_ratio: float,
) -> tuple[float, float]:
    split = temporal_train_val_split(features, checkpoint.train_cutoff, val_ratio)
    if split.val.empty:
        return 1.0, float("inf")

    candidates = np.linspace(0.75, 1.5, 31)
    best_temperature = 1.0
    best_nll = float("inf")

    for temperature in candidates:
        total = 0.0
        count = 0
        for _, row in split.val.iterrows():
            if pd.isna(row["home_score_ft"]) or pd.isna(row["away_score_ft"]):
                continue
            prediction = predictor.predict_row(row, temperature=float(temperature))
            total += score_nll(
                int(row["home_score_ft"]),
                int(row["away_score_ft"]),
                prediction.output.matrix,
                prediction.output.overflow_prob,
            )
            count += 1
        if count == 0:
            continue
        avg = total / count
        if avg < best_nll:
            best_nll = avg
            best_temperature = float(temperature)

    return best_temperature, best_nll


def calibrate_midlevel_checkpoint(
    checkpoint: MidlevelCheckpoint,
    predictor: MidlevelPredictor,
    features: pd.DataFrame,
    val_ratio: float,
    checkpoint_dir: Path,
) -> MidlevelCheckpoint:
    temperature, _ = fit_temperature(checkpoint, predictor, features, val_ratio)
    calibrated = replace(
        checkpoint,
        temperature=temperature,
        calibrated_at=datetime.now(UTC).isoformat(),
    )
    save_midlevel_checkpoint(calibrated, predictor.model.state_dict(), checkpoint_dir)
    return calibrated


def fit_scoregen_temperature(
    checkpoint: ScoregenCheckpoint,
    predictor: ScoregenPredictor,
    features: pd.DataFrame,
    val_ratio: float,
) -> tuple[float, float]:
    split = temporal_train_val_split(features, checkpoint.train_cutoff, val_ratio)
    if split.val.empty:
        return 1.0, float("inf")

    candidates = np.linspace(0.75, 1.5, 31)
    best_temperature = 1.0
    best_nll = float("inf")

    for temperature in candidates:
        total = 0.0
        count = 0
        for _, row in split.val.iterrows():
            if pd.isna(row["home_score_ft"]) or pd.isna(row["away_score_ft"]):
                continue
            prediction = predictor.predict_row(row, temperature=float(temperature))
            total += score_nll(
                int(row["home_score_ft"]),
                int(row["away_score_ft"]),
                prediction.output.matrix,
                prediction.output.overflow_prob,
            )
            count += 1
        if count == 0:
            continue
        avg = total / count
        if avg < best_nll:
            best_nll = avg
            best_temperature = float(temperature)

    return best_temperature, best_nll


def calibrate_scoregen_checkpoint(
    checkpoint: ScoregenCheckpoint,
    predictor: ScoregenPredictor,
    features: pd.DataFrame,
    val_ratio: float,
    checkpoint_dir: Path,
) -> ScoregenCheckpoint:
    temperature, _ = fit_scoregen_temperature(checkpoint, predictor, features, val_ratio)
    calibrated = replace(
        checkpoint,
        temperature=temperature,
        calibrated_at=datetime.now(UTC).isoformat(),
    )
    save_scoregen_checkpoint(calibrated, predictor.model.state_dict(), checkpoint_dir)
    return calibrated
