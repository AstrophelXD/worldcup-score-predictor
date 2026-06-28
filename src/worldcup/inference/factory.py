from __future__ import annotations

import os
from pathlib import Path

from worldcup.inference.midlevel_predictor import MidlevelPredictor
from worldcup.inference.predictor import BaselinePredictor
from worldcup.inference.scoregen_predictor import ScoregenPredictor
from worldcup.models.registry import (
    checkpoint_model_type,
    latest_checkpoint,
    load_checkpoint,
    load_midlevel_checkpoint,
    load_scoregen_checkpoint,
)
from worldcup.utils.paths import project_root

Predictor = BaselinePredictor | MidlevelPredictor | ScoregenPredictor

DEFAULT_BASELINE_MODEL = "baseline_dixon_coles"
DEFAULT_MIDLEVEL_MODEL = "midlevel_tabular"
DEFAULT_SCOREGEN_MODEL = "scoregen_football"
DEFAULT_CHECKPOINT_DIR = project_root() / "artifacts" / "checkpoints"


def default_model_name() -> str:
    return os.environ.get("WORLDCUP_MODEL", DEFAULT_BASELINE_MODEL)


def default_odds_path() -> Path | None:
    root = project_root()
    candidates = [
        root / "data" / "samples" / "odds.csv",
        root / "data" / "curated" / "odds.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path if path.suffix == ".csv" else None
    return None


def load_predictor(
    model_name: str | None = None,
    checkpoint_dir: Path | None = None,
) -> Predictor:
    name = model_name or default_model_name()
    root = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    checkpoint_path = latest_checkpoint(root, name)
    if checkpoint_path is None:
        raise FileNotFoundError(f"checkpoint not found for model: {name}")

    model_type = checkpoint_model_type(checkpoint_path)
    if model_type == "scoregen":
        return ScoregenPredictor.from_path(checkpoint_path, odds_path=default_odds_path())
    if model_type == "midlevel":
        return MidlevelPredictor.from_path(checkpoint_path)
    return BaselinePredictor(load_checkpoint(checkpoint_path))


def checkpoint_metadata(
    model_name: str | None = None,
    checkpoint_dir: Path | None = None,
) -> dict:
    name = model_name or default_model_name()
    root = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    checkpoint_path = latest_checkpoint(root, name)
    if checkpoint_path is None:
        return {"model_name": name, "checkpoint_path": None, "model_type": None}

    model_type = checkpoint_model_type(checkpoint_path)
    if model_type == "scoregen":
        checkpoint, _ = load_scoregen_checkpoint(checkpoint_path)
        return {
            "model_name": checkpoint.model_name,
            "model_version": checkpoint.model_version,
            "model_type": "scoregen",
            "checkpoint_path": str(checkpoint_path),
            "train_cutoff": checkpoint.train_cutoff,
            "temperature": checkpoint.temperature,
            "calibrated_at": checkpoint.calibrated_at,
            "train_nll": checkpoint.train_nll,
            "val_nll": checkpoint.val_nll,
            "n_components": checkpoint.n_components,
        }

    if model_type == "midlevel":
        checkpoint, _ = load_midlevel_checkpoint(checkpoint_path)
        return {
            "model_name": checkpoint.model_name,
            "model_version": checkpoint.model_version,
            "model_type": "midlevel",
            "checkpoint_path": str(checkpoint_path),
            "train_cutoff": checkpoint.train_cutoff,
            "temperature": checkpoint.temperature,
            "calibrated_at": checkpoint.calibrated_at,
            "train_nll": checkpoint.train_nll,
            "val_nll": checkpoint.val_nll,
        }

    checkpoint = load_checkpoint(checkpoint_path)
    return {
        "model_name": checkpoint.model_name,
        "model_version": checkpoint.model_version,
        "model_type": "baseline",
        "checkpoint_path": str(checkpoint_path),
        "train_cutoff": checkpoint.train_cutoff,
        "lambda_scale": checkpoint.lambda_scale,
        "calibrated_at": checkpoint.calibrated_at,
    }
