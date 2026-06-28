from __future__ import annotations

import os
from pathlib import Path

from worldcup.inference.midlevel_predictor import MidlevelPredictor
from worldcup.inference.predictor import BaselinePredictor
from worldcup.models.registry import (
    is_midlevel_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    load_midlevel_checkpoint,
)
from worldcup.utils.paths import project_root

Predictor = BaselinePredictor | MidlevelPredictor

DEFAULT_BASELINE_MODEL = "baseline_dixon_coles"
DEFAULT_MIDLEVEL_MODEL = "midlevel_tabular"
DEFAULT_CHECKPOINT_DIR = project_root() / "artifacts" / "checkpoints"


def default_model_name() -> str:
    return os.environ.get("WORLDCUP_MODEL", DEFAULT_BASELINE_MODEL)


def load_predictor(
    model_name: str | None = None,
    checkpoint_dir: Path | None = None,
) -> Predictor:
    name = model_name or default_model_name()
    root = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    checkpoint_path = latest_checkpoint(root, name)
    if checkpoint_path is None:
        raise FileNotFoundError(f"checkpoint not found for model: {name}")

    if is_midlevel_checkpoint(checkpoint_path):
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

    if is_midlevel_checkpoint(checkpoint_path):
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
