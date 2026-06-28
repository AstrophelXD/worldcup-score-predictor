from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from worldcup.utils.paths import ensure_dir


@dataclass
class BaselineCheckpoint:
    model_name: str
    model_version: str
    home_advantage: float
    rho: float
    grid_max_goal: int
    attack: dict[str, float]
    defense: dict[str, float]
    train_cutoff: str
    trained_at: str
    train_match_count: int
    lambda_scale: float = 1.0
    calibrated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MidlevelCheckpoint:
    model_name: str
    model_version: str
    grid_max_goal: int
    hidden_dims: list[int]
    dropout: float
    feature_spec: dict[str, Any]
    train_cutoff: str
    trained_at: str
    train_match_count: int
    train_nll: float
    val_nll: float
    temperature: float = 1.0
    calibrated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoregenCheckpoint:
    model_type: str
    model_name: str
    model_version: str
    grid_max_goal: int
    n_components: int
    d_model: int
    n_heads: int
    n_layers: int
    seq_len: int
    player_slots: int
    dropout: float
    feature_spec: dict[str, Any]
    train_cutoff: str
    trained_at: str
    train_match_count: int
    train_nll: float
    val_nll: float
    temperature: float = 1.0
    calibrated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_checkpoint(checkpoint: BaselineCheckpoint, checkpoint_dir: Path) -> Path:
    ensure_dir(checkpoint_dir)
    filename = f"{checkpoint.model_name}_{checkpoint.model_version}.json"
    path = checkpoint_dir / filename
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: Path) -> BaselineCheckpoint:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(BaselineCheckpoint)}
    filtered = {key: value for key, value in data.items() if key in allowed}
    filtered.setdefault("lambda_scale", 1.0)
    filtered.setdefault("calibrated_at", None)
    return BaselineCheckpoint(**filtered)


def save_midlevel_checkpoint(
    checkpoint: MidlevelCheckpoint,
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path,
) -> Path:
    ensure_dir(checkpoint_dir)
    filename = f"{checkpoint.model_name}_{checkpoint.model_version}.json"
    path = checkpoint_dir / filename
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
    weights_path = path.with_suffix(".pt")
    torch.save(state_dict, weights_path)
    return path


def load_midlevel_checkpoint(path: Path) -> tuple[MidlevelCheckpoint, dict[str, torch.Tensor]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(MidlevelCheckpoint)}
    filtered = {key: value for key, value in data.items() if key in allowed}
    filtered.setdefault("temperature", 1.0)
    filtered.setdefault("calibrated_at", None)
    checkpoint = MidlevelCheckpoint(**filtered)
    weights_path = path.with_suffix(".pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"midlevel weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return checkpoint, state_dict


def is_midlevel_checkpoint(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_type") == "scoregen":
        return False
    return "feature_spec" in data and "hidden_dims" in data


def is_scoregen_checkpoint(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("model_type") == "scoregen"


def save_scoregen_checkpoint(
    checkpoint: ScoregenCheckpoint,
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: Path,
) -> Path:
    ensure_dir(checkpoint_dir)
    filename = f"{checkpoint.model_name}_{checkpoint.model_version}.json"
    path = checkpoint_dir / filename
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
    weights_path = path.with_suffix(".pt")
    torch.save(state_dict, weights_path)
    return path


def load_scoregen_checkpoint(path: Path) -> tuple[ScoregenCheckpoint, dict[str, torch.Tensor]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(ScoregenCheckpoint)}
    filtered = {key: value for key, value in data.items() if key in allowed}
    filtered.setdefault("temperature", 1.0)
    filtered.setdefault("calibrated_at", None)
    checkpoint = ScoregenCheckpoint(**filtered)
    weights_path = path.with_suffix(".pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"scoregen weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    return checkpoint, state_dict


def checkpoint_model_type(path: Path) -> str:
    if is_scoregen_checkpoint(path):
        return "scoregen"
    if is_midlevel_checkpoint(path):
        return "midlevel"
    return "baseline"


def latest_checkpoint(checkpoint_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(checkpoint_dir.glob(f"{model_name}_*.json"))
    if not candidates:
        return None
    production = [path for path in candidates if "_world_cup_" not in path.name]
    pool = production if production else candidates
    return pool[-1]
