from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

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


def latest_checkpoint(checkpoint_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(checkpoint_dir.glob(f"{model_name}_*.json"))
    if not candidates:
        return None
    production = [path for path in candidates if "_world_cup_" not in path.name]
    pool = production if production else candidates
    return pool[-1]
