from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    return BaselineCheckpoint(**data)


def latest_checkpoint(checkpoint_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(checkpoint_dir.glob(f"{model_name}_*.json"))
    return candidates[-1] if candidates else None
