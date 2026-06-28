from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.data_ingestion.base import read_parquet
from worldcup.training.baseline_trainer import train_baseline_dixon_coles

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)

    data_cfg = resolved["data"]
    training_cfg = resolved["training"]
    models_cfg = resolved["models"]

    feature_path = Path(str(data_cfg["feature_mart_dir"])) / "match_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"missing feature mart: {feature_path}. Run `python -m scripts.build_features` first."
        )

    features = read_parquet(str(feature_path))
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    model_version = str(training_cfg.get("model_version", models_cfg.get("version", "v1")))
    result = train_baseline_dixon_coles(
        features=features,
        train_cutoff=str(training_cfg["train_cutoff"]),
        val_ratio=float(training_cfg["val_ratio"]),
        home_advantage_init=float(models_cfg["home_advantage"]),
        rho=float(models_cfg["rho"]),
        grid_max_goal=int(models_cfg["grid_max_goal"]),
        model_name=str(models_cfg["name"]),
        model_version=model_version,
        checkpoint_dir=Path(str(training_cfg["checkpoint_dir"])),
    )

    logger.info("Checkpoint saved: %s", result.checkpoint_path)
    logger.info("Train NLL=%.4f Val NLL=%.4f", result.train_nll, result.val_nll)


if __name__ == "__main__":
    main()
