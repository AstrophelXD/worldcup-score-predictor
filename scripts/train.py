from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.data_ingestion.base import read_parquet
from worldcup.training.baseline_trainer import train_baseline_dixon_coles
from worldcup.training.midlevel_trainer import MidlevelTrainConfig, train_midlevel_tabular
from worldcup.training.scoregen_trainer import (
    ScoregenTrainConfig,
    train_scoregen_football_transformer,
)
from worldcup.utils.paths import project_root

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
    model_type = str(models_cfg.get("type", "baseline"))
    checkpoint_dir = Path(str(training_cfg["checkpoint_dir"]))

    if model_type == "scoregen":
        odds_path_cfg = models_cfg.get("odds_path")
        odds_path = (
            Path(str(odds_path_cfg))
            if odds_path_cfg
            else project_root() / "data" / "samples" / "odds.csv"
        )
        if not odds_path.exists():
            odds_path = None
        matches_path = Path(str(data_cfg["curated_dir"])) / "matches.parquet"
        matches = read_parquet(str(matches_path))
        if not isinstance(matches, pd.DataFrame):
            matches = pd.DataFrame(matches)
        train_cfg = ScoregenTrainConfig(
            d_model=int(models_cfg["d_model"]),
            n_heads=int(models_cfg["n_heads"]),
            n_layers=int(models_cfg["n_layers"]),
            n_components=int(models_cfg["n_components"]),
            seq_len=int(models_cfg["seq_len"]),
            player_slots=int(models_cfg["player_slots"]),
            dropout=float(models_cfg["dropout"]),
            learning_rate=float(models_cfg["learning_rate"]),
            weight_decay=float(models_cfg["weight_decay"]),
            epochs=int(models_cfg["epochs"]),
            patience=int(models_cfg["patience"]),
            batch_size=int(models_cfg["batch_size"]),
            aux_loss_weight=float(models_cfg["aux_loss_weight"]),
            seed=int(training_cfg["seed"]),
            mixed_precision=str(training_cfg.get("mixed_precision")),
            odds_path=odds_path,
        )
        result = train_scoregen_football_transformer(
            features=features,
            matches=matches,
            train_cutoff=str(training_cfg["train_cutoff"]),
            val_ratio=float(training_cfg["val_ratio"]),
            grid_max_goal=int(models_cfg["grid_max_goal"]),
            model_name=str(models_cfg["name"]),
            model_version=model_version,
            checkpoint_dir=checkpoint_dir,
            train_cfg=train_cfg,
        )
    elif model_type == "midlevel":
        train_cfg = MidlevelTrainConfig(
            hidden_dims=[int(x) for x in models_cfg["hidden_dims"]],
            dropout=float(models_cfg["dropout"]),
            learning_rate=float(models_cfg["learning_rate"]),
            weight_decay=float(models_cfg["weight_decay"]),
            epochs=int(models_cfg["epochs"]),
            patience=int(models_cfg["patience"]),
            batch_size=int(models_cfg["batch_size"]),
            aux_loss_weight=float(models_cfg["aux_loss_weight"]),
            seed=int(training_cfg["seed"]),
            mixed_precision=str(training_cfg.get("mixed_precision")),
        )
        result = train_midlevel_tabular(
            features=features,
            train_cutoff=str(training_cfg["train_cutoff"]),
            val_ratio=float(training_cfg["val_ratio"]),
            grid_max_goal=int(models_cfg["grid_max_goal"]),
            model_name=str(models_cfg["name"]),
            model_version=model_version,
            checkpoint_dir=checkpoint_dir,
            train_cfg=train_cfg,
        )
    else:
        result = train_baseline_dixon_coles(
            features=features,
            train_cutoff=str(training_cfg["train_cutoff"]),
            val_ratio=float(training_cfg["val_ratio"]),
            home_advantage_init=float(models_cfg["home_advantage"]),
            rho=float(models_cfg["rho"]),
            grid_max_goal=int(models_cfg["grid_max_goal"]),
            model_name=str(models_cfg["name"]),
            model_version=model_version,
            checkpoint_dir=checkpoint_dir,
        )

    logger.info("Checkpoint saved: %s", result.checkpoint_path)
    logger.info("Train NLL=%.4f Val NLL=%.4f", result.train_nll, result.val_nll)


if __name__ == "__main__":
    main()
