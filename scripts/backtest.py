from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.backtesting.runner import run_world_cup_backtest
from worldcup.data_ingestion.base import read_parquet

logger = logging.getLogger(__name__)

SCOPE_TO_SEASON = {
    "world_cup_2018": 2018,
    "world_cup_2022": 2022,
}


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)

    data_cfg = resolved["data"]
    training_cfg = resolved["training"]
    models_cfg = resolved["models"]
    backtest_cfg = resolved["backtest"]

    feature_path = Path(str(data_cfg["feature_mart_dir"])) / "match_features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError("feature mart missing; run build_features first")

    features = read_parquet(str(feature_path))
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    scopes = list(backtest_cfg.get("test_scopes", ["world_cup_2018", "world_cup_2022"]))
    test_set = resolved.get("test_set") or backtest_cfg.get("test_set")
    if test_set:
        scopes = [str(test_set)]

    for scope in scopes:
        if scope not in SCOPE_TO_SEASON:
            raise ValueError(f"unsupported test scope: {scope}")
        season = SCOPE_TO_SEASON[scope]
        logger.info("Running backtest scope=%s season=%s", scope, season)
        result = run_world_cup_backtest(
            features=features,
            season=season,
            val_ratio=float(training_cfg["val_ratio"]),
            home_advantage_init=float(models_cfg["home_advantage"]),
            rho=float(models_cfg["rho"]),
            grid_max_goal=int(models_cfg["grid_max_goal"]),
            model_name=str(models_cfg["name"]),
            model_version=str(training_cfg.get("model_version", models_cfg.get("version", "v1"))),
            checkpoint_dir=Path(str(training_cfg["checkpoint_dir"])),
            report_dir=Path(str(backtest_cfg["report_dir"])),
        )
        logger.info("Report saved: %s", result.report_path)
        logger.info("Metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
