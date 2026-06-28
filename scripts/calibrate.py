from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.calibration.fit import calibrate_checkpoint, fit_lambda_scale
from worldcup.data_ingestion.base import read_parquet
from worldcup.models.registry import latest_checkpoint, load_checkpoint

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
    checkpoint_dir = Path(str(training_cfg["checkpoint_dir"]))
    model_name = str(models_cfg["name"])

    if not feature_path.exists():
        raise FileNotFoundError("feature mart missing; run build_features first")

    checkpoint_path = latest_checkpoint(checkpoint_dir, model_name)
    if checkpoint_path is None:
        raise FileNotFoundError("checkpoint missing; run train first")

    features = read_parquet(str(feature_path))
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    checkpoint = load_checkpoint(checkpoint_path)
    scale, val_nll = fit_lambda_scale(
        checkpoint,
        features,
        val_ratio=float(training_cfg["val_ratio"]),
    )
    logger.info("Best lambda_scale=%.4f validation score NLL=%.4f", scale, val_nll)

    calibrated = calibrate_checkpoint(
        checkpoint,
        features,
        val_ratio=float(training_cfg["val_ratio"]),
        checkpoint_dir=checkpoint_dir,
    )
    logger.info(
        "Calibrated checkpoint saved: lambda_scale=%.4f calibrated_at=%s",
        calibrated.lambda_scale,
        calibrated.calibrated_at,
    )


if __name__ == "__main__":
    main()
