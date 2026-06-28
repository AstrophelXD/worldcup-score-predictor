from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.features.builder import build_match_feature_mart

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    data_cfg = resolved["data"]
    features_cfg = resolved["features"]

    result = build_match_feature_mart(
        curated_dir=Path(str(data_cfg["curated_dir"])),
        feature_mart_dir=Path(str(data_cfg["feature_mart_dir"])),
        form_windows=list(features_cfg["windows"]["form"]),
    )
    logger.info("Feature mart written: %s (%s rows)", result.output_path, result.row_count)


if __name__ == "__main__":
    main()
