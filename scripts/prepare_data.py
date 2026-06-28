from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.data_ingestion.sources.prepare import prepare_external_sources

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    data_cfg = resolved["data"]

    result = prepare_external_sources(
        staging_dir=Path(str(data_cfg["staging_dir"])),
        match_sources=list(data_cfg.get("external_inputs", {}).get("matches", [])),
        elo_sources=list(data_cfg.get("external_inputs", {}).get("elo", [])),
        fifa_sources=list(data_cfg.get("external_inputs", {}).get("fifa_rankings", [])),
        include_samples=bool(data_cfg.get("include_samples", False)),
        samples_dir=Path(str(data_cfg["samples_dir"])),
    )

    logger.info("Prepared canonical CSVs:")
    logger.info("  matches=%s", result.matches)
    logger.info("  elo=%s", result.elo)
    logger.info("  fifa_rankings=%s", result.fifa_rankings)
    logger.info("Next: python -m scripts.ingest --config-name=config data=external")


if __name__ == "__main__":
    main()
