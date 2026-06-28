from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts._config import CONFIG_DIR
from worldcup.data_ingestion.validate import validate_curated

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    curated_dir = Path(str(cfg.data.curated_dir))
    report = validate_curated(curated_dir)
    for issue in report.issues:
        logger.log(logging.ERROR if issue.level == "error" else logging.WARNING, issue.message)
    logger.info("Validation stats: %s", report.stats)
    if not report.ok:
        raise SystemExit("Curated data validation failed")
    logger.info("Curated data validation passed")


if __name__ == "__main__":
    main()
