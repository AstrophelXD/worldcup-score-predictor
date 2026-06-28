from __future__ import annotations

import logging

import hydra
import uvicorn
from omegaconf import DictConfig

from scripts._config import CONFIG_DIR

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    host = cfg.api.host
    port = int(cfg.api.port)
    logger.info("Starting API at http://%s:%s", host, port)
    uvicorn.run("worldcup.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
