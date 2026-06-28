from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.data_ingestion.pipeline import run_ingest
from worldcup.data_ingestion.validate import validate_curated

logger = logging.getLogger(__name__)


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    return path if path.exists() else None


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = cfg.data
    resolved = OmegaConf.to_container(data, resolve=True)
    logger.info("Ingest config:\n%s", OmegaConf.to_yaml(resolved))

    source_systems = OmegaConf.to_container(data.source_systems, resolve=True)
    if not isinstance(source_systems, dict):
        source_systems = {}

    result = run_ingest(
        raw_dir=Path(str(data.raw_dir)),
        curated_dir=Path(str(data.curated_dir)),
        mappings_dir=Path(str(data.mappings_dir)),
        matches_csv=_resolve_path(data.sources.matches),
        elo_csv=_resolve_path(data.sources.elo),
        fifa_csv=_resolve_path(data.sources.fifa_rankings),
        players_csv=_resolve_path(data.sources.get("players")),
        lineups_csv=_resolve_path(data.sources.get("lineups")),
        player_stats_csv=_resolve_path(data.sources.get("player_match_stats")),
        source_systems=source_systems,
    )

    logger.info("Raw outputs: %s", {k: str(v) for k, v in result.raw_paths.items()})
    logger.info("Curated outputs: %s", {k: str(v) for k, v in result.curated_paths.items()})
    logger.info(
        "Counts: teams=%s matches=%s elo=%s fifa=%s players=%s lineups=%s player_stats=%s",
        result.team_count,
        result.match_count,
        result.elo_count,
        result.fifa_count,
        result.player_count,
        result.lineup_count,
        result.player_stat_count,
    )

    report = validate_curated(Path(str(data.curated_dir)))
    for issue in report.issues:
        logger.log(logging.ERROR if issue.level == "error" else logging.WARNING, issue.message)
    logger.info("Validation stats: %s", report.stats)
    if not report.ok:
        raise SystemExit("Curated data validation failed")


if __name__ == "__main__":
    main()
