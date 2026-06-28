from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scripts._config import CONFIG_DIR
from worldcup.data_ingestion.base import read_parquet
from worldcup.data_ingestion.validate import validate_curated
from worldcup.utils.paths import ensure_dir, project_root

logger = logging.getLogger(__name__)


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    data_cfg = resolved["data"]

    curated_dir = Path(str(data_cfg["curated_dir"]))
    feature_path = Path(str(data_cfg["feature_mart_dir"])) / "match_features.parquet"
    report_dir = project_root() / "artifacts" / "reports"
    ensure_dir(report_dir)

    validation = validate_curated(curated_dir)
    lines = [
        "# Data QA Report",
        "",
        f"Generated at: {datetime.now(UTC).isoformat()}",
        "",
        "## Curated validation",
        "",
        f"- OK: `{validation.ok}`",
        f"- Stats: `{validation.stats}`",
        "",
    ]
    for issue in validation.issues:
        lines.append(f"- **{issue.level}**: {issue.message}")

    if feature_path.exists():
        features = read_parquet(str(feature_path))
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        wc = int(features["is_world_cup"].astype(bool).sum())
        lines.extend(
            [
                "",
                "## Feature mart",
                "",
                f"- Rows: {len(features)}",
                f"- World Cup rows: {wc} ({_pct(wc, len(features))})",
                f"- Columns: {len(features.columns)}",
                f"- Missing home_elo: {int(features['home_elo'].isna().sum())}",
                f"- Missing away_elo: {int(features['away_elo'].isna().sum())}",
            ]
        )
    else:
        lines.extend(["", "## Feature mart", "", "- Not built"])

    report_path = report_dir / "data_qa.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("QA report written: %s", report_path)
    if not validation.ok:
        raise SystemExit("Curated data validation failed")


if __name__ == "__main__":
    main()
