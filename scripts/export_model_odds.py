"""Write model-implied odds for scheduled World Cup 2026 matches."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from worldcup.data_ingestion.base import read_parquet
from worldcup.inference.factory import default_model_name, load_predictor
from worldcup.inference.prediction_odds import prediction_to_odds_row
from worldcup.utils.paths import project_root

logger = logging.getLogger(__name__)


def export_model_odds(
    *,
    model_name: str | None = None,
    match_prefix: str = "wc2026_",
    samples_dir=None,
    feature_mart_path=None,
) -> int:
    root = project_root()
    samples = samples_dir or root / "data" / "samples"
    odds_path = samples / "odds.csv"
    mart_path = feature_mart_path or root / "data" / "feature_mart" / "match_features.parquet"

    if not mart_path.exists():
        raise FileNotFoundError(f"feature mart not found: {mart_path}")

    features = read_parquet(str(mart_path))
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    target_ids = features.loc[
        features["match_id"].astype(str).str.startswith(match_prefix),
        "match_id",
    ].astype(str)
    if target_ids.empty:
        logger.warning("no matches found with prefix %s", match_prefix)
        return 0

    predictor = load_predictor(model_name=model_name)
    active_model = model_name or default_model_name()
    rows: list[dict] = []
    for match_id in target_ids:
        row = features.loc[features["match_id"] == match_id].iloc[0]
        prediction = predictor.predict_match_id(features, match_id)
        rows.append(
            prediction_to_odds_row(
                match_id=match_id,
                kickoff_ts=str(row["kickoff_ts"]),
                prediction=prediction,
            )
        )

    generated = pd.DataFrame(rows)
    if odds_path.exists():
        existing = pd.read_csv(odds_path)
        existing = existing.loc[~existing["match_id"].astype(str).str.startswith(match_prefix)]
        merged = pd.concat([existing, generated], ignore_index=True)
    else:
        merged = generated

    merged = merged.sort_values(["match_id", "snapshot_ts"]).drop_duplicates(
        subset=["match_id"],
        keep="last",
    )
    merged.to_csv(odds_path, index=False)
    logger.info(
        "Wrote %s model-implied odds rows for prefix %s using model %s",
        len(generated),
        match_prefix,
        active_model,
    )
    return len(generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace mock wc2026 odds with model-implied probabilities.",
    )
    parser.add_argument("--model-name", default=None, help="Checkpoint model name")
    parser.add_argument(
        "--match-prefix",
        default="wc2026_",
        help="Only regenerate odds for match_ids with this prefix",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    count = export_model_odds(model_name=args.model_name, match_prefix=args.match_prefix)
    print(f"model odds rows written: {count}")


if __name__ == "__main__":
    main()
