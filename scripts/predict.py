from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from worldcup.data_ingestion.base import read_parquet
from worldcup.inference.predictor import BaselinePredictor
from worldcup.models.registry import latest_checkpoint
from worldcup.utils.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single match using baseline model.")
    parser.add_argument("--match-id", required=True, help="Match ID from feature mart")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSON path (defaults to latest baseline_dixon_coles)",
    )
    parser.add_argument(
        "--feature-mart",
        default=str(project_root() / "data" / "feature_mart" / "match_features.parquet"),
        help="Feature mart parquet path",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    feature_path = Path(args.feature_mart)
    features = read_parquet(str(feature_path))
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = latest_checkpoint(
            project_root() / "artifacts" / "checkpoints",
            "baseline_dixon_coles",
        )
        if checkpoint_path is None:
            raise FileNotFoundError("no baseline checkpoint found; run `python -m scripts.train`")

    predictor = BaselinePredictor.from_path(checkpoint_path)
    prediction = predictor.predict_match_id(features, args.match_id)
    print(json.dumps(prediction.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
