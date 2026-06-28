import shutil

import pandas as pd
import pytest

from worldcup.data_ingestion.pipeline import run_ingest
from worldcup.features.builder import build_match_feature_mart
from worldcup.features.point_in_time import filter_before
from worldcup.inference.predictor import BaselinePredictor
from worldcup.models.registry import load_checkpoint
from worldcup.training.baseline_trainer import train_baseline_dixon_coles
from worldcup.utils.paths import project_root


@pytest.fixture
def pipeline_paths():
    root = project_root()
    tmp_root = root / "tests" / "tmp" / "phase2"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    paths = {
        "raw_dir": tmp_root / "raw",
        "curated_dir": tmp_root / "curated",
        "feature_mart_dir": tmp_root / "feature_mart",
        "checkpoint_dir": tmp_root / "checkpoints",
        "mappings_dir": root / "data" / "external_mappings",
        "matches_csv": root / "data" / "samples" / "matches.csv",
        "elo_csv": root / "data" / "samples" / "elo.csv",
        "fifa_csv": root / "data" / "samples" / "fifa_rankings.csv",
    }
    yield paths
    shutil.rmtree(tmp_root, ignore_errors=True)


def test_feature_builder_no_self_leakage(pipeline_paths):
    run_ingest(
        raw_dir=pipeline_paths["raw_dir"],
        curated_dir=pipeline_paths["curated_dir"],
        mappings_dir=pipeline_paths["mappings_dir"],
        matches_csv=pipeline_paths["matches_csv"],
        elo_csv=pipeline_paths["elo_csv"],
        fifa_csv=pipeline_paths["fifa_csv"],
        source_systems={"matches": "t", "elo": "t", "fifa_rankings": "t"},
    )
    result = build_match_feature_mart(
        curated_dir=pipeline_paths["curated_dir"],
        feature_mart_dir=pipeline_paths["feature_mart_dir"],
        form_windows=[3, 5],
    )
    features = pd.read_parquet(result.output_path)
    final = features.loc[features["match_id"] == "wc2022_arg_fra_final"].iloc[0]
    assert final["home_matches_last3"] <= 3
    assert final["home_goals_for_last3"] is not None


def test_train_and_predict_pipeline(pipeline_paths):
    run_ingest(
        raw_dir=pipeline_paths["raw_dir"],
        curated_dir=pipeline_paths["curated_dir"],
        mappings_dir=pipeline_paths["mappings_dir"],
        matches_csv=pipeline_paths["matches_csv"],
        elo_csv=pipeline_paths["elo_csv"],
        fifa_csv=pipeline_paths["fifa_csv"],
        source_systems={"matches": "t", "elo": "t", "fifa_rankings": "t"},
    )
    build_match_feature_mart(
        curated_dir=pipeline_paths["curated_dir"],
        feature_mart_dir=pipeline_paths["feature_mart_dir"],
        form_windows=[3, 5, 10],
    )
    features = pd.read_parquet(pipeline_paths["feature_mart_dir"] / "match_features.parquet")
    train_result = train_baseline_dixon_coles(
        features=features,
        train_cutoff="2025-12-31T23:59:59",
        val_ratio=0.15,
        home_advantage_init=0.15,
        rho=-0.13,
        grid_max_goal=7,
        model_name="baseline_dixon_coles",
        model_version="test_v1",
        checkpoint_dir=pipeline_paths["checkpoint_dir"],
    )
    assert train_result.checkpoint_path.exists()
    assert train_result.train_nll > 0

    checkpoint = load_checkpoint(train_result.checkpoint_path)
    predictor = BaselinePredictor(checkpoint)
    prediction = predictor.predict_match_id(features, "wc2022_arg_fra_final")
    assert prediction.output.result_probs["home_win"] >= 0
    assert len(prediction.output.top3_scorelines) == 3
    total = sum(prediction.output.result_probs.values())
    assert abs(total - prediction.output.matrix.sum()) < 1e-6


def test_filter_before_excludes_equal_timestamp():
    df = pd.DataFrame({"kickoff_ts": ["2022-12-18T15:00:00Z"], "value": [1]})
    cutoff = pd.Timestamp("2022-12-18T15:00:00Z", tz="UTC")
    filtered = filter_before(df, cutoff.to_pydatetime(), "kickoff_ts")
    assert filtered.empty
