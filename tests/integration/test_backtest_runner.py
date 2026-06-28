import shutil

import pandas as pd
import pytest

from worldcup.backtesting.runner import run_world_cup_backtest
from worldcup.data_ingestion.pipeline import run_ingest
from worldcup.features.builder import build_match_feature_mart
from worldcup.utils.paths import project_root


@pytest.fixture
def backtest_paths():
    root = project_root()
    tmp_root = root / "tests" / "tmp" / "backtest"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)
    paths = {
        "raw_dir": tmp_root / "raw",
        "curated_dir": tmp_root / "curated",
        "feature_mart_dir": tmp_root / "feature_mart",
        "checkpoint_dir": tmp_root / "checkpoints",
        "report_dir": tmp_root / "reports",
        "mappings_dir": root / "data" / "external_mappings",
        "matches_csv": root / "data" / "samples" / "matches.csv",
        "elo_csv": root / "data" / "samples" / "elo.csv",
        "fifa_csv": root / "data" / "samples" / "fifa_rankings.csv",
    }
    yield paths
    shutil.rmtree(tmp_root, ignore_errors=True)


def _prepare_features(paths) -> pd.DataFrame:
    run_ingest(
        raw_dir=paths["raw_dir"],
        curated_dir=paths["curated_dir"],
        mappings_dir=paths["mappings_dir"],
        matches_csv=paths["matches_csv"],
        elo_csv=paths["elo_csv"],
        fifa_csv=paths["fifa_csv"],
        source_systems={"matches": "t", "elo": "t", "fifa_rankings": "t"},
    )
    build_match_feature_mart(
        curated_dir=paths["curated_dir"],
        feature_mart_dir=paths["feature_mart_dir"],
        form_windows=[3, 5],
    )
    return pd.read_parquet(paths["feature_mart_dir"] / "match_features.parquet")


def test_world_cup_2018_backtest(backtest_paths):
    features = _prepare_features(backtest_paths)
    result = run_world_cup_backtest(
        features=features,
        season=2018,
        val_ratio=0.15,
        home_advantage_init=0.15,
        rho=-0.13,
        grid_max_goal=7,
        model_name="baseline_dixon_coles",
        model_version="test",
        checkpoint_dir=backtest_paths["checkpoint_dir"],
        report_dir=backtest_paths["report_dir"],
    )
    assert result.report_path.exists()
    assert result.metrics["match_count"] > 0
    assert "score_nll" in result.metrics


def test_world_cup_2022_backtest(backtest_paths):
    features = _prepare_features(backtest_paths)
    result = run_world_cup_backtest(
        features=features,
        season=2022,
        val_ratio=0.15,
        home_advantage_init=0.15,
        rho=-0.13,
        grid_max_goal=7,
        model_name="baseline_dixon_coles",
        model_version="test",
        checkpoint_dir=backtest_paths["checkpoint_dir"],
        report_dir=backtest_paths["report_dir"],
    )
    assert result.metrics["match_count"] >= 5
