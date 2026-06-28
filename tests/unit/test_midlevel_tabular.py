import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from worldcup.data_ingestion.pipeline import run_ingest
from worldcup.features.builder import build_match_feature_mart
from worldcup.features.tabular import (
    enrich_tabular_features,
    feature_columns,
    fit_feature_spec,
    vectorize_features,
)
from worldcup.inference.midlevel_predictor import MidlevelPredictor
from worldcup.models.midlevel.tabular_score import TabularScoreModel
from worldcup.models.registry import load_midlevel_checkpoint
from worldcup.training.midlevel_trainer import MidlevelTrainConfig, train_midlevel_tabular
from worldcup.utils.paths import project_root


def test_feature_columns_include_derived():
    columns = feature_columns()
    assert "elo_diff" in columns
    assert len(columns) == 4 + 8 + 2 + 12 + 5


def test_vectorize_features_shape():
    root = project_root()
    features = pd.read_parquet(root / "data" / "feature_mart" / "match_features.parquet")
    enriched = enrich_tabular_features(features.head(5))
    spec = fit_feature_spec(enriched)
    matrix = vectorize_features(enriched, spec)
    assert matrix.shape == (5, len(spec.feature_columns))
    assert np.isfinite(matrix).all()


def test_tabular_score_model_forward():
    model = TabularScoreModel(input_dim=33, hidden_dims=[16, 8], grid_max_goal=7, dropout=0.0)
    model.eval()
    batch = torch.randn(3, 33)
    logits = model(batch)
    assert logits.shape == (3, 64)
    matrix, overflow = model.predict_matrix(batch[:1])
    assert matrix.shape == (8, 8)
    assert abs(matrix.sum() - 1.0) < 1e-5
    assert overflow < 1e-6


@pytest.fixture
def pipeline_paths():
    root = project_root()
    tmp_root = root / "tests" / "tmp" / "midlevel"
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


def test_train_midlevel_smoke(pipeline_paths):
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
    result = train_midlevel_tabular(
        features=features,
        train_cutoff="2025-12-31T23:59:59",
        val_ratio=0.15,
        grid_max_goal=7,
        model_name="midlevel_tabular",
        model_version="test_v1",
        checkpoint_dir=pipeline_paths["checkpoint_dir"],
        train_cfg=MidlevelTrainConfig(
            hidden_dims=[32, 16],
            dropout=0.2,
            learning_rate=0.01,
            weight_decay=0.01,
            epochs=5,
            patience=3,
            batch_size=16,
            aux_loss_weight=0.1,
            seed=42,
            mixed_precision=None,
        ),
    )
    assert result.checkpoint_path.exists()
    assert result.train_nll > 0

    checkpoint, state_dict = load_midlevel_checkpoint(result.checkpoint_path)
    assert checkpoint.model_name == "midlevel_tabular"
    assert len(state_dict) > 0

    predictor = MidlevelPredictor.from_path(result.checkpoint_path)
    prediction = predictor.predict_match_id(features, "wc2022_arg_fra_final")
    assert prediction.output.result_probs["home_win"] >= 0
    assert len(prediction.output.top3_scorelines) == 3
    total = sum(prediction.output.result_probs.values())
    assert abs(total - prediction.output.matrix.sum()) < 1e-6
