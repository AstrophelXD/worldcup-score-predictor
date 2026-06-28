import shutil

import pandas as pd
import pytest
import torch

from worldcup.data_ingestion.pipeline import run_ingest
from worldcup.features.builder import build_match_feature_mart
from worldcup.features.scoregen import fit_scoregen_spec, load_odds_table
from worldcup.inference.scoregen_predictor import ScoregenPredictor
from worldcup.models.advanced.bivariate_score_head import BivariateMixtureScoreHead
from worldcup.models.registry import load_scoregen_checkpoint
from worldcup.models.score_matrix_torch import dixon_coles_matrix, mixture_score_matrix
from worldcup.training.scoregen_trainer import (
    ScoregenTrainConfig,
    train_scoregen_football_transformer,
)
from worldcup.utils.paths import project_root


def test_mixture_score_matrix_sums_to_one():
    weights = torch.tensor([[0.5, 0.3, 0.2]])
    lambda_home = torch.tensor([[1.2, 1.0, 0.8]])
    lambda_away = torch.tensor([[1.0, 1.1, 0.9]])
    rho = torch.tensor([0.0])
    matrix = mixture_score_matrix(weights, lambda_home, lambda_away, rho, grid_max_goal=7)
    assert matrix.shape == (1, 8, 8)
    assert abs(float(matrix.sum()) - 1.0) < 1e-5


def test_bivariate_head_outputs_valid_distribution():
    head = BivariateMixtureScoreHead(input_dim=32, n_components=3, hidden_dim=32)
    context = torch.randn(2, 32)
    matrix, log_probs = head(context)
    assert matrix.shape == (2, 8, 8)
    assert log_probs.shape == (2, 64)
    assert torch.allclose(matrix.sum(dim=(1, 2)), torch.ones(2), atol=1e-5)


def test_dixon_coles_matrix_differentiable():
    lambda_home = torch.tensor([1.3], requires_grad=True)
    lambda_away = torch.tensor([1.1], requires_grad=True)
    rho = torch.tensor([0.0])
    matrix = dixon_coles_matrix(lambda_home, lambda_away, rho, grid_max_goal=7)
    loss = matrix[0, 1, 0]
    loss.backward()
    assert lambda_home.grad is not None
    assert lambda_away.grad is not None


@pytest.fixture
def pipeline_paths():
    root = project_root()
    tmp_root = root / "tests" / "tmp" / "scoregen"
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
        "players_csv": root / "data" / "samples" / "players.csv",
        "lineups_csv": root / "data" / "samples" / "lineups.csv",
        "player_stats_csv": root / "data" / "samples" / "player_match_stats.csv",
        "odds_csv": root / "data" / "samples" / "odds.csv",
    }
    yield paths
    shutil.rmtree(tmp_root, ignore_errors=True)


def test_scoregen_train_and_predict_smoke(pipeline_paths):
    run_ingest(
        raw_dir=pipeline_paths["raw_dir"],
        curated_dir=pipeline_paths["curated_dir"],
        mappings_dir=pipeline_paths["mappings_dir"],
        matches_csv=pipeline_paths["matches_csv"],
        elo_csv=pipeline_paths["elo_csv"],
        fifa_csv=pipeline_paths["fifa_csv"],
        players_csv=pipeline_paths["players_csv"],
        lineups_csv=pipeline_paths["lineups_csv"],
        player_stats_csv=pipeline_paths["player_stats_csv"],
        source_systems={
            "matches": "t",
            "elo": "t",
            "fifa_rankings": "t",
            "players": "t",
            "lineups": "t",
            "player_match_stats": "t",
        },
    )
    build_match_feature_mart(
        curated_dir=pipeline_paths["curated_dir"],
        feature_mart_dir=pipeline_paths["feature_mart_dir"],
        form_windows=[3, 5, 10],
    )
    features = pd.read_parquet(pipeline_paths["feature_mart_dir"] / "match_features.parquet")
    matches = pd.read_parquet(pipeline_paths["curated_dir"] / "matches.parquet")
    odds_df = load_odds_table(pipeline_paths["odds_csv"])
    from worldcup.features.scoregen import load_player_context

    player_context = load_player_context(pipeline_paths["curated_dir"])
    spec = fit_scoregen_spec(
        features,
        matches,
        seq_len=5,
        player_slots=11,
        odds_path=pipeline_paths["odds_csv"],
        player_context=player_context,
    )
    assert spec.player_dim == 6
    assert spec.odds_dim == 6
    assert odds_df.shape[0] >= 1

    result = train_scoregen_football_transformer(
        features=features,
        matches=matches,
        train_cutoff="2025-12-31T23:59:59",
        val_ratio=0.15,
        grid_max_goal=7,
        model_name="scoregen_football",
        model_version="test_v1",
        checkpoint_dir=pipeline_paths["checkpoint_dir"],
        train_cfg=ScoregenTrainConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            n_components=2,
            seq_len=5,
            player_slots=11,
            dropout=0.1,
            learning_rate=0.01,
            weight_decay=0.01,
            epochs=3,
            patience=2,
            batch_size=8,
            aux_loss_weight=0.1,
            seed=42,
            mixed_precision=None,
            odds_path=pipeline_paths["odds_csv"],
            curated_dir=pipeline_paths["curated_dir"],
        ),
    )
    assert result.checkpoint_path.exists()
    checkpoint, state_dict = load_scoregen_checkpoint(result.checkpoint_path)
    assert checkpoint.model_type == "scoregen"
    assert len(state_dict) > 0

    predictor = ScoregenPredictor.from_path(
        result.checkpoint_path,
        matches_path=pipeline_paths["curated_dir"] / "matches.parquet",
        odds_path=pipeline_paths["odds_csv"],
        curated_dir=pipeline_paths["curated_dir"],
    )
    prediction = predictor.predict_match_id(features, "wc2022_arg_fra_final")
    assert len(prediction.output.top3_scorelines) == 3
    total = sum(prediction.output.result_probs.values())
    assert abs(total - prediction.output.matrix.sum()) < 1e-6
