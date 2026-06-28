import numpy as np
import pytest

from worldcup.backtesting.metrics import (
    actual_result,
    brier_binary,
    evaluate_match,
    score_nll,
    top3_hit,
)
from worldcup.inference.decoder import decode_score_matrix
from worldcup.models.score_matrix import independent_score_matrix


def test_score_nll_on_grid_cell():
    matrix, overflow = independent_score_matrix(1.2, 1.0, grid_max_goal=7)
    value = score_nll(1, 0, matrix, overflow)
    assert value > 0


def test_evaluate_match_known_outcome():
    matrix, overflow = independent_score_matrix(1.4, 1.1, grid_max_goal=7)
    output = decode_score_matrix(matrix, overflow)
    metrics = evaluate_match("m1", 2, 1, output)
    assert metrics.brier_1x2 >= 0
    assert metrics.rps >= 0


def test_top3_hit():
    matrix = np.zeros((8, 8))
    matrix[1, 1] = 0.5
    matrix[2, 1] = 0.3
    matrix[0, 0] = 0.2
    output = decode_score_matrix(matrix)
    assert top3_hit(1, 1, output)
    assert not top3_hit(3, 3, output)


def test_brier_binary():
    assert brier_binary(0.7, True) == pytest.approx(0.09)


def test_actual_result():
    assert actual_result(2, 1) == "home_win"
    assert actual_result(0, 0) == "draw"
    assert actual_result(0, 1) == "away_win"
