import numpy as np

from worldcup.inference.decoder import decode_score_matrix
from worldcup.models.baseline.dixon_coles import DixonColesModel
from worldcup.models.score_matrix import independent_score_matrix


def test_independent_matrix_sums_to_one_with_overflow():
    matrix, overflow = independent_score_matrix(1.3, 1.1, grid_max_goal=7)
    assert abs(matrix.sum() + overflow - 1.0) < 1e-9


def test_decode_result_probs_sum_to_matrix_mass():
    matrix, overflow = independent_score_matrix(1.5, 1.2, grid_max_goal=7)
    output = decode_score_matrix(matrix, overflow_prob=overflow)
    total = sum(output.result_probs.values())
    assert abs(total - matrix.sum()) < 1e-6
    assert abs(matrix.sum() + overflow - 1.0) < 1e-6


def test_dixon_coles_output_shape():
    model = DixonColesModel(grid_max_goal=7)
    result = model.predict(lambda_home=1.4, lambda_away=1.0)
    assert result.matrix.shape == (8, 8)
    decoded = decode_score_matrix(result.matrix, overflow_prob=result.overflow_prob)
    assert len(decoded.top3_scorelines) == 3
    assert decoded.expected_goals["total"] > 0


def test_top3_sorted_descending():
    matrix = np.zeros((8, 8))
    matrix[2, 1] = 0.2
    matrix[1, 1] = 0.3
    matrix[0, 0] = 0.1
    matrix[1, 0] = 0.4
    output = decode_score_matrix(matrix)
    probs = [s.prob for s in output.top3_scorelines]
    assert probs == sorted(probs, reverse=True)
