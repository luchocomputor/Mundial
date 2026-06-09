import numpy as np

from evaluation.metrics import ranked_probability_score, expected_calibration_error


def test_rps_perfect_prediction():
    # Prédiction parfaite home win
    rps = ranked_probability_score(0, np.array([1.0, 0.0, 0.0]))
    assert rps == 0.0


def test_rps_uniform():
    rps = ranked_probability_score(0, np.array([1/3, 1/3, 1/3]))
    assert rps > 0


def test_ece_perfect_calibration():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = y_true.astype(float)
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece == 0.0
