import pytest

from pipeline.value_detector import compute_edge, kelly_size


def test_compute_edge_positive():
    assert compute_edge(0.55, 2.0) == pytest.approx(0.05, abs=0.001)


def test_compute_edge_negative():
    assert compute_edge(0.40, 2.0) < 0


def test_compute_edge_invalid_odds():
    assert compute_edge(0.5, 1.0) == -1.0


def test_kelly_size_positive():
    k = kelly_size(0.55, 2.0, fraction=0.25)
    assert k > 0


def test_kelly_size_zero_when_no_edge():
    k = kelly_size(0.40, 2.0, fraction=0.25)
    assert k == 0.0


def test_kelly_respects_cap():
    k = kelly_size(0.90, 10.0, fraction=1.0)
    assert k <= 1.0
