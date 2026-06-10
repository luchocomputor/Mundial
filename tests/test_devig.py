"""Tests dé-vigorisation."""

import pytest

from odds.devig import devig, devig_multiplicative, devig_shin


def test_devig_sums_to_one():
    odds = {"home": 2.0, "draw": 3.5, "away": 4.0}
    for method in ["multiplicative", "shin", "power"]:
        probs = devig(odds, method)
        assert abs(sum(probs.values()) - 1.0) < 0.01


def test_devig_multiplicative():
    odds = {"home": 2.0, "draw": 4.0, "away": 4.0}
    probs = devig_multiplicative(odds)
    assert len(probs) == 3
    assert all(0 < p < 1 for p in probs.values())


def test_devig_shin_reasonable():
    odds = {"home": 1.5, "draw": 4.0, "away": 6.0}
    probs = devig_shin(odds)
    assert probs["home"] > probs["away"]
