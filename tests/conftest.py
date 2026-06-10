"""Tests initiaux."""

import pytest


@pytest.fixture
def sample_odds_1x2():
    return {"home": 2.10, "draw": 3.40, "away": 3.20}
