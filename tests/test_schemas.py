import pandas as pd
import pytest

from pipeline.schemas import validate_matches_df
from pipeline.status import normalize_status


def test_validate_matches_basic():
    df = pd.DataFrame({
        "fixture_id": [1, 2],
        "date": ["2022-01-01", "2022-06-01"],
        "status": ["finished", "notstarted"],
        "home_team": ["France", "Brazil"],
        "away_team": ["Germany", "Argentina"],
        "home_goals": [2, None],
        "away_goals": [1, None],
    })
    out = validate_matches_df(df)
    assert len(out) == 2
    assert out.iloc[1]["status"] == "upcoming"


def test_validate_deduplicates():
    df = pd.DataFrame({
        "fixture_id": [1, 1],
        "date": ["2022-01-01", "2022-01-01"],
        "status": ["finished", "finished"],
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
    })
    out = validate_matches_df(df)
    assert len(out) == 1
