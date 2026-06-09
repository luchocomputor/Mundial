"""Schémas et validation des données."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from pipeline.status import normalize_status

CompetitionType = Literal["wc", "qualifier", "nations_league", "continental", "friendly", "other"]
SnapshotType = Literal["open", "mid", "close"]


class MatchRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fixture_id: int
    date: datetime
    status: Literal["finished", "upcoming", "cancelled"]
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    league_id: int | None = None
    league_name: str = ""
    competition_type: CompetitionType = "other"
    is_friendly: bool = False
    is_neutral: bool = False


class OddsRow(BaseModel):
    fixture_id: int
    bookmaker: str
    snapshot_type: SnapshotType
    captured_at: datetime
    market: str
    side: str
    odds_decimal: float = Field(gt=1.0)


REQUIRED_MATCH_COLS = ["fixture_id", "date", "status", "home_team", "away_team"]


def validate_matches_df(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Valide et normalise un DataFrame de matchs."""
    if df.empty:
        return df

    out = df.copy()
    missing = [c for c in REQUIRED_MATCH_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    out["status"] = out["status"].apply(
        lambda s: normalize_status(s) if isinstance(s, str) else "upcoming"
    )
    out["date"] = pd.to_datetime(out["date"], utc=True)

    if out["fixture_id"].duplicated().any():
        out = out.drop_duplicates(subset=["fixture_id"], keep="last")

    if strict:
        upcoming = out[out["status"] == "upcoming"]
        if upcoming[["home_goals", "away_goals"]].notna().any().any():
            raise ValueError("Fuite de données : scores présents sur matchs upcoming")

    return out


def validate_odds_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if (out["odds_decimal"] <= 1.0).any():
        raise ValueError("Cotes invalides : odds_decimal doit être > 1.0")
    return out
