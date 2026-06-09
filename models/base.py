"""Interface unifiée pour tous les modèles de prédiction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class MatchContext:
    is_neutral: bool = False
    venue: str = ""
    is_friendly: bool = False
    date: pd.Timestamp | None = None
    altitude_adj: float = 0.0


@dataclass
class Prediction:
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    btts: float
    expected_home: float = 0.0
    expected_away: float = 0.0
    uncertainty: dict[str, float] | None = None
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
            "over_2.5": self.over_2_5,
            "btts": self.btts,
            "expected_home": self.expected_home,
            "expected_away": self.expected_away,
            "uncertainty": self.uncertainty,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        return cls(
            home_win=d.get("home_win", 1 / 3),
            draw=d.get("draw", 1 / 3),
            away_win=d.get("away_win", 1 / 3),
            over_2_5=d.get("over_2.5", 0.5),
            btts=d.get("btts", 0.5),
            expected_home=d.get("expected_home", 0.0),
            expected_away=d.get("expected_away", 0.0),
            uncertainty=d.get("uncertainty"),
            source=d.get("source", ""),
        )


class MatchPredictor(Protocol):
    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> None: ...
    def predict_outcomes(
        self, home: str, away: str, context: MatchContext | None = None
    ) -> Prediction: ...
