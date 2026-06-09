"""Elo et pi-ratings — baselines robustes pour sélections."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import MatchContext, Prediction

ROOT = Path(__file__).parent.parent
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "elo.pkl"


class EloRating:
    def __init__(
        self,
        k: float = 20.0,
        home_advantage: float = 100.0,
        initial_rating: float = 1500.0,
    ):
        self.k = k
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.ratings: dict[str, float] = {}
        self._fitted = False

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update(
        self,
        home: str,
        away: str,
        score_home: int,
        score_away: int,
        is_neutral: bool = False,
    ) -> None:
        rh = self._get(home)
        ra = self._get(away)
        ha = 0 if is_neutral else self.home_advantage

        exp_home = self._expected(rh + ha, ra)
        if score_home > score_away:
            actual_home = 1.0
        elif score_home == score_away:
            actual_home = 0.5
        else:
            actual_home = 0.0

        self.ratings[home] = rh + self.k * (actual_home - exp_home)
        self.ratings[away] = ra + self.k * ((1 - actual_home) - (1 - exp_home))

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> None:
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df = df.sort_values("date")
        self.ratings = {}

        for _, row in df.iterrows():
            self.update(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
                is_neutral=bool(row.get("is_neutral", False)),
            )
        self._fitted = True

    def predict_1x2(
        self, home: str, away: str, is_neutral: bool = False
    ) -> tuple[float, float, float]:
        rh = self._get(home)
        ra = self._get(away)
        ha = 0 if is_neutral else self.home_advantage
        p_home = self._expected(rh + ha, ra)
        p_away = self._expected(ra, rh + ha)
        p_draw = max(0.05, 1.0 - p_home - p_away + 0.25)
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def predict_outcomes(
        self, home: str, away: str, context: MatchContext | None = None
    ) -> Prediction:
        is_neutral = context.is_neutral if context else False
        p_h, p_d, p_a = self.predict_1x2(home, away, is_neutral)
        # Estimation OU/BTTS via lambdas implicites
        mu = 1.3 * (p_h / 0.33)
        nu = 1.0 * (p_a / 0.33)
        from scipy.stats import poisson

        over_25 = 1 - sum(
            poisson.pmf(i, mu) * poisson.pmf(j, nu)
            for i in range(6)
            for j in range(6)
            if i + j <= 2
        )
        btts = 1 - poisson.pmf(0, mu) * poisson.pmf(0, nu)

        return Prediction(
            home_win=p_h,
            draw=p_d,
            away_win=p_a,
            over_2_5=float(over_25),
            btts=float(btts),
            expected_home=mu,
            expected_away=nu,
            source="elo",
        )

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "EloRating":
        with open(path, "rb") as f:
            return pickle.load(f)
