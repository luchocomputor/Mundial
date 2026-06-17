"""Elo et pi-ratings — baselines robustes pour sélections."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import MatchContext, Prediction

ROOT = Path(__file__).parent.parent
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "elo.pkl"

# Mapping rating → buts attendus, d'où on dérive 1X2 / O-U / BTTS via une matrice
# de scores Poisson cohérente (au lieu d'un nul forcé à ~20 % et d'un split
# linéaire de l'espérance Elo, qui sur-cotaient la queue).
#   ELO_PER_GOAL  : écart Elo équivalant à 1 but de supériorité attendue (échelle
#                   du rating gap → décompresse la queue).
#   HOME_GOAL_ADV : bonus domicile EN BUTS (découplé de l'Elo home_advantage, qui
#                   ne sert qu'au fit ; ~0.35 but est réaliste, alors que 100 Elo /
#                   ELO_PER_GOAL valait 1 but entier → domicile sur-coté).
#   BASE_GOALS    : total de buts d'un match (moyenne empirique du training).
# Calibrés sur l'OOS walk-forward (EPG=85/HADV=0.35) : 1X2 log-loss 1.009→0.968,
# home/away ECE ÷3, et la queue (outsider d'un gros favori) passe de 0.21→0.13 prédit
# vs 0.13 réalisé. Le split est ADDITIF → mu+nu = BASE_GOALS constant : l'O/U est
# bien calibré mais sans résolution (l'Elo, un seul rating/équipe, ne porte pas le
# total de buts : corr |gap|↔total ≈ 0.10). Pour de vrais edges O/U → modèle buts
# attaque/défense (dixon_coles.py), pas l'Elo.
ELO_PER_GOAL = 85.0
HOME_GOAL_ADV = 0.35
BASE_GOALS = 2.71
MAX_GOALS = 10


def _norm_team(name: str) -> str:
    """Normalise un nom d'équipe (lazy import pour éviter tout cycle). Rend le
    modèle robuste aux variantes parquet/API (USA/United States, Czechia/Czech…)."""
    from pipeline.teams import normalize
    return normalize(name)


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
        self._source = "elo"  # tag model_version (surchargé par les sous-classes)

    def _get(self, team: str) -> float:
        return self.ratings.get(_norm_team(team), self.initial_rating)

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
        h, a = _norm_team(home), _norm_team(away)
        rh = self.ratings.get(h, self.initial_rating)
        ra = self.ratings.get(a, self.initial_rating)
        ha = 0 if is_neutral else self.home_advantage

        exp_home = self._expected(rh + ha, ra)
        if score_home > score_away:
            actual_home = 1.0
        elif score_home == score_away:
            actual_home = 0.5
        else:
            actual_home = 0.0

        self.ratings[h] = rh + self.k * (actual_home - exp_home)
        self.ratings[a] = ra + self.k * ((1 - actual_home) - (1 - exp_home))

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

    def _expected_goals(
        self, home: str, away: str, is_neutral: bool = False
    ) -> tuple[float, float]:
        """Écart de rating → buts attendus (mu_home, mu_away). La supériorité en
        buts croît avec l'écart Elo (ELO_PER_GOAL), répartie autour du total moyen
        BASE_GOALS. C'est le levier qui décompresse la queue : un gros favori a un
        mu_away faible → P(outsider) basse, au lieu du nul forcé à 20 % d'avant."""
        rh, ra = self._get(home), self._get(away)
        sup = (rh - ra) / ELO_PER_GOAL
        if not is_neutral:
            sup += HOME_GOAL_ADV
        mu = max(0.15, (BASE_GOALS + sup) / 2.0)
        nu = max(0.15, (BASE_GOALS - sup) / 2.0)
        return mu, nu

    @staticmethod
    def _score_matrix(mu: float, nu: float, max_goals: int = MAX_GOALS) -> np.ndarray:
        """Matrice P(home=i, away=j) sous Poisson indépendants."""
        from scipy.stats import poisson

        gh = poisson.pmf(np.arange(max_goals + 1), mu)
        ga = poisson.pmf(np.arange(max_goals + 1), nu)
        return np.outer(gh, ga)

    def predict_1x2(
        self, home: str, away: str, is_neutral: bool = False
    ) -> tuple[float, float, float]:
        mu, nu = self._expected_goals(home, away, is_neutral)
        m = self._score_matrix(mu, nu)
        z = float(m.sum())
        p_home = float(np.tril(m, -1).sum()) / z
        p_draw = float(np.trace(m)) / z
        p_away = float(np.triu(m, 1).sum()) / z
        return p_home, p_draw, p_away

    def predict_outcomes(
        self, home: str, away: str, context: MatchContext | None = None
    ) -> Prediction:
        is_neutral = context.is_neutral if context else False
        mu, nu = self._expected_goals(home, away, is_neutral)
        m = self._score_matrix(mu, nu)
        z = float(m.sum())

        p_h = float(np.tril(m, -1).sum()) / z
        p_d = float(np.trace(m)) / z
        p_a = float(np.triu(m, 1).sum()) / z

        idx = np.arange(m.shape[0])
        totals = idx[:, None] + idx[None, :]
        over_15 = float(m[totals > 1].sum()) / z
        over_25 = float(m[totals > 2].sum()) / z
        over_35 = float(m[totals > 3].sum()) / z
        btts = float(m[1:, 1:].sum()) / z

        return Prediction(
            home_win=p_h,
            draw=p_d,
            away_win=p_a,
            over_1_5=over_15,
            over_2_5=over_25,
            over_3_5=over_35,
            btts=btts,
            expected_home=mu,
            expected_away=nu,
            source=getattr(self, "_source", "elo"),
        )

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "EloRating":
        with open(path, "rb") as f:
            return pickle.load(f)
