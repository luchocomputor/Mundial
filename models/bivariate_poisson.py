"""Bivariate Poisson model (Karlis & Ntzoufras 2003)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from models.base import MatchContext, Prediction

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "artifacts" / "bivariate_poisson.pkl"


def _bivariate_pmf(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    """P(X=x, Y=y) pour bivariate Poisson avec lambda3 corrélant."""
    total = 0.0
    for k in range(min(x, y) + 1):
        log_p = (
            k * np.log(max(l3, 1e-10))
            + (x - k) * np.log(max(l1, 1e-10))
            + (y - k) * np.log(max(l2, 1e-10))
            - l1 - l2 - l3
            - gammaln(k + 1)
            - gammaln(x - k + 1)
            - gammaln(y - k + 1)
        )
        total += np.exp(log_p)
    return total


class BivariatePoissonModel:
    def __init__(self, decay: float = 0.0065, friendly_weight: float = 0.5):
        self.decay = decay
        self.friendly_weight = friendly_weight
        self.teams: list[str] = []
        self.alphas: dict[str, float] = {}
        self.betas: dict[str, float] = {}
        self.gamma: float = 0.0
        self.lambda3: float = 0.05
        self._fitted = False

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        if "time_weight" not in df.columns:
            if df["date"].dt.tz is None:
                df["date"] = df["date"].dt.tz_localize("UTC")
            df["days_ago"] = (reference_date - df["date"]).dt.days.clip(lower=0)
            df["time_weight"] = np.exp(-self.decay * df["days_ago"])

        def neg_ll(params):
            alphas = params[:n]
            betas = params[n : 2 * n]
            gamma = params[2 * n]
            l3 = max(params[2 * n + 1], 0.001)
            ll = 0.0
            for _, row in df.iterrows():
                hi = team_idx.get(row["home_team"])
                ai = team_idx.get(row["away_team"])
                if hi is None or ai is None:
                    continue
                is_neutral = bool(row.get("is_neutral", False))
                gamma_eff = 0.0 if is_neutral else gamma
                l1 = np.exp(alphas[hi] + betas[ai] + gamma_eff)
                l2 = np.exp(alphas[ai] + betas[hi])
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                w = float(row.get("time_weight", 1.0))
                p = _bivariate_pmf(hg, ag, l1, l2, l3)
                ll += w * np.log(max(p, 1e-15))
            return -ll

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.1
        x0[2 * n + 1] = 0.05
        result = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
        params = result.x
        alphas = params[:n] - params[:n].mean()
        self.alphas = {t: alphas[i] for i, t in enumerate(self.teams)}
        self.betas = {t: params[n + i] for i, t in enumerate(self.teams)}
        self.gamma = params[2 * n]
        self.lambda3 = max(params[2 * n + 1], 0.001)
        self._fitted = True

    def predict_outcomes(
        self, home: str, away: str, context: MatchContext | None = None
    ) -> Prediction:
        is_neutral = context.is_neutral if context else False
        ah = self.alphas.get(home, 0.0)
        bh = self.betas.get(home, 0.0)
        aa = self.alphas.get(away, 0.0)
        ba = self.betas.get(away, 0.0)
        gamma_eff = 0.0 if is_neutral else self.gamma
        l1 = np.exp(ah + ba + gamma_eff)
        l2 = np.exp(aa + bh)
        l3 = self.lambda3

        max_g = 8
        matrix = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                matrix[i, j] = _bivariate_pmf(i, j, l1, l2, l3)
        matrix /= matrix.sum()

        return Prediction(
            home_win=float(np.tril(matrix, -1).sum()),
            draw=float(np.trace(matrix)),
            away_win=float(np.triu(matrix, 1).sum()),
            over_2_5=float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i + j > 2.5)),
            btts=float(sum(matrix[i, j] for i in range(1, max_g + 1) for j in range(1, max_g + 1))),
            expected_home=l1 + l3,
            expected_away=l2 + l3,
            source="bivariate_poisson",
        )

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "BivariatePoissonModel":
        with open(path, "rb") as f:
            return pickle.load(f)
