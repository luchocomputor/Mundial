"""
Modèle Dixon-Coles pour prédire les scores de matchs de football.

Dixon & Coles (1997) : "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market."

Chaque équipe a un paramètre d'attaque (alpha) et de défense (beta).
home_goals ~ Poisson(alpha_home * beta_away * gamma)
away_goals ~ Poisson(alpha_away * beta_home)
Correction rho sur les scores serrés (0-0, 1-0, 0-1, 1-1).
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "dixon_coles.pkl"


def _tau(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    """Correction Dixon-Coles sur les scores faibles."""
    if x == 0 and y == 0:
        val = 1 - mu * nu * rho
    elif x == 1 and y == 0:
        val = 1 + nu * rho
    elif x == 0 and y == 1:
        val = 1 + mu * rho
    elif x == 1 and y == 1:
        val = 1 - rho
    else:
        return 1.0
    return max(val, 1e-6)


def _neg_log_likelihood(params: np.ndarray, teams: list[str], matches: pd.DataFrame) -> float:
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    alphas = params[:n]
    betas = params[n : 2 * n]
    gamma = params[2 * n]
    rho = params[2 * n + 1]

    log_lik = 0.0
    for _, row in matches.iterrows():
        hi = team_idx.get(row["home_team"])
        ai = team_idx.get(row["away_team"])
        if hi is None or ai is None:
            continue

        mu = np.exp(alphas[hi] + betas[ai] + gamma)
        nu = np.exp(alphas[ai] + betas[hi])
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        w = float(row.get("time_weight", 1.0))

        tau_val = _tau(hg, ag, mu, nu, rho)
        if tau_val <= 0:
            tau_val = 1e-10

        log_lik += w * (
            np.log(tau_val)
            + poisson.logpmf(hg, mu)
            + poisson.logpmf(ag, nu)
        )

    return -log_lik


class DixonColesModel:
    def __init__(self, decay: float = 0.0065, friendly_weight: float = 0.5):
        self.decay = decay
        self.friendly_weight = friendly_weight
        self.teams: list[str] = []
        self.alphas: dict[str, float] = {}
        self.betas: dict[str, float] = {}
        self.gamma: float = 0.0
        self.rho: float = 0.0
        self._fitted = False

        self.corner_alphas: dict[str, float] = {}
        self.corner_betas: dict[str, float] = {}
        self.corner_gamma: float = 0.0
        self._corners_fitted = False

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> None:
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        if "time_weight" not in df.columns:
            df["days_ago"] = (reference_date - pd.to_datetime(df["date"])).dt.days.clip(lower=0)
            df["time_weight"] = np.exp(-self.decay * df["days_ago"])
            if "is_friendly" in df.columns:
                df.loc[df["is_friendly"] == True, "time_weight"] *= self.friendly_weight

        self.teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        n = len(self.teams)

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.1
        x0[2 * n + 1] = 0.01

        bounds = (
            [(-3, 3)] * n
            + [(-3, 3)] * n
            + [(0, 1)]
            + [(-0.4, 0.4)]   # rho borné loin des extrema pour éviter tau<0
        )

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(self.teams, df),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        params = result.x
        self.alphas = {t: params[i] for i, t in enumerate(self.teams)}
        self.betas = {t: params[n + i] for i, t in enumerate(self.teams)}
        self.gamma = params[2 * n]
        self.rho = params[2 * n + 1]
        self._fitted = True
        print(f"Dixon-Coles fitted. gamma={self.gamma:.4f}, rho={self.rho:.4f}")

    def _get_lambdas(self, home: str, away: str, altitude_adj: float = 0.0):
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné. Appelle fit() d'abord.")
        ah = self.alphas.get(home, 0.0)
        bh = self.betas.get(home, 0.0)
        aa = self.alphas.get(away, 0.0)
        ba = self.betas.get(away, 0.0)
        mu = np.exp(ah + ba + self.gamma) + altitude_adj
        nu = np.exp(aa + bh) + altitude_adj
        return max(mu, 0.01), max(nu, 0.01)

    def predict_score_matrix(
        self, home: str, away: str, max_goals: int = 8, altitude_adj: float = 0.0
    ) -> np.ndarray:
        mu, nu = self._get_lambdas(home, away, altitude_adj)
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = _tau(i, j, mu, nu, self.rho)
                matrix[i, j] = tau * poisson.pmf(i, mu) * poisson.pmf(j, nu)
        matrix = matrix / matrix.sum()
        return matrix

    def predict_outcomes(
        self, home: str, away: str, altitude_adj: float = 0.0
    ) -> dict:
        mu, nu = self._get_lambdas(home, away, altitude_adj)
        matrix = self.predict_score_matrix(home, away, altitude_adj=altitude_adj)

        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())

        over_25 = float(
            sum(
                matrix[i, j]
                for i in range(matrix.shape[0])
                for j in range(matrix.shape[1])
                if i + j > 2.5
            )
        )

        btts = float(
            sum(
                matrix[i, j]
                for i in range(1, matrix.shape[0])
                for j in range(1, matrix.shape[1])
            )
        )

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over_2.5": over_25,
            "btts": btts,
            "expected_home": mu,
            "expected_away": nu,
        }

    def fit_corners(self, matches: pd.DataFrame) -> None:
        if "home_corners" not in matches.columns:
            print("Pas de données corners disponibles.")
            return

        df = matches.dropna(subset=["home_corners", "away_corners"]).copy()
        df["home_corners"] = df["home_corners"].astype(int)
        df["away_corners"] = df["away_corners"].astype(int)

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        def neg_ll_corners(params):
            alphas = params[:n]
            betas = params[n : 2 * n]
            gamma = params[2 * n]
            ll = 0.0
            for _, row in df.iterrows():
                hi = team_idx.get(row["home_team"])
                ai = team_idx.get(row["away_team"])
                if hi is None or ai is None:
                    continue
                mu = np.exp(alphas[hi] + betas[ai] + gamma)
                nu = np.exp(alphas[ai] + betas[hi])
                w = float(row.get("time_weight", 1.0))
                ll += w * (poisson.logpmf(int(row["home_corners"]), mu) + poisson.logpmf(int(row["away_corners"]), nu))
            return -ll

        x0 = np.zeros(2 * n + 1)
        x0[2 * n] = 0.1
        result = minimize(neg_ll_corners, x0, method="L-BFGS-B", options={"maxiter": 500})
        params = result.x
        self.corner_alphas = {t: params[i] for i, t in enumerate(teams)}
        self.corner_betas = {t: params[n + i] for i, t in enumerate(teams)}
        self.corner_gamma = params[2 * n]
        self._corners_fitted = True
        print("Modèle corners fitted.")

    def predict_corners(self, home: str, away: str) -> dict:
        if not self._corners_fitted:
            return {}
        ah = self.corner_alphas.get(home, 0.0)
        ba = self.corner_betas.get(away, 0.0)
        aa = self.corner_alphas.get(away, 0.0)
        bh = self.corner_betas.get(home, 0.0)
        mu = np.exp(ah + ba + self.corner_gamma)
        nu = np.exp(aa + bh)
        total = mu + nu
        p_over_9 = 1 - sum(poisson.pmf(k, total) for k in range(10))
        p_over_10 = 1 - sum(poisson.pmf(k, total) for k in range(11))
        return {
            "expected_home_corners": mu,
            "expected_away_corners": nu,
            "expected_total": total,
            "over_9.5": p_over_9,
            "over_10.5": p_over_10,
        }

    def save(self, path: Path = MODEL_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Modèle sauvegardé : {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "DixonColesModel":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2018, 2019, 2020, 2021, 2022])
    args = parser.parse_args()

    if args.train:
        from pipeline.features import load_raw_matches, build_training_data

        print("Chargement des données...")
        df = load_raw_matches()
        df = df[df["date"].dt.year.isin(args.seasons)]
        training = build_training_data(df)

        model = DixonColesModel()
        model.fit(training)
        model.save()
