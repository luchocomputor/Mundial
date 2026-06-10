"""
Modèle Dixon-Coles pour prédire les scores de matchs de football.
Corrections : identifiabilité, terrain neutre, decay ancré, vectorisation partielle.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import poisson

from models.base import MatchContext, Prediction

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "dixon_coles.pkl"
CONF_PATH = ROOT / "data" / "confederations.yaml"


def _tau(x: int, y: int, mu: float, nu: float, rho: float) -> float:
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


def _neg_log_likelihood_vectorized(
    params: np.ndarray,
    teams: list[str],
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    is_neutral: np.ndarray,
) -> float:
    n = len(teams)
    alphas = params[:n]
    betas = params[n: 2 * n]
    gamma = params[2 * n]
    rho = params[2 * n + 1]

    gamma_eff = np.where(is_neutral, 0.0, gamma)
    mu = np.exp(alphas[home_idx] + betas[away_idx] + gamma_eff)
    nu = np.exp(alphas[away_idx] + betas[home_idx])

    log_lik = 0.0
    for i in range(len(hg)):
        tau_val = _tau(int(hg[i]), int(ag[i]), mu[i], nu[i], rho)
        log_lik += weights[i] * (
            np.log(max(tau_val, 1e-10))
            + poisson.logpmf(int(hg[i]), mu[i])
            + poisson.logpmf(int(ag[i]), nu[i])
        )
    return -log_lik


class DixonColesModel:
    def __init__(self, decay: float = 0.0065, friendly_weight: float = 0.5, l2_reg: float = 0.05):
        self.decay = decay
        self.friendly_weight = friendly_weight
        self.l2_reg = l2_reg
        self.teams: list[str] = []
        self.alphas: dict[str, float] = {}
        self.betas: dict[str, float] = {}
        self.gamma: float = 0.0
        self.rho: float = 0.0
        self.global_attack: float = 0.0
        self.global_defense: float = 0.0
        self.conf_attack: dict[str, float] = {}
        self.conf_defense: dict[str, float] = {}
        self.team_confederation: dict[str, str] = {}
        self._fitted = False
        self.reference_date: pd.Timestamp | None = None

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        if reference_date is None:
            raise ValueError("reference_date obligatoire")

        if reference_date.tzinfo is None:
            reference_date = reference_date.tz_localize("UTC")
        self.reference_date = reference_date

        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        if "time_weight" not in df.columns:
            if df["date"].dt.tz is None:
                df["date"] = df["date"].dt.tz_localize("UTC")
            df["days_ago"] = (reference_date - df["date"]).dt.days.clip(lower=0)
            df["time_weight"] = np.exp(-self.decay * df["days_ago"])
            if "is_friendly" in df.columns:
                df.loc[df["is_friendly"] == True, "time_weight"] *= self.friendly_weight

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        home_idx = np.array([team_idx[t] for t in df["home_team"]])
        away_idx = np.array([team_idx[t] for t in df["away_team"]])
        hg = df["home_goals"].values
        ag = df["away_goals"].values
        weights = df["time_weight"].values.astype(float)
        is_neutral = df.get("is_neutral", pd.Series([False] * len(df))).values.astype(bool)

        # Pré-compiler les arrays numpy — évite de passer le DataFrame à chaque éval
        valid = df[df["home_team"].isin(team_idx) & df["away_team"].isin(team_idx)]
        matches_arrays = (
            np.array([team_idx[t] for t in valid["home_team"]], dtype=np.int32),
            np.array([team_idx[t] for t in valid["away_team"]], dtype=np.int32),
            valid["home_goals"].to_numpy(dtype=np.float64),
            valid["away_goals"].to_numpy(dtype=np.float64),
            valid["time_weight"].to_numpy(dtype=np.float64),
        )

        # Initialiser alpha/beta à partir du classement FIFA comme prior informatif.
        # Sans prior, l'optimiseur ne distingue pas Spain de Cabo Verde sans
        # observations directes — il converge vers des valeurs arbitraires.
        from pipeline.features import get_fifa_ranking
        x0 = np.zeros(2 * n + 2)
        for i, team in enumerate(self.teams):
            rank = get_fifa_ranking(team)  # 1 = meilleur, 50 = faible
            # alpha (attaque) : top 5 ~ +0.6, rank 50 ~ -0.6
            x0[i] = np.clip(0.6 * (1 - (rank - 1) / 25), -0.8, 0.8)
            # beta (défense) : top 5 ~ -0.6 (peu de buts encaissés), rank 50 ~ 0
            x0[n + i] = np.clip(-0.6 * (1 - (rank - 1) / 25), -0.8, 0.0)
        x0[2 * n] = 0.1
        x0[2 * n + 1] = 0.01

        bounds = (
            [(-3, 3)] * n
            + [(-3, 3)] * n
            + [(0, 1)]
            + [(-0.4, 0.4)]
        )

        result = minimize(
            _neg_log_likelihood_vectorized,
            x0,
            args=(self.teams, home_idx, away_idx, hg, ag, weights, is_neutral),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        params = result.x
        alphas_raw = params[:n]
        betas_raw = params[n : 2 * n]

        # Contrainte identifiabilité : mean(attack) = 0
        alphas_centered = alphas_raw - alphas_raw.mean()
        self.alphas = {t: alphas_centered[i] for i, t in enumerate(self.teams)}
        self.betas = {t: betas_raw[i] for i, t in enumerate(self.teams)}
        self.global_attack = float(alphas_raw.mean())
        self.global_defense = float(betas_raw.mean())
        self.gamma = params[2 * n]
        self.rho = params[2 * n + 1]
        self._compute_confederation_priors()
        self._fitted = True
        print(f"Dixon-Coles fitted. gamma={self.gamma:.4f}, rho={self.rho:.4f}")

    def _load_confederation_map(self) -> dict[str, str]:
        if not CONF_PATH.exists():
            return {}
        with open(CONF_PATH) as f:
            data = yaml.safe_load(f) or {}
        return data.get("teams", {})

    def _compute_confederation_priors(self) -> None:
        """Moyennes attaque/défense par confédération pour shrink équipes sparse."""
        conf_map = self._load_confederation_map()
        self.team_confederation = {t: conf_map.get(t, "") for t in self.teams}

        conf_alphas: dict[str, list[float]] = {}
        conf_betas: dict[str, list[float]] = {}
        for team in self.teams:
            conf = self.team_confederation.get(team) or "UNKNOWN"
            conf_alphas.setdefault(conf, []).append(self.alphas[team])
            conf_betas.setdefault(conf, []).append(self.betas[team])

        self.conf_attack = {c: float(np.mean(v)) for c, v in conf_alphas.items() if v}
        self.conf_defense = {c: float(np.mean(v)) for c, v in conf_betas.items() if v}

    def _get_team_params(self, team: str) -> tuple[float, float]:
        """Shrink équipes inconnues vers prior confédération (pas moyenne globale)."""
        if team in self.alphas:
            return self.alphas[team], self.betas[team]

        conf_map = self.team_confederation or self._load_confederation_map()
        conf = conf_map.get(team, "")
        if conf and conf in self.conf_attack:
            return self.conf_attack[conf], self.conf_defense.get(conf, self.global_defense)
        return self.global_attack, self.global_defense

    def _get_lambdas(
        self,
        home: str,
        away: str,
        altitude_adj: float = 0.0,
        is_neutral: bool = False,
    ):
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné.")
        ah, bh = self._get_team_params(home)
        aa, ba = self._get_team_params(away)
        gamma_eff = 0.0 if is_neutral else self.gamma
        mu = max(np.exp(ah + ba + gamma_eff) + altitude_adj, 0.01)
        nu = max(np.exp(aa + bh) + altitude_adj, 0.01)
        return mu, nu

    def predict_score_matrix(
        self, home: str, away: str, max_goals: int = 8,
        altitude_adj: float = 0.0, is_neutral: bool = False,
    ) -> np.ndarray:
        mu, nu = self._get_lambdas(home, away, altitude_adj, is_neutral)
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                tau = _tau(i, j, mu, nu, self.rho)
                matrix[i, j] = tau * poisson.pmf(i, mu) * poisson.pmf(j, nu)
        return matrix / matrix.sum()

    def predict_outcomes(
        self,
        home: str,
        away: str,
        altitude_adj: float = 0.0,
        is_neutral: bool = False,
        context: MatchContext | None = None,
    ) -> dict | Prediction:
        if context:
            altitude_adj = context.altitude_adj or altitude_adj
            is_neutral = context.is_neutral

        mu, nu = self._get_lambdas(home, away, altitude_adj, is_neutral)
        matrix = self.predict_score_matrix(home, away, altitude_adj=altitude_adj, is_neutral=is_neutral)

        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())
        over_25 = float(sum(matrix[i, j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if i + j > 2.5))
        btts = float(sum(matrix[i, j] for i in range(1, matrix.shape[0]) for j in range(1, matrix.shape[1])))

        return Prediction(
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            over_2_5=over_25,
            btts=btts,
            expected_home=mu,
            expected_away=nu,
            source="dixon_coles",
        )

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

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
        from pipeline.config import load_config
        from pipeline.features import build_training_data, load_raw_matches

        cfg = load_config()
        df = load_raw_matches()
        df = df[df["date"].dt.year.isin(args.seasons)]
        ref = pd.Timestamp(f"{max(args.seasons)}-12-31", tz="UTC")
        training = build_training_data(
            df,
            friendly_weight=cfg.model.friendly_weight,
            reference_date=ref,
            decay=cfg.model.decay,
        )
        model = DixonColesModel(decay=cfg.model.decay, friendly_weight=cfg.model.friendly_weight)
        model.fit(training, reference_date=ref)
        model.save()
