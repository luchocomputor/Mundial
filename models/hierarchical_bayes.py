"""Modèle hiérarchique bayésien avec pooling par confédération (PyMC)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import MatchContext, Prediction

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "artifacts" / "hierarchical_bayes.pkl"

CONFEDERATIONS = ["UEFA", "CONMEBOL", "AFC", "CAF", "CONCACAF", "OFC", "OTHER"]


class HierarchicalBayesModel:
    """
    Modèle bayésien hiérarchique simplifié.
    Utilise MAP/variational pour la prod ; NUTS pour recherche approfondie.
    """

    def __init__(self, decay: float = 0.0065):
        self.decay = decay
        self.team_attack: dict[str, float] = {}
        self.team_defense: dict[str, float] = {}
        self.conf_attack: dict[str, float] = {}
        self.conf_defense: dict[str, float] = {}
        self.gamma: float = 0.1
        self.uncertainty: dict[str, float] = {}
        self._fitted = False

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()

        try:
            self._fit_pymc(df, reference_date)
        except Exception as e:
            print(f"PyMC indisponible ou échec ({e}), fallback MLE hiérarchique")
            self._fit_mle_hierarchical(df, reference_date)

        self._fitted = True

    def _fit_mle_hierarchical(self, df: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        """Fallback : pooling partiel par confédération via shrinkage."""
        from models.dixon_coles import DixonColesModel

        dc = DixonColesModel(decay=self.decay)
        dc.fit(df, reference_date)
        self.team_attack = dc.alphas.copy()
        self.team_defense = dc.betas.copy()
        self.gamma = dc.gamma

        conf_teams: dict[str, list[str]] = {c: [] for c in CONFEDERATIONS}
        for team in dc.teams:
            conf = "OTHER"
            for _, row in df.iterrows():
                if row["home_team"] == team:
                    conf = row.get("home_confederation") or conf
                    break
                if row["away_team"] == team:
                    conf = row.get("away_confederation") or conf
                    break
            conf_teams.setdefault(conf, []).append(team)

        for conf, teams in conf_teams.items():
            if teams:
                self.conf_attack[conf] = float(np.mean([self.team_attack.get(t, 0) for t in teams]))
                self.conf_defense[conf] = float(np.mean([self.team_defense.get(t, 0) for t in teams]))

        n_matches = df.groupby("home_team").size().to_dict()
        for team in dc.teams:
            n = n_matches.get(team, 0)
            self.uncertainty[team] = 1.0 / (1 + n * 0.1)

    def _fit_pymc(self, df: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        import pymc as pm

        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        home_idx = [team_idx[t] for t in df["home_team"]]
        away_idx = [team_idx[t] for t in df["away_team"]]
        hg = df["home_goals"].values.astype(int)
        ag = df["away_goals"].values.astype(int)

        with pm.Model() as model:
            mu_att = pm.Normal("mu_att", 0, 1)
            sigma_att = pm.HalfNormal("sigma_att", 1)
            mu_def = pm.Normal("mu_def", 0, 1)
            sigma_def = pm.HalfNormal("sigma_def", 1)

            att = pm.Normal("att", mu_att, sigma_att, shape=n_teams)
            deff = pm.Normal("def", mu_def, sigma_def, shape=n_teams)
            gamma = pm.HalfNormal("gamma", 0.5)
            home_adv = pm.Data("home_adv", np.ones(len(hg)))

            mu = pm.math.exp(att[home_idx] + deff[away_idx] + gamma * home_adv)
            nu = pm.math.exp(att[away_idx] + deff[home_idx])

            pm.Poisson("home_goals", mu=mu, observed=hg)
            pm.Poisson("away_goals", mu=nu, observed=ag)

            idata = pm.fit(n=5000, method="advi")
            means = idata.mean["posterior"]

        self.team_attack = {teams[i]: float(means["att"].values[i]) for i in range(n_teams)}
        self.team_defense = {teams[i]: float(means["def"].values[i]) for i in range(n_teams)}
        self.gamma = float(means["gamma"].values)
        for team in teams:
            self.uncertainty[team] = 0.1

    def _get_params(self, team: str, conf: str = "OTHER") -> tuple[float, float, float]:
        att = self.team_attack.get(team, self.conf_attack.get(conf, 0.0))
        deff = self.team_defense.get(team, self.conf_defense.get(conf, 0.0))
        unc = self.uncertainty.get(team, 0.5)
        return att, deff, unc

    def predict_outcomes(
        self, home: str, away: str, context: MatchContext | None = None
    ) -> Prediction:
        is_neutral = context.is_neutral if context else False
        ah, bd_a, unc_h = self._get_params(home)
        aa, bd_h, unc_a = self._get_params(away)
        gamma_eff = 0.0 if is_neutral else self.gamma

        from scipy.stats import poisson

        mu = max(np.exp(ah + bd_a + gamma_eff), 0.01)
        nu = max(np.exp(aa + bd_h), 0.01)

        max_g = 8
        matrix = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                matrix[i, j] = poisson.pmf(i, mu) * poisson.pmf(j, nu)
        matrix /= matrix.sum()

        unc = {"home_win": (unc_h + unc_a) / 2, "over_2.5": (unc_h + unc_a) / 2}

        return Prediction(
            home_win=float(np.tril(matrix, -1).sum()),
            draw=float(np.trace(matrix)),
            away_win=float(np.triu(matrix, 1).sum()),
            over_2_5=float(sum(matrix[i, j] for i in range(max_g + 1) for j in range(max_g + 1) if i + j > 2.5)),
            btts=float(sum(matrix[i, j] for i in range(1, max_g + 1) for j in range(1, max_g + 1))),
            expected_home=mu,
            expected_away=nu,
            uncertainty=unc,
            source="hierarchical_bayes",
        )

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "HierarchicalBayesModel":
        with open(path, "rb") as f:
            return pickle.load(f)
