"""Prior valeur-marchande Transfermarkt mélangé à l'Elo.

Constat (validé OOS, 142 matchs 2024+ entre équipes CDM) : la valeur d'effectif
TM porte un signal de force que l'Elo, même enrichi, capture mal — surtout sur la
CDM 2026 (TM seul bat Elo seul : log-loss 1.30→1.13). Le blend 40 % Elo / 60 % TM
bat les deux (1.091→1.057). On mélange donc au niveau du RATING EFFECTIF : la
sous-classe `TMBlendedElo` surcharge `_get`, et tout le mapping Poisson de
`EloRating` (rating → buts → matrice de scores → 1X2/O-U/BTTS) est réutilisé tel
quel. Pas de TM pour une équipe (hors 48 CDM) → Elo pur (fallback transparent).

→ artefact `models/artifacts/elo_tm.pkl`, préféré par le model_loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.ratings import ARTIFACT_PATH, EloRating, _norm_team

ROOT = Path(__file__).parent.parent
TM_SQUADS = ROOT / "data" / "raw" / "transfermarkt" / "wc2026_squads.csv"
TM_ARTIFACT = ROOT / "models" / "artifacts" / "elo_tm.pkl"

# Poids de l'Elo dans le blend (1-w = poids TM). Optimum OOS sur 142 matchs 2024+
# (bassin plat 0.2-0.5) ; la CDM 2026 préfère encore plus de TM → 0.4 est prudent.
W_ELO = 0.4
TOP_N = 16  # on agrège la valeur des 16 joueurs les plus chers (cœur d'effectif)


def _team_logvalue() -> pd.Series:
    """Équipe normalisée → log(somme valeur des TOP_N joueurs)."""
    df = pd.read_csv(TM_SQUADS)
    df["mv"] = pd.to_numeric(df["market_value_eur"], errors="coerce").fillna(0)
    df["tn"] = df["team"].map(_norm_team)
    top = df.groupby("tn")["mv"].agg(lambda s: s.nlargest(TOP_N).sum())
    return np.log(top.clip(lower=1e6))


def build_tm_ratings(elo: EloRating) -> dict[str, float]:
    """log-valeur TM → rating sur l'échelle Elo, par régression linéaire sur les
    ratings Elo courants des équipes CDM (ne fixe que l'ÉCHELLE ; le classement
    vient de la valeur marchande, exogène aux résultats)."""
    logv = _team_logvalue()
    teams = list(logv.index)
    x = logv.values
    y = np.array([elo._get(t) for t in teams])
    slope, intercept = np.polyfit(x, y, 1)
    return {t: float(intercept + slope * logv[t]) for t in teams}


class TMBlendedElo(EloRating):
    """Elo dont le rating effectif (lu par `_get`, donc par tout le mapping de
    prédiction) est mélangé au prior TM. Le fit/update restent du pur Elo (ils
    écrivent `self.ratings` directement, sans passer par `_get`)."""

    def __init__(self, elo: EloRating, tm_rating: dict[str, float], w_elo: float = W_ELO):
        super().__init__(k=elo.k, home_advantage=elo.home_advantage,
                         initial_rating=elo.initial_rating)
        self.ratings = elo.ratings
        self._fitted = elo._fitted
        self.tm_rating = tm_rating
        self.w_elo = w_elo
        self._source = "elo_tm"  # tag model_version

    def _get(self, team: str) -> float:
        n = _norm_team(team)
        e = self.ratings.get(n, self.initial_rating)
        t = self.tm_rating.get(n)
        return e if t is None else self.w_elo * e + (1.0 - self.w_elo) * t


def build_blended(elo: EloRating | None = None, w_elo: float = W_ELO) -> TMBlendedElo:
    if elo is None:
        elo = EloRating.load()
    return TMBlendedElo(elo, build_tm_ratings(elo), w_elo)


def run():
    """Construit le modèle blendé depuis l'Elo courant et le sauvegarde."""
    if not ARTIFACT_PATH.exists():
        print("Pas d'elo.pkl — lance d'abord `python -m models.train_elo`.")
        return
    if not TM_SQUADS.exists():
        print(f"Pas de squads TM ({TM_SQUADS.name}) — blend ignoré, Elo pur conservé.")
        return
    elo = EloRating.load()
    blended = build_blended(elo)
    blended.save(TM_ARTIFACT)
    print(f"✅ Elo+TM blendé (w_elo={blended.w_elo}) → {TM_ARTIFACT.name} · "
          f"{len(blended.tm_rating)} équipes TM")


if __name__ == "__main__":
    run()
