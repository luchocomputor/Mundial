"""Réentraîne l'Elo sur CDM historiques + internationaux récents + CDM 2026.

Le modèle CDM-only était cassé (corrélation ~0 aux résultats, Brésil #26). Avec
les amicaux/qualifs/Nations League récents, les ratings redeviennent sensés. On
ajoute en bout de chaîne les matchs déjà joués de la CDM 2026 (scores_cache) :
ce sont les signaux les plus récents et les plus pertinents sur les 48 équipes,
et l'Elo séquentiel les traite en dernier → ils pèsent le plus sur les ratings
courants. Sauvegarde l'ancien artefact en .bak avant d'écraser.

Usage : python -m models.train_elo
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.ratings import ARTIFACT_PATH, EloRating
from pipeline.wc2026_results import load_wc2026_results

ROOT = Path(__file__).parent.parent
COLS = ["date", "home_team", "away_team", "home_goals", "away_goals", "is_neutral"]


def build_training() -> pd.DataFrame:
    wc = pd.read_parquet(ROOT / "data/raw/wc_all.parquet")
    hist = wc[(wc.date.dt.year < 2026) & wc.home_goals.notna()].copy()
    hist = hist[[c for c in COLS if c in hist.columns]]
    if "is_neutral" not in hist:
        hist["is_neutral"] = True  # CDM = terrain neutre

    rec = pd.read_parquet(ROOT / "data/raw/recent_internationals.parquet").copy()
    rec = rec[["date", "home_team", "away_team", "home_goals", "away_goals"]]
    rec["is_neutral"] = False  # qualifs/amicaux : domicile/extérieur réel

    wc2026 = load_wc2026_results()  # tournoi en cours (terrain neutre)

    train = pd.concat([hist, rec, wc2026], ignore_index=True)
    train = train.dropna(subset=["home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    return train


def run():
    train = build_training()
    print(f"Entraînement sur {len(train)} matchs "
          f"({train.date.min().date()} → {train.date.max().date()})")
    m = EloRating()
    m.fit(train, reference_date=pd.Timestamp("2026-06-11", tz="UTC"))

    if ARTIFACT_PATH.exists():
        bak = ARTIFACT_PATH.with_suffix(".pkl.cdmonly.bak")
        if not bak.exists():
            bak.write_bytes(ARTIFACT_PATH.read_bytes())
            print(f"Ancien modèle sauvegardé → {bak.name} (revert : le re-copier en elo.pkl)")
    m.save()
    print(f"✅ Elo enrichi actif → {ARTIFACT_PATH.name} · {len(m.ratings)} équipes")

    # Blend prior valeur-marchande TM (modèle de prod préféré par le model_loader).
    from models.tm_prior import run as build_tm_blend
    build_tm_blend()


if __name__ == "__main__":
    run()
