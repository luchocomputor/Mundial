"""Charge les résultats de la CDM 2026 déjà jouée depuis le cache de scores.

Le dashboard persiste les scores terminés (API bzzoiro, league_id=27) dans
data/scores_cache.json — c'est la seule source des résultats 2026, le plan Free
d'API-Football n'ouvrant pas la saison courante. On les remet au schéma du
training Elo pour que le modèle apprenne de la forme du tournoi en cours (les
signaux les plus récents et les plus pertinents qu'on ait sur les 48 équipes).

→ consommé par models.train_elo.build_training()
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.teams import normalize

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "scores_cache.json"

_FINISHED = {"finished", "ft", "aet", "pen"}
_COLS = ["date", "home_team", "away_team", "home_goals", "away_goals",
         "is_neutral", "competition"]


def load_wc2026_results() -> pd.DataFrame:
    """Matchs CDM 2026 terminés → DataFrame au schéma du training Elo.

    is_neutral=True (terrain neutre, cohérent avec le traitement des CDM
    historiques dans wc_all). Dédoublonne les doublons de cache : une même
    affiche peut être ré-écrite sous une date-clé qui a dérivé entre deux runs
    du dashboard. Retourne un DataFrame vide si le cache n'existe pas encore.
    """
    if not CACHE.exists():
        return pd.DataFrame(columns=_COLS)

    raw = json.loads(CACHE.read_text())
    rows = []
    for key, info in raw.items():
        if (info.get("status") or "").lower() not in _FINISHED:
            continue
        sh, sa = info.get("sh"), info.get("sa")
        if sh is None or sa is None:
            continue
        rows.append(dict(
            date=pd.to_datetime(key.split("|")[-1], utc=True),  # date de la clé
            home_team=info.get("home_team", ""),
            away_team=info.get("away_team", ""),
            home_goals=int(sh),
            away_goals=int(sa),
            is_neutral=True,
            competition="World Cup 2026",
        ))

    df = pd.DataFrame(rows, columns=_COLS)
    if df.empty:
        return df

    # Une affiche se joue une fois ; les doublons (même affiche, même score)
    # viennent de la date-clé. On garde la date la plus tôt (= coup d'envoi réel).
    df["_h"] = df.home_team.map(normalize)
    df["_a"] = df.away_team.map(normalize)
    df = (df.sort_values("date")
            .drop_duplicates(subset=["_h", "_a", "home_goals", "away_goals"], keep="first")
            .drop(columns=["_h", "_a"])
            .reset_index(drop=True))
    return df


if __name__ == "__main__":
    df = load_wc2026_results()
    print(f"{len(df)} matchs CDM 2026 terminés "
          f"({df.date.min().date()} → {df.date.max().date()})" if len(df) else "Aucun match.")
    for _, r in df.iterrows():
        print(f"  {r.home_team:22s} {r.home_goals}-{r.away_goals} {r.away_team}")
