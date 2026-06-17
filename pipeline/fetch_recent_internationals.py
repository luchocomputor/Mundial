"""Backfill des matchs internationaux SENIORS récents (2022-2024) via API-Football.

But : donner à l'Elo la forme récente qui lui manque (le parquet n'a que des CDM).
Amicaux + Nations League + Euro/Copa/AFCON/Asian Cup + qualifs CDM. Plan Free :
seules les saisons 2022-2024 sont accessibles. Pacé pour le rate-limit (~10/min).

→ data/raw/recent_internationals.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pipeline.fetch_api_football import get, remaining

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "raw" / "recent_internationals.parquet"

# (league_id, [saisons], libellé)  — sélections seniors hommes uniquement
PULLS = [
    (10,  [2022, 2023, 2024], "Friendlies"),
    (5,   [2022, 2024],       "UEFA Nations League"),
    (536, [2022, 2023, 2024], "CONCACAF Nations League"),
    (4,   [2024],             "Euro"),
    (960, [2023],             "Euro Qualification"),
    (9,   [2024],             "Copa America"),
    (6,   [2023],             "Africa Cup of Nations"),
    (36,  [2023],             "AFCON Qualification"),
    (7,   [2023],             "Asian Cup"),
    (35,  [2022, 2024],       "Asian Cup Qualification"),
    (29,  [2022, 2023],       "WCQ Africa"),
    (30,  [2022],             "WCQ Asia"),
    (31,  [2022],             "WCQ CONCACAF"),
    (32,  [2024],             "WCQ Europe"),
    (33,  [2022],             "WCQ Oceania"),
    (34,  [2022],             "WCQ South America"),
    (37,  [2022],             "WCQ Intercontinental"),
]
PACE_S = 6.0


def run():
    rows = []
    for league, seasons, label in PULLS:
        for season in seasons:
            if remaining() < 3:
                print("Budget quasi épuisé — arrêt (reprise demain).")
                _save(rows)
                return
            try:
                d = get("fixtures", {"league": league, "season": season})
            except Exception as e:
                print(f"  ! {label} {season} échec : {e}")
                time.sleep(PACE_S)
                continue
            n = 0
            for f in d.get("response", []):
                if f["fixture"]["status"]["short"] not in ("FT", "AET", "PEN"):
                    continue
                g = f["goals"]
                if g["home"] is None or g["away"] is None:
                    continue
                rows.append(dict(
                    date=f["fixture"]["date"],
                    home_team=f["teams"]["home"]["name"],
                    away_team=f["teams"]["away"]["name"],
                    home_goals=int(g["home"]),
                    away_goals=int(g["away"]),
                    league_id=league,
                    competition=label,
                    is_friendly=(league == 10),
                ))
                n += 1
            print(f"  {label:26s} {season}: {n:4d} matchs · budget {remaining()}")
            time.sleep(PACE_S)
    _save(rows)


def _save(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        print("Aucun match récupéré.")
        return
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"]).sort_values("date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n✅ {len(df)} matchs internationaux récents → {OUT}")
    print(df.groupby("competition").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    run()
