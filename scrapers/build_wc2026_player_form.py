"""Forme récente (saison courante) de chaque joueur CDM 2026 via TM ceapi.

Lit data/raw/transfermarkt/wc2026_squads.csv (player_id), récupère pour chacun
ses stats par compétition (matchs, buts, passes, minutes, % titu, cartons…) et
produit data/raw/transfermarkt/wc2026_player_form.csv.

Poli + reprenable (cache disque + pause du scraper). ~1248 joueurs → run long :
à lancer en arrière-plan. Usage : python -m scrapers.build_wc2026_player_form [limite]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scrapers.transfermarkt import CACHE, player_performance  # noqa: E402

SQUADS = CACHE / "wc2026_squads.csv"
OUT = CACHE / "wc2026_player_form.csv"

FIELDS = {
    "competition": "competitionDescription", "season": "nameSeason",
    "games": "gamesPlayed", "possible_games": "possibleGames",
    "goals": "goalsScored", "assists": "assists", "minutes": "minutesPlayed",
    "start11_pct": "startElevenPercent", "yellow": "yellowCards",
    "red": "redCards", "clean_sheets": "cleanSheets", "conceded": "concededGoals",
    # — champs étendus (consommés par models.gems si présents) —
    "minutes_pct": "minutesPlayedPercent", "second_yellow": "secondYellowCards",
    "gc_pct": "goalsContributedPercent",
    "gc_pct_nopen": "goalsContributedPercentWithoutPenalty",
}


def run(limit: int | None = None):
    sq = pd.read_csv(SQUADS)
    sq = sq[sq.player_id.notna()]
    if limit:
        sq = sq.head(limit)
    print(f"Joueurs à traiter : {len(sq)}")

    rows = []
    for i, (_, p) in enumerate(sq.iterrows(), 1):
        pid = int(p.player_id)
        try:
            comps = player_performance(pid)
        except Exception as e:
            print(f"  ! {p.player} ({pid}) : {e}")
            continue
        for c in comps:
            row = {"team": p.team, "player": p.player, "player_id": pid,
                   "position": p.position}
            for col, key in FIELDS.items():
                row[col] = c.get(key)
            rows.append(row)
        if i % 100 == 0:
            print(f"  … {i}/{len(sq)} joueurs")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n✅ {len(df)} lignes (joueur×compétition) · {df.player_id.nunique()} joueurs → {OUT}")
    if not df.empty:
        d = df.copy()
        d["goals"] = pd.to_numeric(d.goals, errors="coerce").fillna(0)
        d["minutes"] = pd.to_numeric(d.minutes.astype(str).str.replace(".", "", regex=False), errors="coerce").fillna(0)
        top = d.groupby(["team", "player"]).agg(buts=("goals", "sum"), min=("minutes", "sum")).reset_index()
        top = top.sort_values("buts", ascending=False).head(10)
        print("\nTop 10 buteurs (toutes comps, saison courante) :")
        for _, r in top.iterrows():
            print(f"  {int(r.buts):2d} buts  {r.player:24s} {r.team}  ({int(r['min'])} min)")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(lim)
