"""Pull stats joueurs WC 2022 (API-Football) → data/raw/wc2022_players.csv.

But : dataset réel buteurs/minutes/postes pour CALIBRER models/goalscorer.py
(aujourd'hui heuristique : poids par poste + tiers manuels). Avec les vraies
minutes + buts par poste, on remplace les poids devinés par des taux empiriques.

Contraintes plan Free : 100 req/jour + ~10 req/min. Le client cache tout sur
disque (re-run gratuit) ; ici on pace à 6.5s/req et on s'arrête si le budget
journalier tombe sous 4. Reprise possible le lendemain (cache conservé).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.fetch_api_football import get, remaining  # noqa: E402

SEASON = 2022
OUT = ROOT / "data" / "raw" / "wc2022_players.csv"
PACE_S = 6.5  # ~9 req/min, sous la limite free


def _safe_get(endpoint, params, tries=3):
    for i in range(tries):
        try:
            d = get(endpoint, params)
            if d.get("errors"):
                print(f"  ! erreurs {endpoint} {params}: {d['errors']}")
            return d
        except Exception as e:
            print(f"  ! {endpoint} {params} échec ({e}) — retry {i+1}/{tries}")
            time.sleep(8)
    return {"response": [], "paging": {"total": 1}}


def run():
    print(f"Budget départ : {remaining()}/100")
    teams = _safe_get("teams", {"league": 1, "season": SEASON}).get("response", [])
    print(f"Équipes WC{SEASON} : {len(teams)}")

    rows = []
    stopped = False
    for t in teams:
        tid, tname = t["team"]["id"], t["team"]["name"]
        page, total = 1, 1
        while page <= total:
            if remaining() < 4:
                print("Budget quasi épuisé — arrêt propre (reprise demain).")
                stopped = True
                break
            d = _safe_get("players", {"league": 1, "season": SEASON, "team": tid, "page": page})
            for p in d.get("response", []):
                info, stt = p["player"], p["statistics"][0]
                g = stt.get("games", {})
                rows.append(dict(
                    team=tname,
                    player=info.get("name"),
                    age=info.get("age"),
                    position=g.get("position"),
                    appearances=g.get("appearences") or 0,
                    minutes=g.get("minutes") or 0,
                    goals=(stt.get("goals") or {}).get("total") or 0,
                    assists=(stt.get("goals") or {}).get("assists") or 0,
                    shots_total=(stt.get("shots") or {}).get("total") or 0,
                    shots_on=(stt.get("shots") or {}).get("on") or 0,
                    penalties=(stt.get("penalty") or {}).get("scored") or 0,
                ))
            total = d.get("paging", {}).get("total", 1)
            page += 1
            time.sleep(PACE_S)
        print(f"  {tname:18s} → {len([r for r in rows if r['team']==tname]):2d} joueurs · budget {remaining()}")
        if stopped:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["team", "player"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n✅ {len(df)} joueurs → {OUT}  ({'PARTIEL' if stopped else 'complet'})")

    # Aperçu calibration : taux buts / 90 min par poste
    if not df.empty:
        d = df[df.minutes >= 90].copy()
        d["g90"] = d.goals / (d.minutes / 90)
        print("\nButs / 90 min par poste (joueurs ≥90 min) — vs poids heuristiques actuels :")
        heur = {"Attacker": 3.5, "Midfielder": 1.0, "Defender": 0.25, "Goalkeeper": 0.02}
        for pos, grp in d.groupby("position"):
            print(f"  {pos:12s} n={len(grp):3d}  g/90 moyen={grp.g90.mean():.3f}  (poids heur. {heur.get(pos,'?')})")


if __name__ == "__main__":
    run()
