"""Stats avancées MLS via l'API publique American Soccer Analysis (xG/xA, SANS auth).

Comble le trou MLS de la couche style (Understat = top-5 only ; FBref = bloqué ici).
ASA est joignable depuis cet environnement (≠ FBref/Understat/Sofascore). Un appel
`/{league}/players/xgoals` donne par joueur×saison : minutes, shots, shots_on_target,
goals, key_passes, xgoals (xG), xassists (xA). On résout les noms, on écrit un CSV au
format que `build_wc2026_fbref.py` ingère (signature « asa ») dans data/raw/fbref/raw/.

ASA ne couvre que le foot nord-américain (mls, nwsl, uslc, usl1, mlsnp) — PAS la Liga MX
(→ FBref). Usage : python -m scrapers.fetch_asa_mls [saison]   (défaut : 2025)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
OUTDIR = ROOT / "data" / "raw" / "fbref" / "raw"
BASE = "https://app.americansocceranalysis.com/api/v1"
H = {"User-Agent": "Mozilla/5.0"}


def _get(path: str, **params) -> list:
    r = requests.get(BASE + path, headers=H, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def run(season: str = "2025", league: str = "mls", min_minutes: int = 400) -> pd.DataFrame:
    xg = _get(f"/{league}/players/xgoals", season_name=season, minimum_minutes=min_minutes)
    if not xg:
        print(f"ASA {league} {season} : aucune donnée (saison pas commencée ?).")
        return pd.DataFrame()
    df = pd.DataFrame(xg)
    names = {p["player_id"]: p["player_name"] for p in _get(f"/{league}/players")}
    teams = {t["team_id"]: t["team_name"] for t in _get(f"/{league}/teams")}
    # team_id peut être une liste (joueur transféré en cours de saison) → 1er club
    df["player_name"] = df["player_id"].map(names)
    df["team_title"] = df["team_id"].apply(
        lambda t: teams.get(t[0] if isinstance(t, list) else t, ""))
    cols = ["player_name", "team_title", "minutes_played", "shots", "shots_on_target",
            "goals", "key_passes", "xgoals", "xassists"]
    out = df[[c for c in cols if c in df.columns]].dropna(subset=["player_name"])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dest = OUTDIR / f"asa_{league}_{season}.csv"
    out.to_csv(dest, index=False)
    print(f"✅ ASA {league} {season} : {len(out)} joueurs (xG/xA) → {dest}")
    top = out.sort_values("xgoals", ascending=False).head(6)
    for _, r in top.iterrows():
        print(f"   {str(r.player_name)[:22]:22s} {str(r.team_title)[:18]:18s} "
              f"xG {r.xgoals:.1f} xA {r.xassists:.1f} ({int(r.minutes_played)}min)")
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "2025")
