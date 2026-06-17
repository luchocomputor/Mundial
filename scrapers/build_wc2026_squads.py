"""Construit le dataset effectifs CDM 2026 (Transfermarkt) :
  1. extrait les 48 sélections depuis wc_all.parquet (phase de groupes)
  2. résout chaque verein_id via la recherche TM
  3. scrape chaque effectif (valeurs marchandes, postes, âges, sélections)
→ data/raw/transfermarkt/wc2026_verein_ids.json  (carte nom → id)
→ data/raw/transfermarkt/wc2026_squads.csv        (tous les joueurs)

Poli par construction (cache disque + pause 3s du scraper). Reprise gratuite :
les pages déjà en cache ne sont pas re-téléchargées.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scrapers.transfermarkt import CACHE, search_verein_id, squad  # noqa: E402

IDS_OUT = CACHE / "wc2026_verein_ids.json"
SQUADS_OUT = CACHE / "wc2026_squads.csv"


def wc2026_teams() -> list[str]:
    wc = pd.read_parquet(ROOT / "data" / "raw" / "wc_all.parquet")
    wc26 = wc[wc["date"].dt.year == 2026]
    ok = lambda n: bool(re.match(r"^[A-Za-zÀ-ÿ\s\-\'\\.&]+$", str(n))) and len(str(n)) > 2
    names = set()
    for _, r in wc26.iterrows():
        if ok(r.home_team):
            names.add(r.home_team)
        if ok(r.away_team):
            names.add(r.away_team)
    return sorted(names)


def run():
    teams = wc2026_teams()
    print(f"Sélections CDM 2026 détectées : {len(teams)}")

    ids: dict[str, int | None] = {}
    for n in teams:
        try:
            ids[n] = search_verein_id(n)
        except Exception as e:
            ids[n] = None
            print(f"  ! {n} : recherche échouée ({e})")
        print(f"  {n:26s} → verein {ids[n]}")
    CACHE.mkdir(parents=True, exist_ok=True)
    IDS_OUT.write_text(json.dumps(ids, ensure_ascii=False, indent=2))
    missing = [n for n, v in ids.items() if not v]
    print(f"\nRésolus : {sum(1 for v in ids.values() if v)}/{len(teams)}"
          + (f" · non résolus : {missing}" if missing else ""))

    rows = []
    for n, vid in ids.items():
        if not vid:
            continue
        try:
            sq = squad(vid)
        except Exception as e:
            print(f"  ! {n} (verein {vid}) : scrape échoué ({e})")
            continue
        for p in sq:
            p = {"team": n, "verein_id": vid, **p}
            rows.append(p)
        print(f"  {n:26s} {len(sq):2d} joueurs")

    df = pd.DataFrame(rows)
    df.to_csv(SQUADS_OUT, index=False)
    print(f"\n✅ {len(df)} joueurs · {df.team.nunique()} équipes → {SQUADS_OUT}")

    if not df.empty:
        val = (df.groupby("team").market_value_eur.sum() / 1e6).sort_values(ascending=False)
        print("\nTop 10 sélections par valeur d'effectif (M€) :")
        for t, v in val.head(10).items():
            print(f"  {t:24s} {v:6.0f} M€")


if __name__ == "__main__":
    run()
