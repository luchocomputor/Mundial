"""Télécharge l'historique des matchs tennis (résultats + cotes de clôture).

Source principale : tennis-data.co.uk — un xlsx par an et par tour, avec résultats
ET cotes de clôture Pinnacle/B365 (→ Elo + backtest CLV). Joignable depuis l'env
agent (contrairement à GitHub, filtré). Schéma : Date, Surface, Winner, Loser,
WRank/LRank, Comment, PSW/PSL…

    python -m pipeline.fetch_tennis_data --tour atp --since 2019
    python -m pipeline.fetch_tennis_data --tour wta --since 2019

Source complémentaire (qualif/challenger, bas-classés) : Jeff Sackmann sur GitHub.
FILTRÉ dans l'env agent → à lancer depuis le terminal local de Louis :
    python -m pipeline.fetch_tennis_data --tour atp --sackmann --since 2019

Destination : data/raw/tennis/<tour>/.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "tennis"

# tennis-data.co.uk : ATP = /{year}/{year}.xlsx ; WTA = /{year}w/{year}.xlsx
TD_URL = "http://www.tennis-data.co.uk/{path}/{year}.xlsx"
# Sackmann (GitHub, filtré dans l'env agent)
SACK_RAW = "https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{fname}"
QUAL_SUFFIX = {"atp": "qual_chall", "wta": "qual_itf"}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15"


def fetch_tennisdata(tour: str = "atp", since: int = 2019, until: int | None = None,
                     timeout: float = 30) -> list[Path]:
    until = until or date.today().year
    out_dir = DATA_DIR / tour
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for year in range(since, until + 1):
        path = str(year) if tour == "atp" else f"{year}w"
        url = TD_URL.format(path=path, year=year)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        except requests.RequestException as e:
            print(f"  ✗ {year} : {e}")
            continue
        if r.status_code != 200 or "openxml" not in r.headers.get("content-type", ""):
            print(f"  · {year} : absent (HTTP {r.status_code})")
            continue
        dest = out_dir / f"tennisdata_{tour}_{year}.xlsx"
        dest.write_bytes(r.content)
        print(f"  ✓ {year} : {len(r.content) // 1024} Ko → {dest.name}")
        saved.append(dest)
    return saved


def fetch_sackmann(tour: str = "atp", since: int = 2019, until: int | None = None,
                   timeout: float = 30) -> list[Path]:
    until = until or date.today().year
    out_dir = DATA_DIR / tour
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for year in range(since, until + 1):
        for fname in (f"{tour}_matches_{year}.csv",
                      f"{tour}_matches_{QUAL_SUFFIX[tour]}_{year}.csv"):
            url = SACK_RAW.format(tour=tour, fname=fname)
            try:
                r = requests.get(url, timeout=timeout)
            except requests.RequestException as e:
                print(f"  ✗ {fname} : {e}")
                continue
            if r.status_code != 200 or not r.text.startswith("tourney"):
                print(f"  · {fname} : HTTP {r.status_code} (egress GitHub filtré ? lance en local)")
                continue
            dest = out_dir / fname
            dest.write_text(r.text, encoding="utf-8")
            print(f"  ✓ {fname} : {r.text.count(chr(10))} lignes")
            saved.append(dest)
    return saved


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Fetch données tennis (résultats + cotes)")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--since", type=int, default=2019)
    ap.add_argument("--until", type=int, default=None)
    ap.add_argument("--sackmann", action="store_true",
                    help="utiliser Sackmann/GitHub (qualif/challenger) au lieu de tennis-data")
    args = ap.parse_args()
    src = "Sackmann" if args.sackmann else "tennis-data.co.uk"
    print(f"Téléchargement {args.tour.upper()} ({src}) {args.since}→{args.until or date.today().year} …")
    saved = (fetch_sackmann if args.sackmann else fetch_tennisdata)(args.tour, args.since, args.until)
    print(f"\n{len(saved)} fichiers → {DATA_DIR / args.tour}")


if __name__ == "__main__":
    _cli()
