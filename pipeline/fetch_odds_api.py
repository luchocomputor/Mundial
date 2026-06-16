"""
Couche cotes via The Odds API — remplace bzzoiro.

- Betclic FR : les cotes où tu paries réellement (book de mise).
- Pinnacle   : le book le plus sharp au monde → ancre no-vig (meilleure
               estimation de la "vraie" probabilité qui existe).

Pré-match uniquement : les lignes live remontent incohérentes. Les matchs sont
rattachés aux fixtures wc_all par nom normalisé (pipeline.teams) + date Montréal
(pipeline.tz), puis indexés par fixture_id (contrat inchangé pour le dashboard).

Produit deux fichiers au format marché du dashboard :
  data/raw/betclic_odds_flat.json   (book de mise — lu par le dashboard)
  data/raw/pinnacle_odds_flat.json  (ancre sharp — passée au value detector)

Sur l'endpoint /odds standard, Betclic n'expose que le 1X2 et Pinnacle le
1X2 + une ligne O/U. L'ampleur (O/U multi-lignes, BTTS, corners) viendra
d'API-Football (phase suivante).

Usage : python -m pipeline.fetch_odds_api
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from pipeline.teams import normalize
from pipeline.tz import mtl_date

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
WC_PATH = DATA_RAW / "wc_all.parquet"
BETCLIC_FLAT = DATA_RAW / "betclic_odds_flat.json"
PINNACLE_FLAT = DATA_RAW / "pinnacle_odds_flat.json"
SPORT = "soccer_fifa_world_cup"


def _load_cfg() -> dict:
    """base_url/regions/sport depuis config.yaml ; clé depuis .env (ODDS_API_KEY)
    en priorité — la clé ne doit jamais vivre dans config.yaml (tracké git)."""
    load_dotenv(ROOT / ".env")
    cfg = (yaml.safe_load(open(ROOT / "config.yaml")) or {}).get("odds_api", {}) or {}
    env_key = os.environ.get("ODDS_API_KEY")
    if env_key:
        cfg["key"] = env_key
    return cfg


def _parse_book(markets: list, home: str, away: str) -> dict:
    """Marchés d'un bookmaker → format nested {"1X2": {...}, "over_2.5": {...}}."""
    out: dict[str, dict[str, float]] = {}
    for mk in markets:
        if mk["key"] == "h2h":
            o = {x["name"]: x["price"] for x in mk["outcomes"]}
            d = {}
            if o.get(home):  d["home"] = o[home]
            if o.get("Draw"): d["draw"] = o["Draw"]
            if o.get(away):  d["away"] = o[away]
            if len(d) == 3:
                out["1X2"] = d
        elif mk["key"] == "totals":
            for x in mk["outcomes"]:
                pt = x.get("point")
                if pt is None:
                    continue
                out.setdefault(f"over_{pt}", {})[x["name"].lower()] = x["price"]
    return out


def _swap_1x2(markets: dict) -> dict:
    """Inverse domicile/extérieur du 1X2 (orientation API ≠ wc_all)."""
    if "1X2" in markets:
        d = markets["1X2"]
        markets["1X2"] = {k: v for k, v in
                          {"home": d.get("away"), "draw": d.get("draw"), "away": d.get("home")}.items()
                          if v}
    return markets


def fetch_odds_api(cfg=None) -> dict:
    if cfg is None:
        cfg = _load_cfg()
    if not WC_PATH.exists():
        return {"matched": 0, "error": "wc_all.parquet manquant"}

    resp = requests.get(
        f"{cfg['base_url']}/sports/{SPORT}/odds",
        params={
            "apiKey": cfg["key"],
            "regions": cfg.get("regions", "eu"),
            "markets": "h2h,totals",
            "bookmakers": "pinnacle,betclic_fr",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    remaining = resp.headers.get("x-requests-remaining", "?")
    resp.raise_for_status()
    api = resp.json()

    # Lookup wc_all : paire non ordonnée + date Montréal → (fid, home_normalisé)
    wc = pd.read_parquet(WC_PATH)
    wc26 = wc[wc["date"].dt.year == 2026]
    lookup: dict = {}
    for _, x in wc26.iterrows():
        if pd.isna(x.fixture_id):
            continue
        key = (frozenset({normalize(x.home_team), normalize(x.away_team)}), mtl_date(x.date))
        lookup[key] = (str(int(x.fixture_id)), normalize(x.home_team))

    now = datetime.now(timezone.utc)
    betclic: dict = {}
    pinnacle: dict = {}
    unmatched: list[str] = []

    for m in api:
        if datetime.fromisoformat(m["commence_time"].replace("Z", "+00:00")) <= now:
            continue  # pré-match seulement
        nh, na = normalize(m["home_team"]), normalize(m["away_team"])
        rec = lookup.get((frozenset({nh, na}), mtl_date(m["commence_time"])))
        if not rec:
            unmatched.append(f"{m['home_team']} v {m['away_team']}")
            continue
        fid, wc_home = rec
        flip = wc_home != nh
        for b in m.get("bookmakers", []):
            mk = _parse_book(b["markets"], m["home_team"], m["away_team"])
            if flip:
                mk = _swap_1x2(mk)
            if not mk:
                continue
            (betclic if b["key"] == "betclic_fr" else pinnacle)[fid] = mk

    if betclic:
        BETCLIC_FLAT.write_text(json.dumps(betclic, indent=2, ensure_ascii=False))
    if pinnacle:
        PINNACLE_FLAT.write_text(json.dumps(pinnacle, indent=2, ensure_ascii=False))

    return {
        "matched": len(set(betclic) | set(pinnacle)),
        "betclic": len(betclic),
        "pinnacle": len(pinnacle),
        "unmatched": unmatched,
        "quota_remaining": remaining,
    }


if __name__ == "__main__":
    res = fetch_odds_api()
    if res.get("error"):
        print(f"Erreur : {res['error']}")
    else:
        print(f"Matchés : {res['matched']} | Betclic 1X2 : {res['betclic']} | "
              f"Pinnacle : {res['pinnacle']} | quota restant : {res['quota_remaining']}")
        if res["unmatched"]:
            print(f"Non matchés ({len(res['unmatched'])}) : {res['unmatched'][:8]}")
