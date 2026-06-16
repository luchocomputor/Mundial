"""
Régénère data/raw/betclic_odds_flat.json — la source de cotes du dashboard.

Avant ce module, le fichier était orphelin : aucun script ne le produisait, le
dashboard lisait des cotes figées (régénérées à la main). On fetch ici les cotes
bzzoiro par fixture pour tous les matchs WC 2026 à venir et on les aplatit au
format marché consommé par scan_value_bets (réutilise _bzzoiro_to_market_odds,
source unique de la table de correspondance).

Usage : python -m pipeline.dashboard_odds [--all]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import bzzoiro_headers, load_config
from pipeline.placeholder import filter_real_teams
from pipeline.value_detector import _bzzoiro_to_market_odds

ROOT = Path(__file__).parent.parent
FLAT_PATH = ROOT / "data" / "raw" / "betclic_odds_flat.json"
WC_PATH = ROOT / "data" / "raw" / "wc_all.parquet"


def refresh_flat_odds(cfg=None, only_upcoming: bool = True) -> dict:
    """Fetch bzzoiro + écrit betclic_odds_flat.json. Ne touche au fichier que si
    au moins une cote a été mise à jour (sinon le mtime reste honnête pour le
    badge de fraîcheur)."""
    if cfg is None:
        cfg = load_config()
    if not WC_PATH.exists():
        return {"updated": 0, "total": 0, "error": "wc_all.parquet manquant"}

    wc = pd.read_parquet(WC_PATH)
    wc26 = wc[wc["date"].dt.year == 2026].copy()
    if only_upcoming:
        wc26 = wc26[wc26["date"].dt.date >= date.today()]
    real = filter_real_teams(wc26)

    flat = json.loads(FLAT_PATH.read_text()) if FLAT_PATH.exists() else {}
    headers = bzzoiro_headers(cfg)

    updated = 0
    for _, row in real.iterrows():
        if pd.isna(row["fixture_id"]):
            continue
        fid = str(int(row["fixture_id"]))
        try:
            r = requests.get(
                f"{cfg.bzzoiro_base_url}/events/{fid}/odds/",
                headers=headers,
                timeout=10,
            )
            if r.status_code != 200:
                continue
            raw = {k: v for k, v in (r.json().get("odds") or {}).items() if v}
            nested = _bzzoiro_to_market_odds(raw)
            if nested:
                flat[fid] = nested
                updated += 1
        except requests.RequestException:
            continue
        time.sleep(0.1)

    if updated > 0:
        FLAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FLAT_PATH.write_text(json.dumps(flat, indent=2, ensure_ascii=False))

    return {"updated": updated, "total": len(flat), "path": str(FLAT_PATH)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Inclure les matchs passés")
    args = parser.parse_args()
    res = refresh_flat_odds(only_upcoming=not args.all)
    if res.get("error"):
        print(f"Erreur : {res['error']}")
    else:
        print(f"Cotes : {res['updated']} mises à jour · {res['total']} matchs au total")
        print(f"→ {res['path']}")
