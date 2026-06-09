"""
Mise à jour quotidienne : cotes + prédictions + scan value bets.
Usage : python3 refresh.py
        python3 refresh.py --telegram   (envoie les alertes)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.dixon_coles import DixonColesModel
from pipeline.features import get_altitude_adjustment
from pipeline.fetch_predictions import blend_predictions

ROOT = Path(__file__).parent
TOKEN = None


def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def headers():
    return {"Authorization": f"Token {cfg()['bzzoiro']['token']}"}


def refresh_odds():
    """Re-fetch les cotes pour tous les matchs CDM 2026 à venir."""
    print("Rafraîchissement des cotes...")
    wc = pd.read_parquet(ROOT / "data" / "raw" / "wc_all.parquet")
    wc2026 = wc[(wc["date"].dt.year == 2026) & (wc["status"] == "notstarted")]
    real = wc2026[~wc2026["home_team"].str.match(r"^[W|L|R|Q|H|G|1|2|3]")]

    existing_path = ROOT / "data" / "raw" / "odds_wc2026.json"
    odds = json.loads(existing_path.read_text()) if existing_path.exists() else {}

    updated = 0
    for _, row in real.iterrows():
        fid = int(row["fixture_id"])
        r = requests.get(
            f"https://sports.bzzoiro.com/api/v2/events/{fid}/odds/",
            headers=headers(), timeout=10,
        )
        o = r.json().get("odds", {})
        non_null = {k: v for k, v in o.items() if v is not None}
        if non_null:
            odds[str(fid)] = non_null
            updated += 1
        time.sleep(0.1)

    existing_path.write_text(json.dumps(odds, indent=2))
    print(f"  Cotes: {updated} matchs mis à jour, {len(odds)} total")


def refresh_predictions():
    """Re-fetch les prédictions bzzoiro pour WC 2026."""
    print("Rafraîchissement des prédictions...")
    rows, offset = [], 0
    while True:
        r = requests.get(
            "https://sports.bzzoiro.com/api/v2/predictions/",
            headers=headers(), params={"league_id": 27, "limit": 100, "offset": offset}, timeout=15,
        )
        data = r.json()
        rows.extend(data.get("results", []))
        if not data.get("next"):
            break
        offset += 100
        time.sleep(0.2)

    preds = []
    for p in rows:
        ev = p["event"]
        m = p["markets"]
        mr = m.get("match_result", {})
        ou = m.get("over_under", {})
        bt = m.get("btts", {})
        xg = m.get("expected_goals", {})
        preds.append({
            "event_id": ev["id"], "home_team": ev["home_team"], "away_team": ev["away_team"],
            "date": ev["event_date"],
            "prob_home":   (mr.get("prob_home") or 0) / 100,
            "prob_draw":   (mr.get("prob_draw") or 0) / 100,
            "prob_away":   (mr.get("prob_away") or 0) / 100,
            "prob_over_25":(ou.get("prob_over_25") or 0) / 100,
            "prob_btts":   (bt.get("prob_yes") or 0) / 100,
            "xg_home": xg.get("home"), "xg_away": xg.get("away"),
        })

    pd.DataFrame(preds).to_parquet(ROOT / "data" / "raw" / "predictions_wc2026.parquet", index=False)
    print(f"  Prédictions: {len(preds)} matchs")


def scan_and_print(send_telegram: bool = False):
    from scan_now import scan
    bets = scan()
    if send_telegram and bets:
        asyncio.run(_send_alerts(bets))
    return bets


async def _send_alerts(bets):
    from telegram import Bot
    from pipeline.value_detector import format_alert
    c = cfg()
    bot = Bot(token=c["telegram"]["token"])
    chat_id = c["telegram"]["chat_id"]
    for bet in bets:
        try:
            await bot.send_message(chat_id=chat_id, text=format_alert(bet))
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"  Telegram error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--odds-only", action="store_true")
    args = parser.parse_args()

    refresh_odds()
    if not args.odds_only:
        refresh_predictions()
        scan_and_print(send_telegram=args.telegram)


if __name__ == "__main__":
    main()
