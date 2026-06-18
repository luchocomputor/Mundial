"""
Mise à jour quotidienne : fetch → build → train → scan paper → CLV.
Usage : python refresh.py [--telegram] [--full]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.build_dataset import build_master_dataset
from pipeline.config import load_config
from pipeline.fetch_data import fetch_and_save_all
from pipeline.fetch_odds import backfill_odds_history, save_odds_history
from pipeline.production import alerts_allowed

ROOT = Path(__file__).parent


def refresh_odds():
    print("Rafraîchissement des cotes...")
    cfg = load_config()
    wc_path = ROOT / "data" / "raw" / "wc_all.parquet"
    all_path = ROOT / "data" / "raw" / "all_matches.parquet"
    path = wc_path if wc_path.exists() else all_path
    if not path.exists():
        print("  Pas de matchs — skip odds")
        return

    from pipeline.config import bzzoiro_headers
    from pipeline.placeholder import filter_real_teams

    wc = pd.read_parquet(path)
    wc2026 = wc[(wc["date"].dt.year == 2026) & (wc["status"].isin(["upcoming", "notstarted"]))]
    real = filter_real_teams(wc2026)

    existing_path = ROOT / "data" / "raw" / "odds_wc2026.json"
    odds = json.loads(existing_path.read_text()) if existing_path.exists() else {}

    updated = 0
    for _, row in real.iterrows():
        fid = int(row["fixture_id"])
        r = requests.get(
            f"{cfg.bzzoiro_base_url}/events/{fid}/odds/",
            headers=bzzoiro_headers(cfg),
            timeout=10,
        )
        o = r.json().get("odds", {})
        non_null = {k: v for k, v in o.items() if v is not None}
        if non_null:
            odds[str(fid)] = non_null
            updated += 1
        time.sleep(0.1)

    existing_path.write_text(json.dumps(odds, indent=2))
    print(f"  Cotes: {updated} mis à jour, {len(odds)} total")


def refresh_predictions():
    print("Rafraîchissement des prédictions...")
    from pipeline.config import bzzoiro_headers
    cfg = load_config()
    rows, offset = [], 0
    while True:
        r = requests.get(
            f"{cfg.bzzoiro_base_url}/predictions/",
            headers=bzzoiro_headers(cfg),
            params={"league_id": 27, "limit": 100, "offset": offset},
            timeout=15,
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
            "event_id": ev["id"],
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "date": ev["event_date"],
            "prob_home": (mr.get("prob_home") or 0) / 100,
            "prob_draw": (mr.get("prob_draw") or 0) / 100,
            "prob_away": (mr.get("prob_away") or 0) / 100,
            "prob_over_25": (ou.get("prob_over_25") or 0) / 100,
            "prob_btts": (bt.get("prob_yes") or 0) / 100,
            "xg_home": xg.get("home"),
            "xg_away": xg.get("away"),
        })

    out = ROOT / "data" / "raw" / "predictions_wc2026.parquet"
    pd.DataFrame(preds).to_parquet(out, index=False)
    print(f"  Prédictions: {len(preds)} matchs")


async def _send_alerts(bets):
    from telegram import Bot
    from pipeline.value_detector import format_alert

    cfg = load_config()
    if not cfg.telegram_token or not alerts_allowed(cfg):
        print("  Alertes désactivées (paper_only ou NO_GO)")
        return
    bot = Bot(token=cfg.telegram_token)
    for bet in bets:
        try:
            await bot.send_message(chat_id=cfg.telegram_chat_id, text=format_alert(bet))
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"  Telegram error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--full", action="store_true", help="Fetch complet + build dataset")
    parser.add_argument("--train", action="store_true", help="Entraînement hebdomadaire")
    parser.add_argument("--odds-only", action="store_true")
    args = parser.parse_args()

    if args.full:
        fetch_and_save_all()
        matches_path = ROOT / "data" / "raw" / "all_matches.parquet"
        if matches_path.exists():
            matches = pd.read_parquet(matches_path)
            odds_df = backfill_odds_history(matches, max_events=500, resume=True)
            if not odds_df.empty:
                save_odds_history(odds_df)
        build_master_dataset()

    if args.train:
        # Pipeline gated : Dixon-Coles, ensemble, rapport de gates sur l'historique.
        from scripts.train_pipeline import run_train_pipeline
        run_train_pipeline()
        # Elo + calibrateur de PROD : source enrichie (internationaux récents +
        # CDM 2026 en cours via scores_cache). En dernier → c'est le modèle que
        # charge le value detector ; il intègre les résultats du tournoi au lieu
        # d'être écrasé par la version ≤2025 du pipeline gated.
        from models.train_elo import run as train_elo
        from models.train_calibrator import run as train_calibrator
        train_elo()
        train_calibrator()

    refresh_odds()
    if not args.odds_only:
        refresh_predictions()
        from scan_now import scan
        bets = scan(paper=True)
        # Auto-résolution riche (score + buteurs) des paris dont le match est joué.
        from pipeline.resolve_scores import run as resolve_scores
        rr = resolve_scores(dry_run=False)
        if rr["resolved"]:
            print(f"  Résolus: {rr['resolved']} paris auto-résolus")

        from monitoring.paper_trading import fetch_and_update_clv, snapshot_closing_lines
        n_snap = snapshot_closing_lines()  # fige les lignes sharp des matchs à venir
        n_clv = fetch_and_update_clv()
        if n_snap or n_clv:
            print(f"  Lignes de clôture: {n_snap} figées · CLV mis à jour pour {n_clv} paris")

        # Mode Œil : cotes buteur des pronos à l'œil + leur CLV.
        from pipeline.eye import fetch_eye_clv, snapshot_eye_closing
        snapshot_eye_closing()
        n_eye = fetch_eye_clv()
        if n_eye:
            print(f"  CLV œil: {n_eye} pronos")
        if args.telegram and bets:
            asyncio.run(_send_alerts(bets))


if __name__ == "__main__":
    main()
