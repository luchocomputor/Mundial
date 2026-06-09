"""
Détection de value bets par comparaison des probabilités modèle vs cotes bookmaker.
Kelly criterion pour le sizing des mises.
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from models.dixon_coles import DixonColesModel
from pipeline.features import get_altitude_adjustment
from pipeline.fetch_predictions import blend_predictions


ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"
BETS_LOG = ROOT / "data" / "bets_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_edge(p_model: float, cote: float) -> float:
    if cote <= 1:
        return -1.0
    p_implicite = 1.0 / cote
    return p_model - p_implicite


def kelly_size(p_model: float, cote: float, fraction: float = 0.25) -> float:
    b = cote - 1
    q = 1 - p_model
    full_kelly = (b * p_model - q) / b
    return max(0.0, full_kelly * fraction)


def compute_confidence(n_matches_home: int, n_matches_away: int, is_friendly: bool) -> int:
    """Étoiles de confiance 1-5 basées sur la quantité de données."""
    base = min(5, (n_matches_home + n_matches_away) // 10)
    if is_friendly:
        base = max(1, base - 1)
    return max(1, base)


def scan_value_bets(
    matches: list[dict],
    model: DixonColesModel,
    bookmaker_odds: dict,
    cfg: dict,
    bankroll: float,
    bzzoiro_predictions: dict | None = None,
) -> list[dict]:
    """
    matches           : liste de dicts avec les clés home_team, away_team, fixture_id, date, venue, is_friendly
    bookmaker_odds    : {fixture_id: {"1X2": {"home": cote, "draw": cote, "away": cote},
                                      "over_2.5": {"over": cote, "under": cote},
                                      "btts": {"yes": cote, "no": cote},
                                      "corners": {"over_9.5": cote, "over_10.5": cote}}}
    bzzoiro_predictions : {fixture_id: {"prob_home": p, "prob_draw": p, ...}} — optionnel
    """
    threshold = cfg["model"]["min_edge_threshold"]
    kelly_fraction = cfg["model"]["kelly_fraction"]
    max_kelly = cfg["model"]["max_kelly_bet"]
    dc_weight = cfg.get("blend", {}).get("dc_weight", 0.6)

    value_bets = []

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        fid = match.get("fixture_id")
        venue = match.get("venue", "")
        is_friendly = match.get("is_friendly", False)
        altitude_adj = get_altitude_adjustment(venue)

        try:
            dc_preds = model.predict_outcomes(home, away, altitude_adj=altitude_adj)
        except Exception as e:
            logger.warning(f"Prédiction impossible pour {home} vs {away}: {e}")
            continue

        bzzo_preds = (bzzoiro_predictions or {}).get(fid)
        preds = blend_predictions(dc_preds, bzzo_preds, dc_weight=dc_weight)

        corner_preds = model.predict_corners(home, away)
        odds = bookmaker_odds.get(fid, {})
        confidence = compute_confidence(10, 10, is_friendly)

        market_map = [
            ("1X2", "home", "home_win", preds["home_win"]),
            ("1X2", "draw", "draw", preds["draw"]),
            ("1X2", "away", "away_win", preds["away_win"]),
            ("over_2.5", "over", "over_2.5", preds["over_2.5"]),
            ("btts", "yes", "btts", preds["btts"]),
        ]

        if corner_preds:
            market_map += [
                ("corners", "over_9.5", "over_9.5", corner_preds.get("over_9.5", 0)),
                ("corners", "over_10.5", "over_10.5", corner_preds.get("over_10.5", 0)),
            ]

        for market, side, pred_key, p_model in market_map:
            cote = odds.get(market, {}).get(side)
            if not cote or cote <= 1:
                continue

            edge = compute_edge(p_model, cote)
            if edge < threshold:
                continue

            kelly = kelly_size(p_model, cote, kelly_fraction)
            kelly = min(kelly, max_kelly)
            stake_eur = round(bankroll * kelly, 2)

            value_bets.append({
                "fixture_id": fid,
                "date": match.get("date"),
                "home_team": home,
                "away_team": away,
                "venue": venue,
                "is_friendly": is_friendly,
                "market": market,
                "side": side,
                "p_model": round(p_model, 4),
                "p_implicit": round(1 / cote, 4),
                "edge": round(edge, 4),
                "cote": cote,
                "kelly_pct": round(kelly * 100, 2),
                "stake_eur": stake_eur,
                "bankroll": bankroll,
                "confidence": confidence,
                "altitude_adj": altitude_adj,
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def log_bets(bets: list[dict]) -> None:
    """Ajoute les bets détectés dans le CSV de log pour suivi ROI."""
    if not bets:
        return
    BETS_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BETS_LOG.exists()
    with open(BETS_LOG, "a", newline="") as f:
        fieldnames = list(bets[0].keys()) + ["detected_at", "result", "profit_eur"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for bet in bets:
            row = dict(bet)
            row["detected_at"] = datetime.now().isoformat()
            row["result"] = ""
            row["profit_eur"] = ""
            writer.writerow(row)
    logger.info(f"{len(bets)} bets loggés dans {BETS_LOG}")


def format_alert(bet: dict) -> str:
    stars = "★" * bet["confidence"] + "☆" * (5 - bet["confidence"])
    friendly_note = " (Amical)" if bet["is_friendly"] else ""
    altitude_note = f"\nAltitude ajustement : {bet['altitude_adj']:+.2f} buts" if bet["altitude_adj"] != 0 else ""
    return (
        f"🎯 VALUE BET DÉTECTÉE{friendly_note}\n\n"
        f"⚽ {bet['home_team']} vs {bet['away_team']}\n"
        f"📅 {bet['date']}\n"
        f"🏟️ {bet['venue']}{altitude_note}\n\n"
        f"Marché : {bet['market']} — {bet['side']}\n"
        f"Cote : {bet['cote']:.2f}\n"
        f"Probabilité modèle : {bet['p_model']*100:.1f}%\n"
        f"Probabilité implicite : {bet['p_implicit']*100:.1f}%\n"
        f"Edge : +{bet['edge']*100:.1f}% ✅\n\n"
        f"Kelly recommandé : {bet['kelly_pct']:.1f}% bankroll\n"
        f"Mise suggérée : {bet['stake_eur']:.2f}€ (bankroll {bet['bankroll']:.0f}€)\n\n"
        f"Confiance modèle : {stars}"
    )


def compute_roi_from_log() -> dict:
    """Calcule le ROI réel sur les bets loggés avec résultat renseigné."""
    if not BETS_LOG.exists():
        return {"n_bets": 0, "roi": 0.0, "profit": 0.0}
    df = pd.read_csv(BETS_LOG)
    df = df[df["result"].isin(["W", "L", "V"])]
    if df.empty:
        return {"n_bets": 0, "roi": 0.0, "profit": 0.0}
    df["profit_eur"] = pd.to_numeric(df["profit_eur"], errors="coerce").fillna(0)
    total_staked = df["stake_eur"].sum()
    total_profit = df["profit_eur"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    return {
        "n_bets": len(df),
        "roi": round(roi * 100, 2),
        "profit": round(total_profit, 2),
        "total_staked": round(total_staked, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Mode live : scanner les matchs à venir")
    parser.add_argument("--alert-telegram", action="store_true")
    parser.add_argument("--roi", action="store_true", help="Afficher le ROI du log")
    args = parser.parse_args()

    cfg = load_config()
    bankroll = cfg["bankroll"]["initial"]

    if args.roi:
        stats = compute_roi_from_log()
        print(f"ROI réel : {stats}")
    elif args.live:
        model = DixonColesModel.load()
        logger.info("Modèle chargé. En attente de cotes bookmaker...")
