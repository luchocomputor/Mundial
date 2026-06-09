"""
Scanner de value bets CDM 2026 — à lancer dès le modèle entraîné.
Usage : python3 scan_now.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.dixon_coles import DixonColesModel
from pipeline.fetch_predictions import blend_predictions
from pipeline.features import get_altitude_adjustment
from pipeline.value_detector import compute_edge, kelly_size, log_bets, format_alert

ROOT = Path(__file__).parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_venue_map():
    p = ROOT / "data" / "raw" / "venues.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df["venue_id"], df["name"]))


def format_odds_key(bzzoiro_key: str) -> tuple[str, str]:
    """Convertit les clés bzzoiro en (marché, side) pour le scanner."""
    mapping = {
        "home_win":      ("1X2", "home"),
        "draw":          ("1X2", "draw"),
        "away_win":      ("1X2", "away"),
        "over_25_goals": ("over_2.5", "over"),
        "under_25_goals":("over_2.5", "under"),
        "btts_yes":      ("btts", "yes"),
        "btts_no":       ("btts", "no"),
    }
    return mapping.get(bzzoiro_key, (None, None))


def get_model_prob(preds: dict, market: str, side: str) -> float | None:
    if market == "1X2":
        return {"home": preds["home_win"], "draw": preds["draw"], "away": preds["away_win"]}.get(side)
    if market == "over_2.5":
        if side == "over":  return preds["over_2.5"]
        if side == "under": return 1 - preds["over_2.5"]
    if market == "btts":
        if side == "yes": return preds["btts"]
        if side == "no":  return 1 - preds["btts"]
    return None


def scan():
    cfg = load_config()
    threshold = cfg["model"]["min_edge_threshold"]
    kelly_fraction = cfg["model"]["kelly_fraction"]
    max_kelly = cfg["model"]["max_kelly_bet"]
    dc_weight = cfg.get("blend", {}).get("dc_weight", 0.6)
    bankroll = cfg["bankroll"]["initial"]

    print("Chargement modèle + données...")
    model = DixonColesModel.load()
    venue_map = load_venue_map()

    wc = pd.read_parquet(ROOT / "data" / "raw" / "wc_all.parquet")
    wc2026 = wc[wc["date"].dt.year == 2026]
    matches = wc2026[~wc2026["home_team"].str.match(r"^[W|L|R|Q|H|G|1|2|3]")].sort_values("date")

    odds_raw = json.loads((ROOT / "data" / "raw" / "odds_wc2026.json").read_text())
    odds = {int(k): v for k, v in odds_raw.items()}

    bzzo_df = pd.read_parquet(ROOT / "data" / "raw" / "predictions_wc2026.parquet")
    bzzo_map = {int(row["event_id"]): row.to_dict() for _, row in bzzo_df.iterrows()}

    print(f"Scan de {len(matches)} matchs CDM 2026 ({len(odds)} avec cotes)...\n")

    value_bets = []
    for _, row in matches.iterrows():
        fid = int(row["fixture_id"])
        home, away = row["home_team"], row["away_team"]
        venue = venue_map.get(row.get("venue_id"), "")
        altitude_adj = get_altitude_adjustment(venue)
        match_odds = odds.get(fid, {})
        if not match_odds:
            continue

        try:
            dc_preds = model.predict_outcomes(home, away, altitude_adj=altitude_adj)
        except Exception:
            continue

        bzzo_row = bzzo_map.get(fid)
        preds = blend_predictions(dc_preds, bzzo_row, dc_weight=dc_weight)

        for odds_key, cote in match_odds.items():
            if not cote or cote <= 1:
                continue
            market, side = format_odds_key(odds_key)
            if market is None:
                continue
            p_model = get_model_prob(preds, market, side)
            if p_model is None:
                continue

            edge = compute_edge(p_model, cote)
            if edge < threshold:
                continue

            kelly = min(kelly_size(p_model, cote, kelly_fraction), max_kelly)
            stake = round(bankroll * kelly, 2)

            value_bets.append({
                "fixture_id": fid,
                "date": row["date"].strftime("%d/%m %H:%M") if hasattr(row["date"], "strftime") else str(row["date"])[:16],
                "home_team": home,
                "away_team": away,
                "venue": venue,
                "group": row.get("group_name", ""),
                "is_friendly": False,
                "market": market,
                "side": side,
                "p_model": round(p_model, 4),
                "p_implicit": round(1 / cote, 4),
                "edge": round(edge, 4),
                "cote": cote,
                "kelly_pct": round(kelly * 100, 2),
                "stake_eur": stake,
                "bankroll": bankroll,
                "confidence": 3,
                "altitude_adj": altitude_adj,
                "dc_source": round(dc_preds.get("home_win" if side == "home" else "away_win" if side == "away" else "draw", 0), 4),
                "bzzo_source": round(bzzo_row.get("prob_home" if side == "home" else "prob_away" if side == "away" else "prob_draw", 0), 4) if bzzo_row else None,
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)

    if not value_bets:
        print("Aucun value bet détecté avec edge > 5%.")
        return value_bets

    print(f"{'Match':<32} {'Marché':<18} {'Cote':>5} {'Modèle':>7} {'Impl.':>6} {'Edge':>6} {'Kelly':>6} {'Mise':>6}")
    print("─" * 95)
    for b in value_bets:
        match_str = f"{b['home_team']} vs {b['away_team']}"[:31]
        market_str = f"{b['market']} {b['side']}"
        print(f"{match_str:<32} {market_str:<18} {b['cote']:5.2f} {b['p_model']:7.1%} {b['p_implicit']:6.1%} {b['edge']:6.1%} {b['kelly_pct']:5.1f}% {b['stake_eur']:5.2f}€")

    print(f"\n{len(value_bets)} value bets | Mise totale suggérée: {sum(b['stake_eur'] for b in value_bets):.2f}€ / {bankroll}€ bankroll")
    log_bets(value_bets)
    return value_bets


if __name__ == "__main__":
    bets = scan()
