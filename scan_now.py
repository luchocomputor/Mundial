"""
Scanner de value bets CDM 2026.
Cotes sources : Betclic FR (1X2) + Pinnacle (over/under).
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
from pipeline.value_detector import compute_edge, kelly_size, log_bets

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


def normalize_name(name: str) -> str:
    """Normalise les noms d'équipes pour matcher entre sources."""
    mapping = {
        "Czech Republic": "Czechia",
        "Turkey": "Türkiye",
        "Ivory Coast": "Côte d'Ivoire",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
        "Bosnia & Herzegovina": "Bosnia & Herzegovina",
        "Republic of Ireland": "Ireland",
        "DR Congo": "DR Congo",
        "Democratic Republic of Congo": "DR Congo",
        "South Korea": "South Korea",
        "Korea Republic": "South Korea",
    }
    return mapping.get(name, name)


def get_model_prob(preds: dict, market: str, side: str) -> float | None:
    if market == "1X2":
        return {"home": preds["home_win"], "draw": preds["draw"], "away": preds["away_win"]}.get(side)
    if market == "over_2.5":
        if side == "over":  return preds.get("over_2.5")
        if side == "under": return 1 - preds.get("over_2.5", 0)
    return None


def scan(show_all: bool = False):
    cfg = load_config()
    threshold = cfg["model"]["min_edge_threshold"]
    kelly_fraction = cfg["model"]["kelly_fraction"]
    max_kelly = cfg["model"]["max_kelly_bet"]
    dc_weight = cfg.get("blend", {}).get("dc_weight", 0.3)
    bankroll = cfg["bankroll"]["initial"]

    print("Chargement modèle + données...")
    model = DixonColesModel.load()
    venue_map = load_venue_map()

    # Cotes Betclic + Pinnacle (indexées par (home_norm, away_norm))
    betclic_path = ROOT / "data" / "raw" / "odds_betclic.json"
    if not betclic_path.exists():
        print("Cotes Betclic introuvables. Lance: python3 pipeline/fetch_betclic.py")
        return []
    betclic_data = json.loads(betclic_path.read_text())
    odds_map = {}
    for m in betclic_data:
        key = (normalize_name(m["home_team"]), normalize_name(m["away_team"]))
        odds_map[key] = m

    # Prédictions bzzoiro (second signal)
    bzzo_path = ROOT / "data" / "raw" / "predictions_wc2026.parquet"
    bzzo_map = {}
    if bzzo_path.exists():
        bzzo_df = pd.read_parquet(bzzo_path)
        for _, r in bzzo_df.iterrows():
            key = (normalize_name(r["home_team"]), normalize_name(r["away_team"]))
            bzzo_map[key] = r.to_dict()

    # Matchs CDM 2026 (phase de groupes, équipes connues)
    wc = pd.read_parquet(ROOT / "data" / "raw" / "wc_all.parquet")
    wc2026 = wc[wc["date"].dt.year == 2026]
    matches = wc2026[~wc2026["home_team"].str.match(r"^[W|L|R|Q|H|G|1|2|3]")].sort_values("date")

    print(f"Scan de {len(matches)} matchs | {len(odds_map)} avec cotes Betclic/Pinnacle\n")

    value_bets = []
    no_odds = []

    for _, row in matches.iterrows():
        home_raw, away_raw = row["home_team"], row["away_team"]
        home = normalize_name(home_raw)
        away = normalize_name(away_raw)
        venue = venue_map.get(row.get("venue_id"), "")
        altitude_adj = get_altitude_adjustment(venue)

        match_odds = odds_map.get((home, away))
        if not match_odds:
            no_odds.append(f"{home} vs {away}")
            continue

        try:
            dc_preds = model.predict_outcomes(home, away, altitude_adj=altitude_adj)
        except Exception:
            try:
                dc_preds = model.predict_outcomes(home_raw, away_raw, altitude_adj=altitude_adj)
            except Exception:
                continue

        bzzo_row = bzzo_map.get((home, away))
        preds = blend_predictions(dc_preds, bzzo_row, dc_weight=dc_weight)

        date_str = row["date"].strftime("%d/%m %H:%M") if hasattr(row["date"], "strftime") else str(row["date"])[:16]

        # Marchés à scanner
        markets_to_check = []

        # 1X2 — cotes Betclic
        bc = match_odds.get("betclic", {})
        if bc.get("home_win"):
            markets_to_check.append(("1X2", "home", bc["home_win"], "betclic"))
        if bc.get("draw"):
            markets_to_check.append(("1X2", "draw", bc["draw"], "betclic"))
        if bc.get("away_win"):
            markets_to_check.append(("1X2", "away", bc["away_win"], "betclic"))

        # Over/Under — Pinnacle (Betclic ne l'expose pas via cette API)
        pin = match_odds.get("pinnacle", {})
        if pin.get("over_2.5"):
            markets_to_check.append(("over_2.5", "over", pin["over_2.5"], "pinnacle"))
        if pin.get("under_2.5"):
            markets_to_check.append(("over_2.5", "under", pin["under_2.5"], "pinnacle"))

        for market, side, cote, source in markets_to_check:
            if not cote or cote <= 1.0:
                continue

            p_model = get_model_prob(preds, market, side)
            if p_model is None:
                continue

            edge = compute_edge(p_model, cote)
            if edge < threshold:
                continue

            # Filtres de sanité
            if edge > 0.25:
                continue
            if cote > 10.0:
                continue

            # Consensus DC / bzzoiro obligatoire (±15%)
            if bzzo_row is not None:
                bzzo_probs = {
                    ("1X2",    "home"):   bzzo_row.get("prob_home", -1),
                    ("1X2",    "draw"):   bzzo_row.get("prob_draw", -1),
                    ("1X2",    "away"):   bzzo_row.get("prob_away", -1),
                    ("over_2.5","over"):  bzzo_row.get("prob_over_25", -1),
                    ("over_2.5","under"): 1 - bzzo_row.get("prob_over_25", -1) if bzzo_row.get("prob_over_25") is not None else -1,
                }
                p_bzzo = bzzo_probs.get((market, side), -1)
                p_dc   = get_model_prob({"home_win": dc_preds["home_win"], "draw": dc_preds["draw"],
                                         "away_win": dc_preds["away_win"], "over_2.5": dc_preds["over_2.5"],
                                         "btts": dc_preds.get("btts", 0)}, market, side)
                if p_bzzo >= 0 and p_dc is not None and abs(p_dc - p_bzzo) > 0.15:
                    continue

            kelly = min(kelly_size(p_model, cote, kelly_fraction), max_kelly)
            stake = round(bankroll * kelly, 2)

            value_bets.append({
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "market": market,
                "side": side,
                "source": source,
                "p_model": round(p_model, 4),
                "p_implicit": round(1 / cote, 4),
                "edge": round(edge, 4),
                "cote": cote,
                "kelly_pct": round(kelly * 100, 2),
                "stake_eur": stake,
                "bankroll": bankroll,
                "p_dc": round(get_model_prob(dc_preds, market, side) or 0, 4),
                "p_bzzo": round(get_model_prob({"home_win": bzzo_row.get("prob_home",0),
                                                "draw": bzzo_row.get("prob_draw",0),
                                                "away_win": bzzo_row.get("prob_away",0),
                                                "over_2.5": bzzo_row.get("prob_over_25",0),
                                                "btts": bzzo_row.get("prob_btts",0)},
                                               market, side) or 0, 4) if bzzo_row else 0,
                "altitude_adj": altitude_adj,
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)

    if no_odds:
        print(f"Matchs sans cotes ({len(no_odds)}): {', '.join(no_odds[:5])}{'...' if len(no_odds)>5 else ''}\n")

    if not value_bets:
        print("Aucun value bet détecté.")
        return value_bets

    print(f"{'Match':<30} {'Date':<12} {'Src':<8} {'Marché':<14} {'Cote':>5} {'DC':>6} {'Bzzo':>6} {'Blend':>6} {'Impl.':>6} {'Edge':>6} {'Mise':>6}")
    print("─" * 110)
    for b in value_bets:
        match_str = f"{b['home_team']} vs {b['away_team']}"[:29]
        market_str = f"{b['market']} {b['side']}"
        print(f"{match_str:<30} {b['date']:<12} {b['source']:<8} {market_str:<14} {b['cote']:5.2f} "
              f"{b['p_dc']:6.1%} {b['p_bzzo']:6.1%} {b['p_model']:6.1%} {b['p_implicit']:6.1%} "
              f"{b['edge']:6.1%} {b['stake_eur']:5.2f}€")

    total_stake = sum(b["stake_eur"] for b in value_bets)
    print(f"\n{len(value_bets)} value bets | Mise totale: {total_stake:.2f}€ / {bankroll}€")
    log_bets(value_bets)
    return value_bets


if __name__ == "__main__":
    bets = scan()
