"""
Scanner de value bets CDM 2026.
Usage : python scan_now.py [--paper]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import load_config
from pipeline.features import load_raw_matches
from pipeline.model_loader import load_production_model
from pipeline.placeholder import filter_real_teams
from pipeline.value_detector import log_bets, scan_value_bets

ROOT = Path(__file__).parent


def load_venue_map() -> dict:
    p = ROOT / "data" / "raw" / "venues.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df["venue_id"], df["name"]))


def _team_match_counts() -> dict[str, int]:
    try:
        df = load_raw_matches()
        finished = df[df["status"] == "finished"]
        counts: dict[str, int] = {}
        for _, row in finished.iterrows():
            counts[row["home_team"]] = counts.get(row["home_team"], 0) + 1
            counts[row["away_team"]] = counts.get(row["away_team"], 0) + 1
        return counts
    except Exception:
        return {}


def scan(paper: bool = False):
    cfg = load_config()
    bankroll = cfg.bankroll_initial

    if cfg.model.production_mode == "paper_only":
        print("⚠ Mode PAPER ONLY — modèle non validé, observation uniquement\n")

    print("Chargement modèle Elo + données...")
    model = load_production_model(prefer_elo=True)
    venue_map = load_venue_map()
    team_counts = _team_match_counts()

    wc_path = ROOT / "data" / "raw" / "wc_all.parquet"
    all_path = ROOT / "data" / "raw" / "all_matches.parquet"
    path = wc_path if wc_path.exists() else all_path
    if not path.exists():
        print(f"Données manquantes : {path}. Lance make fetch.")
        return []

    wc = pd.read_parquet(path)
    wc2026 = wc[wc["date"].dt.year == cfg.seasons.get("live", 2026)]
    matches = filter_real_teams(wc2026).sort_values("date")

    odds_path = ROOT / "data" / "raw" / "odds_wc2026.json"
    odds = {}
    if odds_path.exists():
        odds_raw = json.loads(odds_path.read_text())
        odds = {int(k): v for k, v in odds_raw.items()}

    bzzo_map = {}
    preds_path = ROOT / "data" / "raw" / "predictions_wc2026.parquet"
    if preds_path.exists():
        bzzo_df = pd.read_parquet(preds_path)
        bzzo_map = {int(row["event_id"]): row.to_dict() for _, row in bzzo_df.iterrows()}

    match_list = []
    for _, row in matches.iterrows():
        fid = int(row["fixture_id"])
        if fid not in odds:
            continue
        venue = venue_map.get(row.get("venue_id"), row.get("venue", ""))
        match_list.append({
            "fixture_id": fid,
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "venue": venue,
            "is_friendly": bool(row.get("is_friendly", False)),
            "is_neutral": bool(row.get("is_neutral", False)),
            "group": row.get("group_name", ""),
        })

    print(f"Scan de {len(match_list)} matchs avec cotes...\n")

    value_bets, guard_stats = scan_value_bets(
        match_list, model, odds, cfg, bankroll, bzzo_map,
        team_match_counts=team_counts,
    )

    print(
        f"Garde-fous : {guard_stats.accepted} acceptés, "
        f"{guard_stats.rejected_divergence} rejetés (divergence > {cfg.model.max_market_divergence:.0%})"
    )

    no_odds_count = len(matches) - len(match_list)
    if no_odds_count:
        print(f"Matchs sans cotes : {no_odds_count}\n")

    if not value_bets:
        print("Aucun value bet plausible détecté.")
        return value_bets

    print(f"\n{'Match':<32} {'Marché':<18} {'Cote':>5} {'Modèle':>7} {'Edge':>6} {'Mise':>6}")
    print("─" * 85)
    for b in value_bets:
        match_str = f"{b['home_team']} vs {b['away_team']}"[:31]
        print(
            f"{match_str:<32} {b['market']} {b['side']:<10} {b['cote']:5.2f} "
            f"{b['p_model']:7.1%} {b['edge']:6.1%} {b['stake_eur']:5.2f}€"
        )

    total_stake = sum(b["stake_eur"] for b in value_bets)
    print(f"\n{len(value_bets)} value bets | Mise totale : {total_stake:.2f}€ / {bankroll}€")

    if paper or cfg.model.production_mode == "paper_only":
        from monitoring.paper_trading import record_paper_bets
        record_paper_bets(value_bets)
    else:
        log_bets(value_bets)

    return value_bets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", help="Mode paper trading")
    args = parser.parse_args()
    scan(paper=args.paper or True)
