"""Scan value bets BUTEUR (anytime) des matchs CDM à venir — via Betclic gRPC.

Pipeline MÉCANIQUEMENT complet : liste matchs → cotes buteur → modèle data-driven
(forme réelle + force de ligue) → de-vig → edges, résolvable par le moteur buteur.

⚠️ NE PAS PARIER EN L'ÉTAT — le SIGNAL D'EDGE EST INEXPLOITABLE. De-viger un marché
buteur anytime (non mutuellement exclusif, overround +600 % concentré sur les
outsiders) est un problème dur ; la méthode actuelle (normaliser au total modèle)
SOUS-ESTIME les favoris → faux +edge SYSTÉMATIQUE sur TOUS les top buteurs (Kane,
Mbappé, Haaland… tous à +15-22 %, ce qui est absurde : le marché ne se trompe pas
de 18 % sur Mbappé buteur). Ce n'est pas de l'edge, c'est le de-vig cassé.

Pour exploiter un jour : il faut un DE-VIG ANYTIME correct (corrigeant le biais
favori-outsider, type Shin / power-method). Tant que ce n'est pas fait, --paper
loggerait des paris sur un de-vig faux → pertes garanties.

Usage : python scan_scorers.py   (affichage seul ; --paper volontairement à éviter)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.base import MatchContext
from models.goalscorer import _nlast, load_player_rates, load_rosters, scan_goalscorer_value
from models.ratings import EloRating
from models.tm_prior import build_blended
from pipeline.fetch_betclic_grpc import list_matches, match_odds
from pipeline.teams import normalize as _norm

ROOT = Path(__file__).parent
UID = "19330064"          # x-bg-user (compte Louis) pour le détail Betclic
MIN_EDGE = 0.08           # buteurs = incertain → seuil plus strict que le 1X2
BANKROLL = 200.0


def _fixture_index() -> dict:
    """(home normalisé, away normalisé, date) → fixture_id wc_all (2 sens)."""
    wc = pd.read_parquet(ROOT / "data" / "raw" / "wc_all.parquet")
    idx = {}
    for _, r in wc.iterrows():
        if pd.isna(r.fixture_id):
            continue
        d = str(pd.Timestamp(r.date).date())
        hn, an = _norm(r.home_team), _norm(r.away_team)
        idx[(hn, an, d)] = int(r.fixture_id)
        idx.setdefault((an, hn, d), int(r.fixture_id))
    return idx


def scan(paper: bool = False) -> list[dict]:
    mdl = build_blended(EloRating.load())
    ros, rates = load_rosters(), load_player_rates()
    fidx = _fixture_index()

    matches = list_matches()
    now = pd.Timestamp.now(tz="UTC")
    upcoming = [m for m in matches if pd.Timestamp(m["date"]) > now]
    print(f"{len(matches)} matchs · {len(upcoming)} à venir · scan buteur (edge≥{MIN_EDGE:.0%})…\n")

    bets = []
    for m in upcoming:
        h, a = m["teams"]
        if h not in ros or a not in ros:
            continue
        odds = match_odds(m["match_id"], UID).get("scorers")
        if not odds:
            continue
        p = mdl.predict_outcomes(h, a, MatchContext(is_neutral=True))
        date = m["date"][:10]
        fid = fidx.get((_norm(h), _norm(a), date))
        for b in scan_goalscorer_value(h, a, p.expected_home, p.expected_away, ros, odds, MIN_EDGE, rates):
            kelly = max(0.0, b["edge"] / (b["odds"] - 1)) * 0.25      # quart-Kelly
            stake = round(min(kelly, 0.02) * BANKROLL, 2)             # cap 2 % bankroll
            bets.append({
                "fixture_id": fid, "date": date, "home_team": h, "away_team": a,
                "market": "buteur", "side": _nlast(b["player"]),
                "cote": b["odds"], "odds_taken": b["odds"],
                "p_model": b["p_model"], "p_implicit": b["p_market"],
                "p_novig": b["p_market"], "edge": b["edge"],
                "kelly_pct": round(kelly * 100, 2), "stake_eur": stake, "bankroll": BANKROLL,
                "confidence": 2, "model_version": "goalscorer", "anchor_weight": 0.0,
                "notes": f"{b['player']} buteur ({b['team']}) — modèle {b['p_model']*100:.0f}% vs marché {b['p_market']*100:.0f}%",
            })

    bets.sort(key=lambda x: -x["edge"])
    print(f"{'match':28s} {'joueur':16s} {'cote':>5s} {'modèle':>7s} {'marché':>7s} {'edge':>6s}")
    for b in bets[:25]:
        print(f"{b['home_team'][:12]+' v '+b['away_team'][:10]:28s} {b['side']:16s} "
              f"{b['cote']:5.2f} {b['p_model']*100:6.1f}% {b['p_implicit']*100:6.1f}% {b['edge']*100:+5.1f}%")
    print(f"\n{len(bets)} value bets buteur")

    if paper and bets:
        from monitoring.paper_trading import record_paper_bets
        n = record_paper_bets(bets)
        print(f"Paper : {n} nouveaux enregistrés ({len(bets) - n} déjà connus) → résolution + suivi auto")
    return bets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    scan(paper=args.paper)
