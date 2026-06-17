"""
Génère un rapport de probabilités buteur par match et par joueur.
Usage: python goalscorer_report.py [--date 2026-06-11] [--team France]
Comparer manuellement avec les cotes Betclic pour trouver les value bets.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from models.goalscorer import compute_player_probas, load_rosters
from models.base import MatchContext
from pipeline.model_loader import load_production_model

# Mapping noms bzzoiro → noms ESPN roster
TEAM_NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Cabo Verde": "Cape Verde",
    "South Korea": "South Korea",
    "USA": "United States",
    "Bosnia": "Bosnia-Herzegovina",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def load_matches(date_filter: str | None = None) -> pd.DataFrame:
    wc = pd.read_parquet("data/raw/wc_all.parquet")
    wc26 = wc[wc["date"].dt.year == 2026].copy()

    def is_real_team(name):
        return bool(re.match(r"^[A-Za-zÀ-ÿ\s\-\'\.']+$", name)) and len(name) > 2

    real = wc26[wc26["home_team"].apply(is_real_team) & wc26["away_team"].apply(is_real_team)]
    real = real.sort_values("date")

    if date_filter:
        real = real[real["date"].dt.strftime("%Y-%m-%d") == date_filter]

    return real


def get_match_lambdas(model, home: str, away: str) -> tuple[float, float]:
    try:
        pred = model.predict_outcomes(home, away, MatchContext(is_neutral=True))
        return pred.expected_home, pred.expected_away
    except Exception:
        return 1.2, 0.9


def print_match_report(
    home: str,
    away: str,
    mu: float,
    nu: float,
    rosters: dict,
    n_players: int = 8,
) -> None:
    home_esp = normalize_team(home)
    away_esp = normalize_team(away)

    h_probas = compute_player_probas(home_esp, mu, rosters)
    a_probas = compute_player_probas(away_esp, nu, rosters)

    print(f"\n{'─'*60}")
    print(f"  {home} ({mu:.2f} xG)  vs  {away} ({nu:.2f} xG)")
    print(f"{'─'*60}")

    col_w = 28
    print(f"  {'DOMICILE':<{col_w}}  {'EXTÉRIEUR':<{col_w}}")
    print(f"  {'-'*col_w}  {'-'*col_w}")

    max_rows = max(len(h_probas[:n_players]), len(a_probas[:n_players]))
    for i in range(max_rows):
        h = h_probas[i] if i < len(h_probas) else None
        a = a_probas[i] if i < len(a_probas) else None

        h_str = f"{h.jersey:>2}. {h.name:<20} {h.p_anytime*100:5.1f}%" if h else ""
        a_str = f"{a.jersey:>2}. {a.name:<20} {a.p_anytime*100:5.1f}%" if a else ""
        print(f"  {h_str:<{col_w+2}}  {a_str}")

    print(f"\n  Mise en face des cotes Betclic :")
    print(f"  Cote 2.5 → seuil model = {1/2.5*100:.0f}% | cote 3.0 → {1/3.0*100:.0f}% | cote 4.0 → {1/4.0*100:.0f}%")
    print(f"  Edge ≥5% : model doit battre la cote no-vig d'au moins 5pp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--team", default=None, help="Filtrer par équipe")
    parser.add_argument("--top", type=int, default=8, help="Nombre de joueurs à afficher par équipe")
    args = parser.parse_args()

    rosters = load_rosters()

    try:
        model = load_production_model()
    except Exception:
        print("Modèle non trouvé, utilisation de lambdas par défaut.")
        model = None

    matches = load_matches(args.date)
    if args.team:
        team_filter = args.team.lower()
        matches = matches[
            matches["home_team"].str.lower().str.contains(team_filter)
            | matches["away_team"].str.lower().str.contains(team_filter)
        ]

    if matches.empty:
        print(f"Aucun match trouvé pour {'date=' + args.date if args.date else 'toute la phase de groupes'}.")
        return

    date_label = args.date or "Phase de groupes complète"
    print(f"\n{'='*60}")
    print(f"  RAPPORT BUTEURS — {date_label}")
    print(f"  {len(matches)} match(s)")
    print(f"{'='*60}")
    print()
    print("  Légende : probabilité P(marquer au moins 1 but) par joueur")
    print("  Comparer avec les cotes Betclic 'Buteur à tout moment'")

    for _, row in matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        if model:
            mu, nu = get_match_lambdas(model, home, away)
        else:
            mu, nu = 1.3, 1.0
        print_match_report(home, away, mu, nu, rosters, n_players=args.top)

    print(f"\n{'='*60}")
    print("  Workflow : noter les cotes Betclic en face des joueurs,")
    print("  calculer edge = p_model - 1/cote_betclic")
    print("  Kelly = max(0, (edge × cote) / (cote - 1)) × 0.25 × bankroll")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
