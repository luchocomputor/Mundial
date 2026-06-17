"""
Modèle buteur : distribue les expected goals d'une équipe entre ses joueurs.
Entrée : mu (expected goals équipe), roster (liste joueurs + positions)
Sortie : P(joueur marque anytime) pour chaque joueur du squad

Approche :
- Poids de base par position (F > M > D > G)
- Multiplicateur tier pour les stars connues
- Normalisation → somme des poids = 1
- P(anytime) = 1 - exp(-mu * player_weight)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
ROSTERS_PATH = ROOT / "data" / "raw" / "wc2026_rosters.json"

# Poids de base par position (avant normalisation)
POSITION_BASE_WEIGHT = {
    "F": 3.5,
    "M": 1.0,
    "D": 0.25,
    "G": 0.02,
}

# Stars tier 1 : ~3x boost (top scorers WC 2026)
TIER1 = {
    "Lionel Messi",
    "Kylian Mbappé",
    "Cristiano Ronaldo",
    "Neymar",
    "Vinícius Júnior",
    "Mohamed Salah",
    "Harry Kane",
    "Romelu Lukaku",
    "Erling Haaland",
}

# Stars tier 2 : ~2x boost
TIER2 = {
    # Europe élite
    "Jude Bellingham",
    "Marcus Thuram",
    "Ousmane Dembélé",
    "Kai Havertz",
    "Leroy Sané",
    "Raphinha",
    "Gavi",
    "Pedri",
    "Alvaro Morata",
    "Gonçalo Ramos",
    "Gonçalo Guedes",
    "Rafael Leão",
    "João Félix",
    "Antoine Griezmann",
    "Bradley Barcola",
    "Lamine Yamal",
    "Ferran Torres",
    "Dušan Vlahović",
    "Bukayo Saka",
    "Phil Foden",
    "Marcus Rashford",
    "Gabriel Martinelli",
    "Endrick null",
    "Matheus Cunha",
    # Afrique du Nord / Afrique
    "Riyad Mahrez",
    "Ayoub El Kaabi",
    # Asie / CONCACAF
    "Son Heung-Min",
    "Takefusa Kubo",
    "Santiago Gimenez",
    "Luis Díaz",
    "Darwin Núñez",
    "Miguel Almirón",
    "Mehdi Taremi",
}

# Stars tier 3 : ~1.5x boost
TIER3 = {
    # France
    "Jean-Philippe Mateta",
    "Désiré Doué",
    # Germany
    "Maximilian Beier",
    "Deniz Undav",
    "Jamal Musiala",
    "Florian Wirtz",
    "Serge Gnabry",
    # Portugal
    "Pedro Neto",
    "Francisco Conceição",
    "Francisco Trincão",
    "Diogo Jota",
    # Netherlands
    "Memphis Depay",
    "Donyell Malen",
    # Senegal
    "Sadio Mané",
    "Ismaïla Sarr",
    "Nicolas Jackson",
    # Belgium
    "Dodi Lukebakio",
    # Italy (hors WC mais au cas)
    "Ciro Immobile",
    # Spain
    "Bernardo Silva",
    "Bruno Fernandes",
    # South America
    "Raúl Jiménez",
    "Cucho Hernández",
    "Jhon Córdoba",
    "Enner Valencia",
    "Antonio Sanabria",
    # CONCACAF
    "Mathew Leckie",
    # Africa
    "Ayoub El Kaabi",
    "Soufiane Rahimi",
    "Jordan Ayew",
    "Iñaki Williams",
    "Kamaldeen Sulemana",
    "Saleh Al-Shehri",
    "Feras Al-Brikan",
    # Europe
    "Andrej Kramaric",
    "Ivan Perisic",
    "Breel Embolo",
    "Marko Arnautovic",
    "Kenan Yildiz",
    "Kerem Aktürkoglu",
    "Junya Ito",
    "Ayase Ueda",
    "Amine Gouiri",
    # Norway
    "Alexander Sørloth",
    "Jørgen Strand Larsen",
    # New Zealand
    "Chris Wood",
    # Sweden
    "Viktor Gyökeres",
    "Alexander Isak",
    # Canada
    "Jonathan David",
    # Uruguay
    "Darwin Núñez",
    # Belgium
    "Leandro Trossard",
    "Dodi Lukebakio",
}


def _tier_multiplier(name: str) -> float:
    if name in TIER1:
        return 3.0
    if name in TIER2:
        return 2.0
    if name in TIER3:
        return 1.5
    return 1.0


@dataclass
class PlayerProba:
    name: str
    team: str
    position: str
    jersey: str
    p_anytime: float
    expected_goals: float


def load_rosters(path: Path = ROSTERS_PATH) -> dict[str, list[dict]]:
    """Returns {team_name: [player_dicts]}"""
    players = json.loads(path.read_text())
    rosters: dict[str, list[dict]] = {}
    for p in players:
        rosters.setdefault(p["team_name"], []).append(p)
    return rosters


def compute_player_probas(
    team: str,
    mu: float,
    rosters: dict[str, list[dict]],
) -> list[PlayerProba]:
    """
    Pour une équipe avec expected goals = mu, retourne P(anytime scorer) par joueur.
    """
    squad = rosters.get(team, [])
    if not squad:
        return []

    # Calcul des poids bruts
    raw_weights = []
    for p in squad:
        base = POSITION_BASE_WEIGHT.get(p["position"], 0.5)
        mult = _tier_multiplier(p["name"])
        raw_weights.append(base * mult)

    total = sum(raw_weights) or 1.0
    norm_weights = [w / total for w in raw_weights]

    results = []
    for p, w in zip(squad, norm_weights):
        lambda_player = mu * w
        p_anytime = 1.0 - math.exp(-lambda_player)
        results.append(PlayerProba(
            name=p["name"],
            team=team,
            position=p["position"],
            jersey=p.get("jersey", ""),
            p_anytime=p_anytime,
            expected_goals=lambda_player,
        ))

    results.sort(key=lambda x: x.p_anytime, reverse=True)
    return results


def scan_goalscorer_value(
    home: str,
    away: str,
    mu: float,
    nu: float,
    rosters: dict[str, list[dict]],
    bookmaker_odds: dict[str, float],
    min_edge: float = 0.05,
) -> list[dict]:
    """
    bookmaker_odds: {player_name: decimal_odds_anytime_scorer}
    Retourne les value bets anytime scorer (edge >= min_edge).
    """
    bets = []

    for team, team_mu in [(home, mu), (away, nu)]:
        probas = compute_player_probas(team, team_mu, rosters)
        for pp in probas:
            bk_odds = bookmaker_odds.get(pp.name)
            if not bk_odds or bk_odds <= 1:
                continue
            p_implicit = 1.0 / bk_odds
            edge = pp.p_anytime - p_implicit
            if edge >= min_edge:
                bets.append({
                    "player": pp.name,
                    "team": pp.team,
                    "position": pp.position,
                    "p_model": round(pp.p_anytime, 4),
                    "p_implicit": round(p_implicit, 4),
                    "edge": round(edge, 4),
                    "odds": bk_odds,
                })

    bets.sort(key=lambda x: x["edge"], reverse=True)
    return bets
