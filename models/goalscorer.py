"""
Modèle buteur DATA-DRIVEN : distribue les expected goals d'une équipe entre ses
joueurs selon leur taux de buts RÉEL (saison en cours, Transfermarkt), pas des
tiers codés en dur.

Entrée : mu (expected goals équipe), roster, taux de buts par joueur.
Sortie : P(joueur marque anytime) par joueur.

Approche :
- poids joueur = buts/match réels (player_form, toutes compétitions, lissé)
  + repli position pour les joueurs sans données.
- normalisation dans l'effectif → distribue mu.
- P(anytime) = 1 - exp(-mu * poids_normalisé).

Le scan buteur DE-VIGE le marché (overround buteur ~+600 %) avant l'edge.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
ROSTERS_PATH = ROOT / "data" / "raw" / "wc2026_rosters.json"
FORM_PATH = ROOT / "data" / "raw" / "transfermarkt" / "wc2026_player_form.csv"

# Repli pour les joueurs SANS données de forme (poids buts/match faute de mieux).
POSITION_FALLBACK = {"F": 0.18, "M": 0.06, "D": 0.012, "G": 0.001}


def _nfull(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", " ").replace("-", " ").split())


def _nlast(s: str) -> str:
    parts = [p for p in _nfull(s).split() if len(p) > 1]
    return parts[-1] if parts else _nfull(s)

# Stars tier 1 : ~3x boost (top scorers WC 2026)
import pandas as pd


def load_player_rates(path: Path = FORM_PATH) -> tuple[dict, dict]:
    """Agrège player_form (toutes compétitions) → buts par match (lissés) par
    joueur. Retourne (par_nom_complet, par_nom_de_famille) pour le matching."""
    if not path.exists():
        return {}, {}
    pf = pd.read_csv(path)
    agg = pf.groupby("player").agg(
        goals=("goals", "sum"), games=("games", "sum"), pos=("position", "first")
    ).reset_index()
    by_full: dict[str, float] = {}
    by_last: dict[str, tuple[float, float]] = {}  # last -> (games, gpg) le + joué
    for _, r in agg.iterrows():
        games = float(r.games or 0)
        gpg = float(r.goals or 0) / (games + 3.0)  # lissage bayésien vers 0
        by_full[_nfull(r.player)] = gpg
        last = _nlast(r.player)
        if last not in by_last or games > by_last[last][0]:
            by_last[last] = (games, gpg)
    return by_full, {k: v[1] for k, v in by_last.items()}


def player_weight(name: str, position: str, rates: tuple[dict, dict]) -> float:
    """Poids buts/match d'un joueur : forme réelle, sinon repli position."""
    by_full, by_last = rates
    w = by_full.get(_nfull(name))
    if w is None:
        w = by_last.get(_nlast(name))
    if w is None:
        w = POSITION_FALLBACK.get(position, 0.05)
    return max(w, 0.0005)


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
    rates: tuple[dict, dict] | None = None,
) -> list[PlayerProba]:
    """P(anytime scorer) par joueur : on distribue mu selon les poids buts/match
    réels (forme saison), puis P(anytime) = 1 - exp(-mu * poids_normalisé)."""
    squad = rosters.get(team, [])
    if not squad:
        return []
    if rates is None:
        rates = load_player_rates()

    raw_weights = [player_weight(p["name"], p.get("position", ""), rates) for p in squad]
    total = sum(raw_weights) or 1.0
    norm_weights = [w / total for w in raw_weights]

    results = []
    for p, w in zip(squad, norm_weights):
        lambda_player = mu * w
        results.append(PlayerProba(
            name=p["name"], team=team, position=p.get("position", ""),
            jersey=p.get("jersey", ""), p_anytime=1.0 - math.exp(-lambda_player),
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
    rates: tuple[dict, dict] | None = None,
) -> list[dict]:
    """Value bets anytime scorer, après DE-VIG du marché. bookmaker_odds =
    {player_name: cote_décimale}. L'overround buteur est énorme (~+600 % : ~46
    joueurs listés) → on normalise les implicites du book au total de buteurs
    attendu du modèle avant de comparer, sinon tous les edges sont faux-négatifs."""
    if rates is None:
        rates = load_player_rates()

    model: dict[str, dict] = {}
    for team, team_mu in [(home, mu), (away, nu)]:
        for pp in compute_player_probas(team, team_mu, rosters, rates):
            model[pp.name] = {"p": pp.p_anytime, "team": pp.team, "position": pp.position}

    # joueurs présents dans le book ET le modèle
    common = [n for n, o in bookmaker_odds.items() if o and o > 1 and n in model]
    if not common:
        return []
    raw_imp = {n: 1.0 / bookmaker_odds[n] for n in common}
    sum_imp = sum(raw_imp.values())
    sum_mod = sum(model[n]["p"] for n in common)
    scale = (sum_mod / sum_imp) if sum_imp else 0.0  # de-vig → total marché = total modèle

    bets = []
    for n in common:
        p_fair = raw_imp[n] * scale          # proba marché de-viggée (sans marge)
        edge = model[n]["p"] - p_fair
        if edge >= min_edge:
            bets.append({
                "player": n, "team": model[n]["team"], "position": model[n]["position"],
                "p_model": round(model[n]["p"], 4), "p_market": round(p_fair, 4),
                "p_implicit_raw": round(raw_imp[n], 4), "edge": round(edge, 4),
                "odds": bookmaker_odds[n],
            })
    bets.sort(key=lambda x: x["edge"], reverse=True)
    return bets
