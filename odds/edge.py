"""Calcul d'edge sur probabilités no-vig."""

from __future__ import annotations

from odds.devig import devig


def compute_edge_novig(
    p_model: float,
    odds_decimal: float,
    method: str = "shin",
    market_odds: dict[str, float] | None = None,
) -> float:
    """
    Edge = p_model - p_novig.
    Si market_odds fourni (toutes les cotes du marché), dé-vigorise proprement.
    Sinon fallback sur 1/odds brut (conservateur).
    """
    if odds_decimal <= 1.0:
        return -1.0

    if market_odds:
        novig = devig(market_odds, method)
        # Trouver la clé correspondant à cette cote
        for k, v in market_odds.items():
            if abs(v - odds_decimal) < 0.001 and k in novig:
                return p_model - novig[k]
        # Fallback : utiliser la proba implicite normalisée
        p_implied = 1.0 / odds_decimal
        total_implied = sum(1.0 / v for v in market_odds.values() if v > 1.0)
        p_novig = p_implied / total_implied if total_implied > 0 else p_implied
        return p_model - p_novig

    return p_model - (1.0 / odds_decimal)


def compute_edge_vs_close(p_model: float, p_close_novig: float) -> float:
    """Edge vs cote de clôture dé-vigorisée."""
    return p_model - p_close_novig
