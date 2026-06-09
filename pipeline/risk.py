"""Gestion du risque : Kelly avec incertitude, exposition journalière."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BetCandidate:
    fixture_id: int
    market: str
    side: str
    p_model: float
    odds: float
    edge: float
    variance: float = 0.0


def quarter_kelly(p_model: float, odds: float, fraction: float = 0.25) -> float:
    b = odds - 1
    if b <= 0:
        return 0.0
    q = 1 - p_model
    full_kelly = (b * p_model - q) / b
    return max(0.0, full_kelly * fraction)


def kelly_with_uncertainty(
    p_model: float,
    odds: float,
    variance: float = 0.0,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """Kelly fractionné avec shrinkage par incertitude d'estimation."""
    kelly = quarter_kelly(p_model, odds, fraction)
    if variance > 0:
        kelly *= 1.0 / (1.0 + variance)
    return min(cap, kelly)


def optimize_daily_portfolio(
    bets: list[BetCandidate],
    bankroll: float,
    max_daily_exposure: float = 0.15,
    kelly_fraction: float = 0.25,
    max_kelly: float = 0.05,
) -> list[float]:
    """
    Alloue les mises avec plafond d'exposition journalière totale.
    Approche simplifiée : Kelly individuel puis scale-down si dépassement.
    """
    stakes = []
    for bet in bets:
        k = kelly_with_uncertainty(
            bet.p_model, bet.odds, bet.variance, kelly_fraction, max_kelly
        )
        stakes.append(bankroll * k)

    total = sum(stakes)
    max_total = bankroll * max_daily_exposure
    if total > max_total and total > 0:
        scale = max_total / total
        stakes = [s * scale for s in stakes]

    return [round(s, 2) for s in stakes]


def check_drawdown(current_bankroll: float, peak_bankroll: float, max_drawdown: float = 0.35) -> bool:
    """Retourne True si drawdown acceptable."""
    if peak_bankroll <= 0:
        return True
    dd = (peak_bankroll - current_bankroll) / peak_bankroll
    return dd <= max_drawdown
