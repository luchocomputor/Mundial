"""Dé-vigorisation des cotes bookmaker."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def devig_multiplicative(odds: dict[str, float]) -> dict[str, float]:
    """Normalisation multiplicative : diviser par la somme des probas implicites."""
    if not odds:
        return {}
    inv = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in inv.items()}


def devig_power(odds: dict[str, float], power: float = 1.0) -> dict[str, float]:
    """Power method : elever les probas implicites à une puissance k."""
    if not odds:
        return {}
    inv = np.array([1.0 / v for v in odds.values() if v and v > 1.0])
    keys = [k for k, v in odds.items() if v and v > 1.0]
    if len(inv) == 0:
        return {}
    # Trouver k tel que sum(p^k) = 1
    def objective(k):
        return np.sum(inv ** k) - 1.0

    try:
        k = brentq(objective, 0.5, 2.0)
    except ValueError:
        return devig_multiplicative(odds)
    probs = inv ** k
    return dict(zip(keys, probs / probs.sum()))


def devig_shin(odds: dict[str, float], max_iter: int = 100, tol: float = 1e-8) -> dict[str, float]:
    """
    Méthode de Shin (1992/1993) pour retrait du vig sur marchés multinomiaux.
    Modélise la présence d'insiders dans le marché.
    """
    if not odds:
        return {}
    keys = [k for k, v in odds.items() if v and v > 1.0]
    o = np.array([odds[k] for k in keys], dtype=float)
    n = len(o)
    if n == 0:
        return {}
    if n == 1:
        return {keys[0]: 1.0}

    inv = 1.0 / o
    overround = inv.sum()

    def shin_probs(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * inv**2 / overround**2) - z) / (2 * (1 - z) / overround)

    try:
        z = brentq(lambda z: shin_probs(z).sum() - 1.0, 0.0, 0.99, maxiter=max_iter, xtol=tol)
        probs = shin_probs(z)
    except (ValueError, RuntimeError):
        return devig_multiplicative(dict(zip(keys, o)))

    return dict(zip(keys, probs))


def devig(odds: dict[str, float], method: str = "shin") -> dict[str, float]:
    methods = {
        "multiplicative": devig_multiplicative,
        "power": devig_power,
        "shin": devig_shin,
    }
    fn = methods.get(method, devig_shin)
    return fn(odds)


def select_best_devig_method(
    historical: list[tuple[dict[str, float], str]],
    methods: list[str] | None = None,
) -> str:
    """
    Sélectionne la méthode de dé-vigorisation minimisant la log-loss.
    historical: liste de (odds_dict, outcome_key)
    """
    from sklearn.metrics import log_loss

    if methods is None:
        methods = ["multiplicative", "power", "shin"]

    best_method = "shin"
    best_loss = float("inf")

    for method in methods:
        y_true = []
        y_pred = []
        for odds, outcome in historical:
            probs = devig(odds, method)
            if outcome not in probs:
                continue
            y_true.append(list(probs.keys()).index(outcome))
            y_pred.append([probs[k] for k in probs.keys()])

        if len(y_true) < 10:
            continue
        try:
            loss = log_loss(y_true, y_pred)
            if loss < best_loss:
                best_loss = loss
                best_method = method
        except Exception:
            continue

    return best_method
