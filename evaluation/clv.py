"""Closing Line Value — KPI souverain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def clv(odds_taken: float, odds_close: float) -> float:
    """
    CLV en format décimal : odds_taken / odds_close - 1.
    Positif = on a obtenu une meilleure cote que la clôture.
    """
    if odds_close <= 1.0 or odds_taken <= 1.0:
        return 0.0
    return odds_taken / odds_close - 1.0


def clv_novig(odds_taken: float, p_close_novig: float) -> float:
    """CLV basé sur probabilité no-vig de clôture."""
    if p_close_novig <= 0:
        return 0.0
    fair_odds = 1.0 / p_close_novig
    return clv(odds_taken, fair_odds)


@dataclass
class CLVReport:
    mean_clv: float
    median_clv: float
    beat_close_pct: float
    n_bets: int
    ci_lower: float
    ci_upper: float
    significant: bool


def bootstrap_clv_significance(
    clv_series: pd.Series, n_bootstrap: int = 10000, alpha: float = 0.05
) -> CLVReport:
    arr = clv_series.dropna().values
    n = len(arr)
    if n == 0:
        return CLVReport(0, 0, 0, 0, 0, 0, False)

    mean_clv = float(arr.mean())
    beat_pct = float((arr > 0).mean())

    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    significant = ci_lower > 0

    return CLVReport(
        mean_clv=mean_clv,
        median_clv=float(np.median(arr)),
        beat_close_pct=beat_pct,
        n_bets=n,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
    )
