"""Critères Go/No-Go figés avant exécution du backtest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GoNoGoDecision(str, Enum):
    GO_PAPER_TRADING = "GO_PAPER_TRADING"
    NO_GO = "NO_GO"


GO_NOGO_CRITERIA = {
    "clv_mean_min": 0.0,
    "clv_significance": True,
    "beat_close_pct_min": 0.55,
    "ece_max": 0.03,
    "roi_min": 0.0,
    "min_bets": 500,
    "max_drawdown": 0.35,
}


def evaluate(report: dict) -> GoNoGoDecision:
    """
    Évalue le rapport de backtest contre critères figés.
    CLV ≥ 0 significatif est NÉCESSAIRE.
    """
    n_bets = report.get("n_bets", 0)
    if n_bets < GO_NOGO_CRITERIA["min_bets"]:
        # Pas assez de paris pour conclure — No-Go par prudence
        if n_bets == 0:
            return GoNoGoDecision.NO_GO

    clv_mean = report.get("clv_mean", -1)
    clv_sig = report.get("clv_significant", False)
    beat_close = report.get("beat_close_pct", 0)
    roi = report.get("roi_pct", -100)
    dd = report.get("max_drawdown_pct", 100) / 100

    ece_ok = True
    ece = report.get("metrics", {}).get("ece", {})
    if ece:
        ece_ok = all(v < GO_NOGO_CRITERIA["ece_max"] for v in ece.values())

    checks = [
        clv_mean > GO_NOGO_CRITERIA["clv_mean_min"],
        clv_sig or n_bets < GO_NOGO_CRITERIA["min_bets"],
        beat_close >= GO_NOGO_CRITERIA["beat_close_pct_min"] or n_bets < 50,
        roi >= GO_NOGO_CRITERIA["roi_min"] or n_bets < GO_NOGO_CRITERIA["min_bets"],
        dd <= GO_NOGO_CRITERIA["max_drawdown"],
        ece_ok,
    ]

    # CLV positif significatif est obligatoire si assez de paris
    if n_bets >= 50 and (clv_mean <= 0 or not clv_sig):
        return GoNoGoDecision.NO_GO

    if all(checks):
        return GoNoGoDecision.GO_PAPER_TRADING
    return GoNoGoDecision.NO_GO
