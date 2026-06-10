"""Métriques d'évaluation des modèles de prédiction football."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def ranked_probability_score(y_true: int, y_prob: np.ndarray) -> float:
    """
    RPS pour issues ordinales 1X2.
    y_true: 0=home, 1=draw, 2=away
    y_prob: [p_home, p_draw, p_away]
    """
    cum_pred = np.cumsum(y_prob)
    cum_true = np.cumsum([1 if i == y_true else 0 for i in range(3)])
    return float(np.sum((cum_pred - cum_true) ** 2) / 2)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE pour marchés binaires."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def compute_rps_batch(y_true: list[int], y_prob: list[np.ndarray]) -> float:
    return float(np.mean([ranked_probability_score(t, p) for t, p in zip(y_true, y_prob)]))


def compute_log_loss_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_prob))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


@dataclass
class MetricsReport:
    rps: float | None = None
    log_loss_1x2: float | None = None
    log_loss_binary: dict[str, float] = field(default_factory=dict)
    brier: dict[str, float] = field(default_factory=dict)
    ece: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0


def compute_all_metrics(
    predictions: list[dict],
    outcomes: list[dict],
    market: str = "1x2",
) -> MetricsReport:
    report = MetricsReport(n_samples=len(predictions))

    if market == "1x2":
        y_true = []
        y_prob = []
        for pred, actual in zip(predictions, outcomes):
            if actual.get("home_win") is not None:
                if actual["home_win"]:
                    y_true.append(0)
                elif actual.get("draw"):
                    y_true.append(1)
                else:
                    y_true.append(2)
                y_prob.append(
                    np.array([pred["home_win"], pred["draw"], pred["away_win"]])
                )
        if y_true:
            report.rps = compute_rps_batch(y_true, y_prob)

    for mkt in ["over_2.5", "btts"]:
        y_t, y_p = [], []
        for pred, actual in zip(predictions, outcomes):
            if mkt in pred and mkt in actual:
                y_t.append(actual[mkt])
                y_p.append(pred[mkt])
        if y_t:
            yt = np.array(y_t)
            yp = np.array(y_p)
            report.log_loss_binary[mkt] = compute_log_loss_binary(yt, yp)
            report.brier[mkt] = compute_brier(yt, yp)
            report.ece[mkt] = expected_calibration_error(yt, yp)

    return report
