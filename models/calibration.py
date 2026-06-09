"""
Calibration isotonique des probabilités du modèle Dixon-Coles.
Compare la calibration brute vs bookmaker sur des données historiques.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


ROOT = Path(__file__).parent.parent
CALIB_PATH = ROOT / "models" / "calibrator.pkl"


class ProbabilityCalibrator:
    def __init__(self):
        self.calibrators: dict[str, IsotonicRegression] = {}

    def fit(self, predictions: list[dict], outcomes: list[dict]) -> None:
        """
        predictions : liste de {"home_win": p, "draw": p, "away_win": p, ...}
        outcomes    : liste de {"home_win": 1/0, "draw": 1/0, "away_win": 1/0}
        """
        markets = ["home_win", "draw", "away_win", "over_2.5", "btts"]
        for market in markets:
            y_pred = np.array([p.get(market, np.nan) for p in predictions])
            y_true = np.array([o.get(market, np.nan) for o in outcomes])
            mask = ~(np.isnan(y_pred) | np.isnan(y_true))
            if mask.sum() < 10:
                continue
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_pred[mask], y_true[mask])
            self.calibrators[market] = ir

    def calibrate(self, pred: dict) -> dict:
        calibrated = {}
        for market, prob in pred.items():
            if market in self.calibrators:
                calibrated[market] = float(self.calibrators[market].predict([prob])[0])
            else:
                calibrated[market] = prob
        return calibrated

    def save(self, path: Path = CALIB_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = CALIB_PATH) -> "ProbabilityCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


def plot_calibration(predictions: list[dict], outcomes: list[dict], market: str = "home_win"):
    y_pred = np.array([p.get(market, np.nan) for p in predictions])
    y_true = np.array([o.get(market, np.nan) for o in outcomes])
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))

    prob_true, prob_pred = calibration_curve(y_true[mask], y_pred[mask], n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Parfait")
    plt.plot(prob_pred, prob_true, "o-", label="Modèle")
    plt.xlabel("Probabilité prédite")
    plt.ylabel("Fréquence réelle")
    plt.title(f"Calibration — {market}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "data" / "processed" / f"calibration_{market}.png", dpi=150)
    plt.close()
    print(f"Courbe de calibration sauvegardée pour {market}.")
