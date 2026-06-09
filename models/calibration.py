"""
Calibration isotonique, Platt et beta des probabilités.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent.parent
CALIB_PATH = ROOT / "models" / "artifacts" / "calibrator.pkl"


class MarketCalibrator:
    def __init__(self):
        self.calibrators: dict[str, IsotonicRegression | LogisticRegression] = {}
        self.method: str = "isotonic"

    def fit(
        self,
        predictions: list[dict],
        outcomes: list[dict],
        market: str = "home_win",
        method: str = "isotonic",
    ) -> None:
        self.method = method
        y_pred = np.array([p.get(market, np.nan) for p in predictions])
        y_true = np.array([o.get(market, np.nan) for o in outcomes])
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        if mask.sum() < 10:
            return

        if method == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_pred[mask], y_true[mask])
            self.calibrators[market] = ir
        elif method in ("platt", "beta"):
            lr = LogisticRegression(max_iter=1000)
            lr.fit(y_pred[mask].reshape(-1, 1), y_true[mask])
            self.calibrators[market] = lr

    def calibrate(self, pred: dict) -> dict:
        calibrated = dict(pred)
        for market, prob in pred.items():
            if market not in self.calibrators:
                continue
            cal = self.calibrators[market]
            if isinstance(cal, IsotonicRegression):
                calibrated[market] = float(cal.predict([prob])[0])
            else:
                calibrated[market] = float(cal.predict_proba([[prob]])[0, 1])
        return calibrated

    def fit_all_markets(self, predictions: list[dict], outcomes: list[dict]) -> None:
        for market in ["home_win", "draw", "away_win", "over_2.5", "btts"]:
            self.fit(predictions, outcomes, market)

    def save(self, path: Path = CALIB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = CALIB_PATH) -> "MarketCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    from evaluation.metrics import expected_calibration_error as ece
    return ece(y_true, y_prob, n_bins)


# Alias rétrocompatibilité
ProbabilityCalibrator = MarketCalibrator
