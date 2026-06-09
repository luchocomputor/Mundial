"""Modèle basé xG plutôt que buts réels."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import MatchContext, Prediction
from models.dixon_coles import DixonColesModel

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "artifacts" / "xg_model.pkl"


class XGModel:
    def __init__(self, decay: float = 0.0065):
        self.decay = decay
        self._dc = DixonColesModel(decay=decay)
        self._fitted = False

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        df = matches.copy()
        has_xg = df["xg_home"].notna() & df["xg_away"].notna()
        if has_xg.sum() < 20:
            # Fallback sur buts réels si xG insuffisant
            self._dc.fit(matches, reference_date)
        else:
            xg_df = df[has_xg].copy()
            xg_df = xg_df.rename(columns={"xg_home": "home_goals", "xg_away": "away_goals"})
            xg_df["home_goals"] = xg_df["home_goals"].round().astype(int).clip(0, 10)
            xg_df["away_goals"] = xg_df["away_goals"].round().astype(int).clip(0, 10)
            self._dc.fit(xg_df, reference_date)
        self._fitted = True

    def predict_outcomes(
        self, home: str, away: str, context: MatchContext | None = None
    ) -> Prediction:
        pred = self._dc.predict_outcomes(home, away, context=context)
        if isinstance(pred, Prediction):
            pred.source = "xg_model"
            return pred
        d = pred if isinstance(pred, dict) else pred.to_dict()
        return Prediction.from_dict({**d, "source": "xg_model"})

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "XGModel":
        with open(path, "rb") as f:
            return pickle.load(f)
