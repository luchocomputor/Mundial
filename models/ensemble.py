"""Stacking ensemble — remplace le blend 60/40 arbitraire."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.base import MatchContext, MatchPredictor, Prediction

ROOT = Path(__file__).parent.parent
ARTIFACT_PATH = ROOT / "models" / "artifacts" / "ensemble.pkl"
META_PATH = ROOT / "models" / "artifacts" / "ensemble_meta.json"


class StackingEnsemble:
    def __init__(self, base_models: list | None = None):
        self.base_models: list = base_models or []
        self.meta_model: LogisticRegression | None = None
        self.model_names: list[str] = []
        self._fitted = False

    def add_model(self, model, name: str) -> None:
        self.base_models.append((name, model))
        self.model_names.append(name)

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> None:
        for name, model in self.base_models:
            model.fit(matches, reference_date)
        self._fitted = True

    def fit_meta(
        self,
        oof_predictions: pd.DataFrame,
        outcomes: pd.DataFrame,
    ) -> None:
        """
        oof_predictions : colonnes {model_name}_home, {model_name}_draw, {model_name}_away
        outcomes : colonnes home_win, draw, away_win (0/1)
        """
        feature_cols = [c for c in oof_predictions.columns if c.endswith(("_home", "_draw", "_away"))]
        if not feature_cols:
            return

        X = oof_predictions[feature_cols].values
        y = outcomes[["home_win", "draw", "away_win"]].values.argmax(axis=1)

        self.meta_model = LogisticRegression(max_iter=1000, multi_class="multinomial")
        self.meta_model.fit(X, y)

        meta = {
            "feature_cols": feature_cols,
            "coef": self.meta_model.coef_.tolist(),
            "intercept": self.meta_model.intercept_.tolist(),
        }
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, indent=2))

    def _predict_base(self, model, home: str, away: str, context: MatchContext | None):
        """Appelle predict_outcomes en gérant les signatures différentes."""
        try:
            return model.predict_outcomes(home, away, context=context)
        except TypeError:
            pass
        if context:
            try:
                return model.predict_outcomes(
                    home, away,
                    altitude_adj=context.altitude_adj,
                    is_neutral=context.is_neutral,
                )
            except TypeError:
                pass
        return model.predict_outcomes(home, away)

    def predict_outcomes(
        self,
        home: str,
        away: str,
        context: MatchContext | None = None,
        bzzoiro_features: dict | None = None,
    ) -> Prediction:
        if not self.base_models:
            raise RuntimeError("Aucun modèle de base.")

        preds = []
        feature_vec = []

        for name, model in self.base_models:
            p = self._predict_base(model, home, away, context)
            if isinstance(p, dict):
                p = Prediction.from_dict(p)
            preds.append(p)
            feature_vec.extend([p.home_win, p.draw, p.away_win])

        if bzzoiro_features:
            feature_vec.extend([
                bzzoiro_features.get("prob_home", 0),
                bzzoiro_features.get("prob_draw", 0),
                bzzoiro_features.get("prob_away", 0),
            ])

        if self.meta_model is not None:
            proba = self.meta_model.predict_proba([feature_vec[: len(self.meta_model.coef_[0])]])[0]
            return Prediction(
                home_win=float(proba[0]),
                draw=float(proba[1]),
                away_win=float(proba[2]),
                over_2_5=float(np.mean([p.over_2_5 for p in preds])),
                btts=float(np.mean([p.btts for p in preds])),
                expected_home=float(np.mean([p.expected_home for p in preds])),
                expected_away=float(np.mean([p.expected_away for p in preds])),
                source="stacking_ensemble",
            )

        # Sans meta-model entraîné : utiliser Elo seul (pas de moyenne arbitraire)
        for name, p in zip(self.model_names, preds):
            if name == "elo":
                p.source = "elo_fallback_no_meta"
                return p
        p = preds[0]
        p.source = "ensemble_fallback_no_meta"
        return p

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "StackingEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_default_ensemble(decay: float = 0.0065) -> StackingEnsemble:
    from models.dixon_coles import DixonColesModel
    from models.ratings import EloRating

    ensemble = StackingEnsemble()
    ensemble.add_model(EloRating(), "elo")
    ensemble.add_model(DixonColesModel(decay=decay), "dixon_coles")
    return ensemble
