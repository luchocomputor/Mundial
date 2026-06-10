"""Charge le modèle de production (Elo par défaut)."""

from __future__ import annotations

from pathlib import Path

from models.ensemble import StackingEnsemble
from models.ratings import EloRating

ROOT = Path(__file__).parent.parent
ELO_PATH = ROOT / "models" / "artifacts" / "elo.pkl"
ENSEMBLE_PATH = ROOT / "models" / "artifacts" / "ensemble.pkl"
DC_PATH = ROOT / "models" / "dixon_coles.pkl"


def load_production_model(prefer_elo: bool = True):
    """
    Charge Elo par défaut (meilleur RPS audit CDM 2022).
    N'utilise l'ensemble que si meta_model entraîné.
    """
    if ELO_PATH.exists():
        return EloRating.load()
    if prefer_elo:
        return EloRating()

    if ENSEMBLE_PATH.exists():
        ensemble = StackingEnsemble.load()
        if ensemble.meta_model is not None:
            return ensemble

    if DC_PATH.exists():
        from models.dixon_coles import DixonColesModel
        return DixonColesModel.load()

    return EloRating()


def load_calibrator():
    from models.calibration import MarketCalibrator, CALIB_PATH
    if CALIB_PATH.exists():
        return MarketCalibrator.load()
    return None
