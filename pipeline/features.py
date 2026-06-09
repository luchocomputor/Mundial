"""
Feature engineering à partir des données brutes.
Produit un DataFrame enrichi pour l'entraînement Dixon-Coles.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

STADIUM_ALTITUDE = {
    "Estadio Azteca": 2240,
    "Estadio Akron": 1650,
    "Estadio BBVA": 500,
    "Estadio Ciudad de Mexico": 2240,
    "AT&T Stadium": 170,
    "MetLife Stadium": 10,
    "SoFi Stadium": 90,
    "Rose Bowl": 270,
    "Levi's Stadium": 15,
    "Arrowhead Stadium": 310,
    "Hard Rock Stadium": 2,
    "Mercedes-Benz Stadium": 300,
    "NRG Stadium": 13,
    "Gillette Stadium": 20,
    "Q2 Stadium": 150,
    "Lincoln Financial Field": 10,
    "BMO Field": 76,
    "BC Place": 3,
}

# Coefficient altitude estimé (placeholder — remplacé par régression si assez de données)
ALTITUDE_COEF = -0.15


def get_altitude_adjustment(venue: str) -> float:
    altitude = STADIUM_ALTITUDE.get(venue, 0)
    return ALTITUDE_COEF * (altitude / 1000)


def add_confederations(df: pd.DataFrame, conf_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["home_confederation"] = out["home_team"].map(conf_map).fillna("")
    out["away_confederation"] = out["away_team"].map(conf_map).fillna("")
    return out


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Jours depuis dernier match et congestion — sans fuite temporelle."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out = out.sort_values("date").reset_index(drop=True)

    last_match: dict[str, pd.Timestamp] = {}
    matches_14d: dict[str, list[pd.Timestamp]] = {}

    days_home, days_away = [], []
    m14_home, m14_away = [], []

    for _, row in out.iterrows():
        home, away, dt = row["home_team"], row["away_team"], row["date"]

        days_home.append(
            (dt - last_match[home]).days if home in last_match else np.nan
        )
        days_away.append(
            (dt - last_match[away]).days if away in last_match else np.nan
        )

        cutoff = dt - pd.Timedelta(days=14)
        m14_home.append(
            sum(1 for d in matches_14d.get(home, []) if d >= cutoff)
        )
        m14_away.append(
            sum(1 for d in matches_14d.get(away, []) if d >= cutoff)
        )

        last_match[home] = dt
        last_match[away] = dt
        matches_14d.setdefault(home, []).append(dt)
        matches_14d.setdefault(away, []).append(dt)

    out["days_since_last_match_home"] = days_home
    out["days_since_last_match_away"] = days_away
    out["matches_last_14d_home"] = m14_home
    out["matches_last_14d_away"] = m14_away
    return out


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_venue_names() -> dict[int, str]:
    path = DATA_RAW / "venues.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["venue_id"], df["name"]))


def load_raw_matches() -> pd.DataFrame:
    path = DATA_RAW / "all_matches.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pas de données à {path}. Lance fetch_data.py d'abord.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if "venue_id" in df.columns and "venue" not in df.columns:
        venue_map = load_venue_names()
        df["venue"] = df["venue_id"].map(venue_map).fillna("")
    return df


def build_training_data(
    df: pd.DataFrame,
    friendly_weight: float = 0.5,
    reference_date: pd.Timestamp | None = None,
    decay: float = 0.0065,
) -> pd.DataFrame:
    if reference_date is None:
        raise ValueError("reference_date obligatoire pour éviter decay dégénéré")

    if reference_date.tzinfo is None:
        reference_date = reference_date.tz_localize("UTC")

    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df["days_ago"] = (reference_date - df["date"]).dt.days.clip(lower=0)
    df["time_weight"] = np.exp(-decay * df["days_ago"])
    if "is_friendly" in df.columns:
        df.loc[df["is_friendly"] == True, "time_weight"] *= friendly_weight

    if "venue" not in df.columns:
        df["venue"] = ""
    df["altitude_adj"] = df["venue"].apply(get_altitude_adjustment)

    return df


def save_processed(df: pd.DataFrame, filename: str = "training_data.parquet"):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_parquet(path, index=False)
    print(f"Sauvegardé : {path} ({len(df)} lignes)")
