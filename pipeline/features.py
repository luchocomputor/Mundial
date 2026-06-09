"""
Feature engineering à partir des données brutes.
Produit un DataFrame enrichi pour l'entraînement Dixon-Coles.
"""

from pathlib import Path

import numpy as np
import pandas as pd


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

FIFA_RANKING_2026 = {
    "Argentina": 1,
    "France": 2,
    "Spain": 3,
    "England": 4,
    "Brazil": 5,
    "Portugal": 6,
    "Belgium": 7,
    "Netherlands": 8,
    "Germany": 9,
    "Colombia": 10,
    "Italy": 11,
    "Croatia": 12,
    "Uruguay": 13,
    "Morocco": 14,
    "Japan": 15,
    "Senegal": 16,
    "USA": 17,
    "Mexico": 18,
    "Switzerland": 19,
    "Denmark": 20,
    "South Korea": 21,
    "Austria": 22,
    "Australia": 23,
    "Ecuador": 24,
    "Turkey": 25,
    "Canada": 26,
    "Serbia": 27,
    "Poland": 28,
    "Iran": 29,
    "Algeria": 30,
    "Egypt": 31,
    "Ghana": 32,
    "Ivory Coast": 33,
    "Saudi Arabia": 34,
    "Tunisia": 35,
    "Paraguay": 36,
    "Sweden": 37,
    "Bosnia and Herzegovina": 38,
    "Czech Republic": 39,
    "New Zealand": 40,
    "Jordan": 41,
    "Uzbekistan": 42,
    "Iraq": 43,
    "Qatar": 44,
    "South Africa": 45,
    "Cabo Verde": 46,
    "DR Congo": 47,
}


def get_altitude_adjustment(venue: str) -> float:
    altitude = STADIUM_ALTITUDE.get(venue, 0)
    return -0.3 * (altitude / 1000)


def get_fifa_ranking(team: str) -> int:
    return FIFA_RANKING_2026.get(team, 50)


def load_venue_names() -> dict[int, str]:
    """Retourne {venue_id: venue_name} depuis le parquet des stades."""
    path = DATA_RAW / "venues.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return dict(zip(df["venue_id"], df["name"]))


# Leagues de qualifications confédérales exclues du training — les niveaux entre zones
# sont incomparables (Japan vs Kyrgyzstan ≠ France vs Portugal) et biaiseraient les
# paramètres des équipes asiatiques / africaines / CONCACAF vers l'haut.
QUALIFICATION_LEAGUE_IDS = {58, 59, 60, 61, 62, 63}


def load_raw_matches(exclude_qualifications: bool = True) -> pd.DataFrame:
    path = DATA_RAW / "all_matches.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Pas de données à {path}. Lance fetch_data.py d'abord.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)

    if exclude_qualifications and "league_id" in df.columns:
        before = len(df)
        df = df[~df["league_id"].isin(QUALIFICATION_LEAGUE_IDS)]
        print(f"  Qualifications exclues: {before - len(df)} matchs retirés ({len(df)} restants)")

    # Résoudre venue_id → nom du stade si disponible
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
        reference_date = pd.Timestamp.now(tz="UTC")
    elif reference_date.tzinfo is None:
        reference_date = reference_date.tz_localize("UTC")

    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    df["days_ago"] = (reference_date - df["date"]).dt.days.clip(lower=0)
    df["time_weight"] = np.exp(-decay * df["days_ago"])
    df.loc[df["is_friendly"] == True, "time_weight"] *= friendly_weight

    df["home_ranking"] = df["home_team"].map(get_fifa_ranking)
    df["away_ranking"] = df["away_team"].map(get_fifa_ranking)
    df["ranking_diff"] = df["home_ranking"] - df["away_ranking"]

    if "venue" not in df.columns:
        df["venue"] = ""
    df["altitude_adj"] = df["venue"].apply(get_altitude_adjustment)

    return df


def build_corner_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sous-ensemble avec statistiques de corners si disponibles."""
    if "home_corners" not in df.columns or "away_corners" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["home_corners", "away_corners"]).copy()


def save_processed(df: pd.DataFrame, filename: str = "training_data.parquet"):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_parquet(path, index=False)
    print(f"Sauvegardé : {path} ({len(df)} lignes)")


if __name__ == "__main__":
    df = load_raw_matches()
    print(f"{len(df)} matchs bruts chargés.")
    training = build_training_data(df)
    save_processed(training)
    print("Feature engineering terminé.")
