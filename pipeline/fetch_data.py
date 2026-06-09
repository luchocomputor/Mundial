"""
Fetching data from bzzoiro Sports Data API.
Docs : https://sports.bzzoiro.com
Token dans config.yaml → bzzoiro.token

League IDs :
  27  = World Cup (2014, 2018, 2022, 2026)
  31  = International Friendly Games
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config.yaml"

BASE_URL = "https://sports.bzzoiro.com/api/v2"
LEAGUE_WC = 27
LEAGUE_FRIENDLY = 31


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _headers(cfg: dict) -> dict:
    return {"Authorization": f"Token {cfg['bzzoiro']['token']}"}


def _get(path: str, params: dict, cfg: dict) -> dict:
    resp = requests.get(
        BASE_URL + path,
        headers=_headers(cfg),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get_all_pages(path: str, params: dict, cfg: dict, sleep: float = 0.2) -> list:
    """Récupère toutes les pages d'un endpoint paginé."""
    results = []
    offset = 0
    limit = 100
    while True:
        data = _get(path, {**params, "limit": limit, "offset": offset}, cfg)
        batch = data.get("results", [])
        results.extend(batch)
        if not data.get("next"):
            break
        offset += limit
        time.sleep(sleep)
    return results


def _parse_event(e: dict, is_friendly: bool) -> dict:
    return {
        "fixture_id": e["id"],
        "date": e["event_date"],
        "league_id": e["league_id"],
        "season_id": e.get("season_id"),
        "is_friendly": is_friendly,
        "home_team": e["home_team"],
        "away_team": e["away_team"],
        "home_team_id": e["home_team_id"],
        "away_team_id": e["away_team_id"],
        "home_goals": e.get("home_score"),
        "away_goals": e.get("away_score"),
        "home_goals_ht": e.get("home_score_ht"),
        "away_goals_ht": e.get("away_score_ht"),
        "venue_id": e.get("venue_id"),
        "group_name": e.get("group_name", ""),
        "round_name": e.get("round_name", ""),
        "status": e.get("status", ""),
        "is_neutral": e.get("is_neutral_ground", False),
        "home_coach_id": e.get("home_coach_id"),
        "away_coach_id": e.get("away_coach_id"),
    }


def fetch_world_cup_matches(cfg: dict | None = None) -> pd.DataFrame:
    """Tous les matchs CDM (2014, 2018, 2022, 2026) depuis bzzoiro."""
    if cfg is None:
        cfg = load_config()
    events = _get_all_pages("/events/", {"league_id": LEAGUE_WC}, cfg)
    rows = [_parse_event(e, is_friendly=False) for e in events]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def fetch_friendly_matches(cfg: dict | None = None) -> pd.DataFrame:
    """Tous les amicaux internationaux disponibles."""
    if cfg is None:
        cfg = load_config()
    events = _get_all_pages("/events/", {"league_id": LEAGUE_FRIENDLY}, cfg)
    rows = [_parse_event(e, is_friendly=True) for e in events]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def fetch_upcoming_fixtures(cfg: dict | None = None) -> pd.DataFrame:
    """Matchs CDM 2026 et amicaux à venir (non joués)."""
    if cfg is None:
        cfg = load_config()

    wc_events = _get_all_pages("/events/", {"league_id": LEAGUE_WC, "status": "notstarted"}, cfg)
    fr_events = _get_all_pages("/events/", {"league_id": LEAGUE_FRIENDLY, "status": "notstarted"}, cfg)

    rows = (
        [_parse_event(e, False) for e in wc_events]
        + [_parse_event(e, True) for e in fr_events]
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_event_odds(event_id: int, cfg: dict | None = None) -> dict:
    """Cotes 1X2/over-under/BTTS pour un match (nulles si pas encore ouvertes)."""
    if cfg is None:
        cfg = load_config()
    try:
        data = _get(f"/events/{event_id}/odds/", {}, cfg)
        return data.get("odds", {})
    except Exception:
        return {}


def fetch_odds_batch(event_ids: list[int], cfg: dict | None = None) -> dict[int, dict]:
    """Cotes pour une liste de matchs. Retourne {event_id: odds_dict}."""
    if cfg is None:
        cfg = load_config()
    result = {}
    for eid in event_ids:
        odds = fetch_event_odds(eid, cfg)
        non_null = {k: v for k, v in odds.items() if v is not None}
        if non_null:
            result[eid] = non_null
        time.sleep(0.15)
    return result


def fetch_venues(cfg: dict | None = None) -> pd.DataFrame:
    """Tous les stades CDM 2026 avec noms et villes."""
    if cfg is None:
        cfg = load_config()

    wc_events = _get_all_pages("/events/", {"league_id": LEAGUE_WC}, cfg)
    venue_ids = {e["venue_id"] for e in wc_events if e.get("venue_id")}

    rows = []
    for vid in venue_ids:
        try:
            v = _get(f"/venues/{vid}/", {}, cfg)
            rows.append({
                "venue_id": v["id"],
                "name": v["name"],
                "city": v.get("city", ""),
                "country": v.get("country", ""),
                "capacity": v.get("capacity"),
                "latitude": v.get("latitude"),
                "longitude": v.get("longitude"),
            })
            time.sleep(0.1)
        except Exception:
            pass

    return pd.DataFrame(rows)


def fetch_wc2026_group_stage(cfg: dict | None = None) -> pd.DataFrame:
    """Matchs de phase de groupes CDM 2026 uniquement (avec noms d'équipes réels)."""
    df = fetch_world_cup_matches(cfg)
    if df.empty:
        return df
    # Garder uniquement 2026 avec group renseigné et équipes nommées
    mask = (
        (df["date"].dt.year == 2026)
        & (df["group_name"].notna())
        & (df["group_name"] != "")
        & (~df["home_team"].str.startswith(("W", "L", "R", "Q", "H", "G", "1", "2", "3")))
    )
    return df[mask].copy()


def fetch_and_save_all():
    cfg = load_config()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("Fetching tous les matchs CDM (2014-2026)...")
    df_wc = fetch_world_cup_matches(cfg)
    if not df_wc.empty:
        df_wc.to_parquet(DATA_RAW / "wc_all.parquet", index=False)
        print(f"  {len(df_wc)} matchs WC ({df_wc['date'].dt.year.min()}–{df_wc['date'].dt.year.max()})")

    print("Fetching amicaux internationaux...")
    df_fr = fetch_friendly_matches(cfg)
    if not df_fr.empty:
        df_fr.to_parquet(DATA_RAW / "friendly_all.parquet", index=False)
        print(f"  {len(df_fr)} amicaux")

    print("Combinaison de toutes les données...")
    all_parts = [df for df in [df_wc, df_fr] if not df.empty]
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        combined = combined.drop_duplicates(subset=["fixture_id"])
        combined.to_parquet(DATA_RAW / "all_matches.parquet", index=False)
        finished = combined[combined["status"] == "finished"]
        print(f"  Total : {len(combined)} matchs dont {len(finished)} terminés")

    print("Fetching stades CDM 2026...")
    df_venues = fetch_venues(cfg)
    if not df_venues.empty:
        df_venues.to_parquet(DATA_RAW / "venues.parquet", index=False)
        print(f"  {len(df_venues)} stades")

    print("Terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Fetche tout et sauvegarde")
    args = parser.parse_args()
    if args.all:
        fetch_and_save_all()
    else:
        fetch_and_save_all()
