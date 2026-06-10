"""
Fetching data from bzzoiro Sports Data API.
Docs : https://sports.bzzoiro.com
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from pipeline.config import bzzoiro_headers, load_config
from pipeline.placeholder import filter_real_teams
from pipeline.schemas import validate_matches_df
from pipeline.status import normalize_status

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"

LEAGUE_WC = 27
LEAGUE_FRIENDLY = 31


def _get(path: str, params: dict, cfg) -> dict:
    import requests

    resp = requests.get(
        cfg.bzzoiro_base_url + path,
        headers=bzzoiro_headers(cfg),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _get_all_pages(path: str, params: dict, cfg, sleep: float = 0.2) -> list:
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


def _load_league_config() -> list[dict]:
    path = ROOT / "data" / "leagues_international.yaml"
    if not path.exists():
        return [
            {"id": LEAGUE_WC, "name": "World Cup", "competition_type": "wc"},
            {"id": LEAGUE_FRIENDLY, "name": "Friendlies", "competition_type": "friendly"},
        ]
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("leagues", [])


def _parse_event(e: dict, competition_type: str, is_friendly: bool) -> dict:
    return {
        "fixture_id": e["id"],
        "date": e["event_date"],
        "league_id": e["league_id"],
        "league_name": e.get("league_name", ""),
        "season_id": e.get("season_id"),
        "competition_type": competition_type,
        "is_friendly": is_friendly,
        "home_team": e["home_team"],
        "away_team": e["away_team"],
        "home_team_id": e.get("home_team_id"),
        "away_team_id": e.get("away_team_id"),
        "home_goals": e.get("home_score"),
        "away_goals": e.get("away_score"),
        "home_goals_ht": e.get("home_score_ht"),
        "away_goals_ht": e.get("away_score_ht"),
        "venue_id": e.get("venue_id"),
        "venue": e.get("venue_name", ""),
        "city": e.get("city", ""),
        "group_name": e.get("group_name", ""),
        "round_name": e.get("round_name", ""),
        "status": normalize_status(e.get("status", "")),
        "is_neutral": e.get("is_neutral_ground", False),
        "home_coach_id": e.get("home_coach_id"),
        "away_coach_id": e.get("away_coach_id"),
        "xg_home": e.get("xg_home"),
        "xg_away": e.get("xg_away"),
    }


def fetch_league_matches(league_id: int, competition_type: str, is_friendly: bool, cfg=None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    events = _get_all_pages("/events/", {"league_id": league_id}, cfg)
    rows = [_parse_event(e, competition_type, is_friendly) for e in events]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def fetch_international_matches(cfg=None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    leagues = _load_league_config()
    parts = []
    for league in leagues:
        lid = league["id"]
        ctype = league.get("competition_type", "other")
        is_friendly = ctype == "friendly"
        print(f"  Fetching league {lid} ({league.get('name', '?')})...")
        df = fetch_league_matches(lid, ctype, is_friendly, cfg)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["fixture_id"])
    return validate_matches_df(combined)


def fetch_world_cup_matches(cfg=None) -> pd.DataFrame:
    return fetch_league_matches(LEAGUE_WC, "wc", False, cfg)


def fetch_friendly_matches(cfg=None) -> pd.DataFrame:
    return fetch_league_matches(LEAGUE_FRIENDLY, "friendly", True, cfg)


def fetch_upcoming_fixtures(cfg=None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    leagues = _load_league_config()
    parts = []
    for league in leagues:
        events = _get_all_pages(
            "/events/",
            {"league_id": league["id"], "status": "notstarted"},
            cfg,
        )
        ctype = league.get("competition_type", "other")
        parts.extend(
            [_parse_event(e, ctype, ctype == "friendly") for e in events]
        )
    df = pd.DataFrame(parts)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_event_odds(event_id: int, cfg=None) -> dict:
    if cfg is None:
        cfg = load_config()
    try:
        data = _get(f"/events/{event_id}/odds/", {}, cfg)
        return data.get("odds", {})
    except Exception:
        return {}


def fetch_odds_batch(event_ids: list[int], cfg=None) -> dict[int, dict]:
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


def fetch_venues(cfg=None) -> pd.DataFrame:
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


def fetch_wc2026_group_stage(cfg=None) -> pd.DataFrame:
    df = fetch_world_cup_matches(cfg)
    if df.empty:
        return df
    mask = (
        (df["date"].dt.year == 2026)
        & (df["group_name"].notna())
        & (df["group_name"] != "")
    )
    return filter_real_teams(df[mask])


def fetch_and_save_all():
    cfg = load_config()
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("Fetching matchs internationaux...")
    df_all = fetch_international_matches(cfg)
    if not df_all.empty:
        df_all.to_parquet(DATA_RAW / "all_matches.parquet", index=False)
        finished = (df_all["status"] == "finished").sum()
        print(f"  {len(df_all)} matchs ({finished} terminés)")

        df_wc = df_all[df_all["competition_type"] == "wc"]
        if not df_wc.empty:
            df_wc.to_parquet(DATA_RAW / "wc_all.parquet", index=False)

        df_fr = df_all[df_all["competition_type"] == "friendly"]
        if not df_fr.empty:
            df_fr.to_parquet(DATA_RAW / "friendly_all.parquet", index=False)

    print("Fetching stades...")
    df_venues = fetch_venues(cfg)
    if not df_venues.empty:
        df_venues.to_parquet(DATA_RAW / "venues.parquet", index=False)
        print(f"  {len(df_venues)} stades")

    print("Terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    fetch_and_save_all()
