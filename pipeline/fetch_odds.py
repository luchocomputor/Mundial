"""Fetch cotes historiques via API bzzoiro — multi-endpoint, reprise batch."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import bzzoiro_headers, load_config
from pipeline.schemas import validate_odds_df

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
PROGRESS_PATH = ROOT / "data" / "processed" / "odds_backfill_progress.json"

BZZOIRO_ODDS_MAP = {
    "home_win": ("1X2", "home"),
    "draw": ("1X2", "draw"),
    "away_win": ("1X2", "away"),
    "over_25_goals": ("over_2.5", "over"),
    "under_25_goals": ("over_2.5", "under"),
    "btts_yes": ("btts", "yes"),
    "btts_no": ("btts", "no"),
}


def _parse_flat_odds(event_id: int, odds_dict: dict, bookmaker: str, snapshot_type: str) -> list[dict]:
    rows = []
    captured_at = pd.Timestamp.now(tz="UTC")
    for key, val in odds_dict.items():
        if val is None or val <= 1.0:
            continue
        mapping = BZZOIRO_ODDS_MAP.get(key)
        if not mapping:
            continue
        market, side = mapping
        rows.append({
            "fixture_id": event_id,
            "bookmaker": bookmaker,
            "snapshot_type": snapshot_type,
            "captured_at": captured_at,
            "market": market,
            "side": side,
            "odds_decimal": float(val),
        })
    return rows


def _parse_multi_bookmaker(event_id: int, payload: dict | list, snapshot_type: str = "close") -> list[dict]:
    """Supporte réponse dict plat ou liste de bookmakers."""
    rows: list[dict] = []
    if isinstance(payload, dict):
        if "home_win" in payload or "draw" in payload:
            bm = payload.get("bookmaker", "bzzoiro")
            return _parse_flat_odds(event_id, payload, bm, snapshot_type)
        for bm_name, odds in payload.items():
            if isinstance(odds, dict):
                rows.extend(_parse_flat_odds(event_id, odds, bm_name, snapshot_type))
        return rows
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            bm = item.get("bookmaker") or item.get("name") or "unknown"
            odds = item.get("odds") or item
            if isinstance(odds, dict):
                rows.extend(_parse_flat_odds(event_id, odds, bm, snapshot_type))
    return rows


def _fetch_event_odds_endpoint(event_id: int, cfg, bookmaker: str | None = None) -> list[dict]:
    """Essaie /events/{id}/odds/ puis /odds/?event_id=."""
    headers = bzzoiro_headers(cfg)
    rows: list[dict] = []

    url = f"{cfg.bzzoiro_base_url}/events/{event_id}/odds/"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.ok:
            data = resp.json()
            odds = data.get("odds") or data.get("results") or data
            rows.extend(_parse_multi_bookmaker(event_id, odds, "close"))
    except Exception:
        pass

    if not rows:
        params: dict = {"event_id": event_id}
        if bookmaker:
            params["bookmaker"] = bookmaker
        try:
            resp = requests.get(
                f"{cfg.bzzoiro_base_url}/odds/",
                headers=headers,
                params=params,
                timeout=15,
            )
            if resp.ok:
                data = resp.json()
                results = data.get("results") or data.get("odds") or []
                if isinstance(results, dict):
                    rows.extend(_parse_multi_bookmaker(event_id, results, "close"))
                elif isinstance(results, list):
                    for item in results:
                        eid = item.get("event_id") or item.get("fixture_id") or event_id
                        odds = item.get("odds") or item
                        rows.extend(_parse_multi_bookmaker(eid, odds, item.get("snapshot_type", "close")))
        except Exception:
            pass

    # Préférer pinnacle si disponible
    pinnacle = [r for r in rows if "pinnacle" in r["bookmaker"].lower()]
    if pinnacle:
        return pinnacle
    return rows


def fetch_odds_for_event(
    event_id: int,
    cfg=None,
    bookmaker: str | None = "pinnacle",
) -> list[dict]:
    if cfg is None:
        cfg = load_config()
    return _fetch_event_odds_endpoint(event_id, cfg, bookmaker)


def fetch_closing_odds_batch(
    fixture_ids: list[int],
    cfg=None,
    sleep: float = 0.15,
    progress_path: Path = PROGRESS_PATH,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()

    done: set[int] = set()
    if progress_path.exists():
        try:
            done = set(json.loads(progress_path.read_text()).get("completed", []))
        except Exception:
            done = set()

    all_rows = []
    completed = list(done)
    for fid in fixture_ids:
        if fid in done:
            continue
        rows = fetch_odds_for_event(fid, cfg)
        if rows:
            all_rows.extend(rows)
        completed.append(fid)
        if len(completed) % 25 == 0:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({"completed": completed}, indent=2))
        time.sleep(sleep)

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps({"completed": completed}, indent=2))

    if not all_rows:
        return pd.DataFrame()
    return validate_odds_df(pd.DataFrame(all_rows))


def backfill_odds_history(
    matches: pd.DataFrame,
    cfg=None,
    max_events: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Backfill cotes pour matchs finished avec reprise."""
    finished = matches[matches["status"] == "finished"].sort_values("date", ascending=False)
    ids = finished["fixture_id"].tolist()

    if resume and PROGRESS_PATH.exists():
        done = set(json.loads(PROGRESS_PATH.read_text()).get("completed", []))
        ids = [i for i in ids if i not in done]

    if max_events:
        ids = ids[:max_events]

    new_df = fetch_closing_odds_batch(ids, cfg)

    existing_path = DATA_RAW / "odds_history.parquet"
    if existing_path.exists() and not new_df.empty:
        old = pd.read_parquet(existing_path)
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["fixture_id", "bookmaker", "market", "side", "snapshot_type"],
            keep="last",
        )
        return validate_odds_df(combined)
    return new_df


def save_odds_history(df: pd.DataFrame) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / "odds_history.parquet"
    df.to_parquet(path, index=False)
    return path
