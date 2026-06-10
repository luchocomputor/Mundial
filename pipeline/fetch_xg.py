"""Fetch xG data from bzzoiro API."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import bzzoiro_headers, load_config

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"


def fetch_event_xg(event_id: int, cfg=None) -> dict | None:
    if cfg is None:
        cfg = load_config()
    url = f"{cfg.bzzoiro_base_url}/events/{event_id}/"
    try:
        resp = requests.get(url, headers=bzzoiro_headers(cfg), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        stats = data.get("statistics", {}) or data.get("stats", {})
        return {
            "fixture_id": event_id,
            "xg_home": stats.get("xg_home") or stats.get("home_xg"),
            "xg_away": stats.get("xg_away") or stats.get("away_xg"),
        }
    except Exception:
        return None


def backfill_xg(matches: pd.DataFrame, cfg=None, max_events: int | None = None) -> pd.DataFrame:
    finished = matches[matches["status"] == "finished"]
    rows = []
    ids = finished["fixture_id"].tolist()
    if max_events:
        ids = ids[:max_events]

    for fid in ids:
        xg = fetch_event_xg(fid, cfg)
        if xg and (xg.get("xg_home") is not None or xg.get("xg_away") is not None):
            rows.append(xg)
        time.sleep(0.1)

    return pd.DataFrame(rows) if rows else pd.DataFrame()
