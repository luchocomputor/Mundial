"""Découverte des league_id internationaux via API bzzoiro."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import bzzoiro_headers, load_config

CLUB_KEYWORDS = [
    "premier", "liga", "serie a", "bundesliga", "ligue 1", "eredivisie",
    "europa league", "champions", "libertadores", "sudamericana",
    "del rey", "fa cup", "copa do brasil", "mls", "super lig",
]

TYPE_RULES = [
    (["world cup 2026", "world cup"], "wc"),
    (["friendly"], "friendly"),
    (["nations league"], "nations_league"),
    (["euro 20", "european championship"], "euro"),
    (["copa america"], "copa"),
    (["gold cup"], "gold_cup"),
    (["africa cup", "afcon"], "can"),
    (["asian cup"], "asian_cup"),
    (["qualification", "qualif"], "qualif"),
]


def _classify(name: str) -> str:
    lower = name.lower()
    for keywords, ctype in TYPE_RULES:
        if any(k in lower for k in keywords):
            return ctype
    return "other"


def _is_international(name: str, country: str | None) -> bool:
    lower = name.lower()
    if any(k in lower for k in CLUB_KEYWORDS):
        return False
    intl_keywords = [
        "world cup", "friendly", "nations league", "euro", "copa america",
        "gold cup", "africa", "asian cup", "qualification", "concacaf",
        "international", "ofc",
    ]
    if any(k in lower for k in intl_keywords):
        return True
    if country and country.lower() in ("international", "world", "fifa"):
        return True
    return False


def discover_international_leagues() -> list[dict]:
    cfg = load_config()
    url = f"{cfg.bzzoiro_base_url}/leagues/"
    resp = requests.get(url, headers=bzzoiro_headers(cfg), params={"limit": 200}, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    international = []
    seen_ids = set()
    for league in results:
        name = league.get("name") or ""
        country = league.get("country") or ""
        if not _is_international(name, country):
            continue
        lid = league["id"]
        if lid in seen_ids:
            continue
        seen_ids.add(lid)
        international.append({
            "id": lid,
            "name": name,
            "competition_type": _classify(name),
        })
    return international


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data" / "leagues_international.yaml"))
    args = parser.parse_args()

    leagues = discover_international_leagues()
    out = {"leagues": leagues}
    Path(args.output).write_text(yaml.dump(out, allow_unicode=True, sort_keys=False))
    print(f"{len(leagues)} ligues → {args.output}")
    print(json.dumps(leagues[:10], indent=2))


if __name__ == "__main__":
    main()
