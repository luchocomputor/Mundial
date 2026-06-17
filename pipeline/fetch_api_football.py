"""Client API-Football (API-Sports direct) — cache disque + budget 100 req/jour.

Plan Free : 100 requêtes/jour. Stratégie : tout réponse est mise en cache sur
disque (data/raw/apifootball/) → une donnée n'est jamais re-payée, et un
compteur journalier bloque avant de dépasser le quota.

⚠️ Plan Free : accès DONNÉES limité aux saisons 2022→2024 (testé). La CDM 2026
(saison courante) renvoie "Free plans do not have access to this season" — donc
PAS de data live 2026 sans plan payant. Accessible et utile ici : WC 2022
(league_id=1, season=2022) en données complètes — fixtures, compos, events,
buteurs, stats match/joueurs, classement → backtest & calibration du modèle.
La clé vit dans .env (API_FOOTBALL_KEY), jamais dans config.yaml (tracké).

Usage :
    from pipeline.fetch_api_football import topscorers, lineups, remaining
    data = topscorers(2026)        # caché : gratuit au 2e appel
    print(remaining(), "requêtes restantes aujourd'hui")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "raw" / "apifootball"
QUOTA = CACHE / "_quota.json"
BASE = "https://v3.football.api-sports.io"
WC_LEAGUE = 1
DAILY_LIMIT = 100


def _key() -> str:
    load_dotenv(ROOT / ".env")
    k = os.environ.get("API_FOOTBALL_KEY", "")
    if not k:
        raise RuntimeError("API_FOOTBALL_KEY absent de .env")
    return k


# ── Quota journalier ────────────────────────────────────────────────────────
def _quota_load() -> dict:
    today = date.today().isoformat()
    if QUOTA.exists():
        try:
            q = json.loads(QUOTA.read_text())
            if q.get("date") == today:
                return q
        except Exception:
            pass
    return {"date": today, "count": 0}


def _quota_save(q: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    QUOTA.write_text(json.dumps(q))


def remaining() -> int:
    """Requêtes restantes aujourd'hui (sur DAILY_LIMIT)."""
    return DAILY_LIMIT - _quota_load()["count"]


# ── GET caché ───────────────────────────────────────────────────────────────
def _cache_path(endpoint: str, params: dict) -> Path:
    sig = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    h = hashlib.md5(sig.encode()).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    return CACHE / f"{safe}_{h}.json"


def get(endpoint: str, params: dict | None = None, ttl_hours: float = 24,
        force: bool = False) -> dict:
    """GET avec cache disque. Ne consomme une requête que si le cache est
    absent ou périmé (> ttl_hours). Respecte le budget DAILY_LIMIT/jour ;
    si le quota est atteint, retombe sur le cache périmé plutôt que d'échouer."""
    params = params or {}
    cp = _cache_path(endpoint, params)
    if cp.exists() and not force:
        age_h = (time.time() - cp.stat().st_mtime) / 3600
        if age_h < ttl_hours:
            return json.loads(cp.read_text())

    q = _quota_load()
    if q["count"] >= DAILY_LIMIT:
        if cp.exists():
            return json.loads(cp.read_text())
        raise RuntimeError(
            f"Quota {DAILY_LIMIT}/jour atteint et aucun cache — réessaie demain.")

    resp = requests.get(f"{BASE}/{endpoint.lstrip('/')}",
                        headers={"x-apisports-key": _key()},
                        params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    q["count"] += 1
    _quota_save(q)
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(data))
    return data


# ── Raccourcis Coupe du Monde (league_id=1) ─────────────────────────────────
def fixtures(season: int = 2026, **kw) -> dict:
    return get("fixtures", {"league": WC_LEAGUE, "season": season, **kw})

def lineups(fixture_id: int) -> dict:
    return get("fixtures/lineups", {"fixture": fixture_id})

def events(fixture_id: int) -> dict:
    return get("fixtures/events", {"fixture": fixture_id})

def match_stats(fixture_id: int) -> dict:
    return get("fixtures/statistics", {"fixture": fixture_id})

def standings(season: int = 2026) -> dict:
    return get("standings", {"league": WC_LEAGUE, "season": season})

def topscorers(season: int = 2026) -> dict:
    return get("players/topscorers", {"league": WC_LEAGUE, "season": season})

def team_statistics(team_id: int, season: int = 2026) -> dict:
    return get("teams/statistics", {"league": WC_LEAGUE, "season": season, "team": team_id})

def injuries(season: int = 2026, **kw) -> dict:
    return get("injuries", {"league": WC_LEAGUE, "season": season, **kw})

def match_odds(season: int = 2026, **kw) -> dict:
    return get("odds", {"league": WC_LEAGUE, "season": season, **kw})


if __name__ == "__main__":
    s = get("status", ttl_hours=0)["response"]
    print(f"Compte : {s['account']['firstname']} · plan {s['subscription']['plan']}")
    print(f"Requêtes API aujourd'hui : {s['requests']['current']}/{s['requests']['limit_day']}")
    print(f"Budget local restant : {remaining()}/{DAILY_LIMIT}")
