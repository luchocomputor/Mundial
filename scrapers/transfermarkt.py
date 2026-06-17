"""Scraper Transfermarkt maison — effectifs des sélections + valeurs marchandes.

Pas d'API officielle TM. On scrape le HTML (bs4) en restant poli : cache disque
(une page n'est re-téléchargée qu'après TTL), User-Agent navigateur, pause entre
requêtes. Fragile par nature (le HTML TM peut bouger) — coût assumé du scraping
maison. La valeur marchande est un excellent proxy de force d'une sélection.

Usage :
    from scrapers.transfermarkt import squad, WC2026_VEREIN
    players = squad(3377)            # France
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "raw" / "transfermarkt"
BASE = "https://www.transfermarkt.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
PAUSE_S = 3.0  # politesse anti-ban

# verein_id Transfermarkt des sélections CDM 2026 (à compléter — récupérables via
# la page compétition TM ; on démarre avec quelques-unes pour valider le scraper).
WC2026_VEREIN: dict[str, int] = {
    "France": 3377, "Spain": 3375, "England": 3299, "Germany": 3262,
    "Argentina": 3437, "Brazil": 3439, "Portugal": 3300, "Netherlands": 3379,
    "Belgium": 3382, "Croatia": 3556, "Morocco": 3575, "USA": 3505,
}


# Noms parquet ≠ libellés de recherche Transfermarkt
TM_NAME_OVERRIDE: dict[str, str] = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Türkiye": "Turkey",
}

# Sélections dont la recherche TM échoue (homonymes, libellé exotique) → id direct
VEREIN_ID_OVERRIDE: dict[str, int] = {
    "USA": 3505,        # US men's national team (recherche renvoie des clubs)
    "DR Congo": 3854,   # "Democratic Republic of the Congo"
}


def _fetch(path: str, ttl_hours: float = 168) -> str:
    """GET caché (TTL 7j par défaut) + pause polie. Ne re-télécharge pas si frais."""
    url = f"{BASE}/{path.lstrip('/')}"
    cp = CACHE / f"{hashlib.md5(url.encode()).hexdigest()[:16]}.html"
    if cp.exists() and (time.time() - cp.stat().st_mtime) / 3600 < ttl_hours:
        return cp.read_text()
    time.sleep(PAUSE_S)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    CACHE.mkdir(parents=True, exist_ok=True)
    cp.write_text(r.text)
    return r.text


def parse_market_value(txt: str) -> int | None:
    """'€30.00m' → 30000000 ; '€500k' → 500000 ; '-' → None."""
    t = (txt or "").replace("€", "").strip().lower()
    m = re.match(r"([\d.,]+)\s*(bn|m|k)?", t)
    if not m or not m.group(1):
        return None
    val = float(m.group(1).replace(",", ""))
    mult = {"k": 1e3, "m": 1e6, "bn": 1e9}.get(m.group(2), 1.0)
    return int(val * mult)


def search_verein_id(name: str) -> int | None:
    """Résout le verein_id d'une sélection via la recherche TM. La sélection
    nationale est la candidate dont le libellé == le nom recherché."""
    if name in VEREIN_ID_OVERRIDE:
        return VEREIN_ID_OVERRIDE[name]
    q = TM_NAME_OVERRIDE.get(name, name)
    html = _fetch(f"schnellsuche/ergebnis/schnellsuche?query={quote(q)}")
    soup = BeautifulSoup(html, "lxml")
    cands = []
    for a in soup.select('a[href*="/verein/"]'):
        m = re.search(r"/verein/(\d+)", a["href"])
        txt = a.get_text(strip=True)
        if m and txt and not txt.isdigit():
            cands.append((txt, int(m.group(1))))
    for txt, vid in cands:
        if txt.lower() == q.lower():
            return vid
    return cands[0][1] if cands else None


def squad(verein_id: int, season: int | None = None) -> list[dict]:
    """Effectif détaillé d'une sélection → liste de joueurs avec valeur marchande."""
    saison = f"/saison_id/{season}" if season else ""
    html = _fetch(f"x/kader/verein/{verein_id}{saison}/plus/1")
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    if not table:
        return []
    out = []
    for tr in table.select("tbody > tr"):
        tds = tr.select(":scope > td")
        if len(tds) < 10:
            continue
        link = tr.select_one("td.hauptlink a")
        if not link:
            continue
        name = link.get_text(strip=True)
        pid_m = re.search(r"/spieler/(\d+)", link.get("href", ""))
        player_id = int(pid_m.group(1)) if pid_m else None
        # poste : 2e ligne de l'inline-table de la cellule joueur
        position = None
        inl = tds[1].select_one("table.inline-table")
        if inl:
            trs = inl.select("tr")
            if len(trs) >= 2:
                position = trs[-1].get_text(strip=True)
        if not position:  # repli : texte cellule moins le nom
            position = tds[1].get_text(" ", strip=True).replace(name, "").strip() or None
        age = None
        ma = re.search(r"\((\d+)\)", tds[2].get_text(" ", strip=True))
        if ma:
            age = int(ma.group(1))
        img = tds[3].select_one("img")
        club = (img.get("alt") or img.get("title")) if img else None
        out.append(dict(
            player=name,
            player_id=player_id,
            position=position,
            age=age,
            club=club,
            caps=tds[6].get_text(strip=True),
            intl_goals=tds[7].get_text(strip=True),
            market_value_eur=parse_market_value(tds[9].get_text(strip=True)),
        ))
    return out


def player_performance(player_id: int) -> list[dict]:
    """Stats par compétition de la SAISON COURANTE (endpoint ceapi JSON, propre).
    Champs : competitionDescription, nameSeason, gamesPlayed, goalsScored, assists,
    minutesPlayed, startElevenPercent, yellowCards, redCards, cleanSheets…"""
    txt = _fetch(f"ceapi/player/{player_id}/performance", ttl_hours=72)
    try:
        data = json.loads(txt)
        return data if isinstance(data, list) else []
    except Exception:
        return []


if __name__ == "__main__":
    import pandas as pd
    rows = squad(WC2026_VEREIN["France"])
    df = pd.DataFrame(rows)
    print(f"France : {len(df)} joueurs")
    print(df[["player", "position", "age", "market_value_eur"]].head(12).to_string(index=False))
    print("Valeur totale effectif :", f"{df.market_value_eur.sum()/1e6:.0f}M€")
