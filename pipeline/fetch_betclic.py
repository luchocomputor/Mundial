"""
Fetch des cotes Betclic (h2h) et Pinnacle (totals) via The Odds API
pour tous les matchs CDM 2026.

Betclic FR : marché 1X2
Pinnacle   : totaux over/under (référence de marché, cotes sans marge)
"""

import json
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
OUTPUT_PATH = DATA_RAW / "odds_betclic.json"


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def fetch_odds(cfg: dict) -> list[dict]:
    api_cfg = cfg["odds_api"]
    params = {
        "apiKey": api_cfg["key"],
        "regions": api_cfg["regions"],
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
        "bookmakers": "betclic_fr,pinnacle",
    }
    r = requests.get(
        f"{api_cfg['base_url']}/sports/{api_cfg['sport']}/odds/",
        params=params,
        timeout=30,
    )
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  Requêtes API restantes: {remaining}")
    r.raise_for_status()
    return r.json()


def parse_match(match: dict) -> dict | None:
    home = match["home_team"]
    away = match["away_team"]
    date = match["commence_time"]

    result = {
        "match_id": match["id"],
        "home_team": home,
        "away_team": away,
        "date": date,
        "betclic": {},
        "pinnacle": {},
    }

    bk_alias = {"betclic_fr": "betclic", "pinnacle": "pinnacle"}

    for bk in match.get("bookmakers", []):
        bk_raw = bk["key"]
        slot = bk_alias.get(bk_raw)
        if slot is None:
            continue

        for mkt in bk.get("markets", []):
            if mkt["key"] == "h2h":
                outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
                result[slot]["home_win"] = outcomes.get(home)
                result[slot]["draw"] = outcomes.get("Draw")
                result[slot]["away_win"] = outcomes.get(away)

            elif mkt["key"] == "totals":
                # Chercher la ligne 2.5 en priorité, sinon prendre la première dispo
                line_25 = [o for o in mkt["outcomes"] if o.get("point") == 2.5]
                outcomes_raw = line_25 if line_25 else mkt["outcomes"]
                over = next((o["price"] for o in outcomes_raw if o["name"] == "Over"), None)
                under = next((o["price"] for o in outcomes_raw if o["name"] == "Under"), None)
                point = outcomes_raw[0].get("point") if outcomes_raw else None
                result[slot]["over_2.5"] = over
                result[slot]["under_2.5"] = under
                result[slot]["totals_line"] = point

            elif mkt["key"] == "spreads":
                for o in mkt["outcomes"]:
                    pt = o.get("point")
                    price = o["price"]
                    name = o["name"]
                    if name == home:
                        result[slot]["spread_home_point"] = pt
                        result[slot]["spread_home_cote"] = price
                    elif name == away:
                        result[slot]["spread_away_point"] = pt
                        result[slot]["spread_away_cote"] = price

    return result


def fetch_and_save() -> list[dict]:
    cfg = load_config()
    print("Fetching cotes Betclic + Pinnacle CDM 2026...")
    raw = fetch_odds(cfg)

    matches = []
    for m in raw:
        parsed = parse_match(m)
        if parsed:
            matches.append(parsed)

    # Stats de couverture
    betclic_h2h = sum(1 for m in matches if m["betclic"].get("home_win"))
    pinnacle_ou = sum(1 for m in matches if m["pinnacle"].get("over_2.5"))
    pinnacle_ah = sum(1 for m in matches if m["pinnacle"].get("spread_home_point") is not None)
    print(f"  {betclic_h2h}/{len(matches)} matchs avec Betclic 1X2")
    print(f"  {pinnacle_ou}/{len(matches)} matchs avec Pinnacle O/U")
    print(f"  {pinnacle_ah}/{len(matches)} matchs avec Pinnacle Asian Handicap")

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(matches, indent=2, ensure_ascii=False))
    print(f"  Sauvegardé: {OUTPUT_PATH}")
    return matches


def load_odds() -> dict[str, dict]:
    """Retourne un dict indexé par (home_team, away_team) → odds dict."""
    if not OUTPUT_PATH.exists():
        return {}
    matches = json.loads(OUTPUT_PATH.read_text())
    return {(m["home_team"], m["away_team"]): m for m in matches}


if __name__ == "__main__":
    fetch_and_save()
