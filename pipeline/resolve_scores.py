"""
Auto-résolution des bets depuis les scores bzzoiro.
Usage : python -m pipeline.resolve_scores [--dry-run]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests

from pipeline.teams import normalize as _norm
from pipeline.tz import mtl_date

ROOT = Path(__file__).parent.parent
BETS_LOG = ROOT / "data" / "bets_log.csv"


def _teams_match(a: str, b: str) -> bool:
    return _norm(str(a)) == _norm(str(b))


# ── Résolution marché ─────────────────────────────────────────────────────────
def resolve_bet(sh: int, sa: int, market: str, side: str) -> str | None:
    """Retourne 'W', 'L', 'V' ou None si marché non supporté."""
    m, s = market.lower(), side.lower()

    if m == "1x2":
        result = "home" if sh > sa else ("draw" if sh == sa else "away")
        return "W" if s == result else "L"

    if m == "btts":
        btts = sh > 0 and sa > 0
        return "W" if (s == "yes") == btts else "L"

    # over / under génériques : over_2.5, over_1.5, over_3.5, team_total …
    ou_match = re.match(r"(over|under)_(\d+(?:\.\d+)?)", m)
    if ou_match:
        direction = ou_match.group(1)
        thresh = float(ou_match.group(2))
        total = sh + sa
        over = total > thresh
        return "W" if (s == "over") == over else "L"

    # team total : korea_over_1.5, mexico_under_0.5 …
    tt_match = re.match(r"(.+?)_(over|under)_(\d+(?:\.\d+)?)", m)
    if tt_match:
        direction = tt_match.group(2)
        thresh = float(tt_match.group(3))
        # Le side précise quelle équipe (home implicite si non déduit)
        # On regarde si side contient "home" / "away" ou le nom de l'équipe
        # Ici on suppose que la colonne home_team/away_team du log est accessible
        # et que le market encode la team. On utilise le champ side pour savoir.
        # Convention : side = "korea_over_1.5" → team encodée dans le market name
        # Pour simplifier : le champ market contient "team_total" et side "korea_over_1.5"
        # On extrait la direction et le seuil depuis side
        side_match = re.match(r"(.+?)_(over|under)_(\d+(?:\.\d+)?)", s)
        if side_match:
            direction = side_match.group(2)
            thresh = float(side_match.group(3))
            # On ne peut pas savoir si c'est home ou away sans plus d'info
            # → non supporté automatiquement, laisser None
            return None
        return None

    return None


# ── Fetch scores bzzoiro ──────────────────────────────────────────────────────
def fetch_finished_scores(cfg=None) -> list[dict]:
    """Retourne tous les events WC 2026 terminés avec scores."""
    import sys; sys.path.insert(0, str(ROOT))
    from pipeline.config import bzzoiro_headers, load_config

    if cfg is None:
        cfg = load_config()

    results = []
    offset = 0
    max_pages = 5
    for _ in range(max_pages):
        resp = requests.get(
            cfg.bzzoiro_base_url + "/events",
            headers=bzzoiro_headers(cfg),
            params={"league_id": 27, "status": "finished", "limit": 100, "offset": offset},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        results.extend(batch)
        if not data.get("next"):
            break
        offset += 100

    # Garder seulement WC 2026
    return [
        r for r in results
        if "2026" in r.get("event_date", "")
        and r.get("home_score") is not None
        and r.get("away_score") is not None
    ]


# ── Résolution team_total ─────────────────────────────────────────────────────
def resolve_team_total(sh: int, sa: int, side: str, home_team: str, away_team: str) -> str | None:
    """
    Résout les marchés team_total.
    side format: 'korea_over_1.5' ou 'mexico_under_0.5'
    """
    m = re.match(r"(.+?)_(over|under)_(\d+(?:\.\d+)?)", side.lower())
    if not m:
        return None

    team_slug, direction, thresh_s = m.group(1), m.group(2), float(m.group(3))

    # Déterminer si home ou away
    if _norm(home_team).startswith(team_slug) or team_slug in _norm(home_team):
        goals = sh
    elif _norm(away_team).startswith(team_slug) or team_slug in _norm(away_team):
        goals = sa
    else:
        # Fallback: chercher la correspondance la plus proche
        hn, an = _norm(home_team), _norm(away_team)
        if any(w in hn for w in team_slug.split("_")):
            goals = sh
        elif any(w in an for w in team_slug.split("_")):
            goals = sa
        else:
            return None

    over = goals > thresh_s
    return "W" if (direction == "over") == over else "L"


# ── Main ──────────────────────────────────────────────────────────────────────
def run(dry_run: bool = False) -> dict:
    if not BETS_LOG.exists():
        return {"resolved": 0, "skipped": 0, "details": []}

    df = pd.read_csv(BETS_LOG)
    if df.empty:
        return {"resolved": 0, "skipped": 0, "details": []}

    # Charger les scores
    scores = fetch_finished_scores()
    score_map: dict[tuple, tuple[int, int]] = {}
    for e in scores:
        d = mtl_date(e["event_date"])
        ht, at = e["home_team"], e["away_team"]
        sh, sa = int(e["home_score"]), int(e["away_score"])
        score_map[(_norm(ht), _norm(at), d)] = (sh, sa)
        # Reverse aussi pour retrouver dans les deux sens
        score_map[(_norm(at), _norm(ht), d)] = (sa, sh)  # inversé

    resolved_count = 0
    skipped_count = 0
    details = []

    for i, row in df.iterrows():
        # Déjà résolu
        if str(row.get("result", "")).strip() in ("W", "L", "V"):
            skipped_count += 1
            continue

        d = str(row.get("date", ""))[:10]
        ht = str(row.get("home_team", ""))
        at = str(row.get("away_team", ""))
        market = str(row.get("market", ""))
        side = str(row.get("side", ""))

        # Chercher le score
        key = (_norm(ht), _norm(at), d)
        if key not in score_map:
            skipped_count += 1
            continue

        sh, sa = score_map[key]

        # Résoudre
        result = None
        if market.lower() == "team_total":
            result = resolve_team_total(sh, sa, side, ht, at)
        else:
            result = resolve_bet(sh, sa, market, side)

        if result is None:
            skipped_count += 1
            details.append(f"⚠️  {ht} vs {at} | {market}/{side} → non supporté")
            continue

        stake = float(row.get("stake_eur") or 0)
        cote = float(row.get("cote") or 1)
        profit = round(stake * (cote - 1) if result == "W" else (-stake if result == "L" else 0), 2)

        details.append(f"{'✅' if result=='W' else '❌'} {ht} vs {at} | {market}/{side} | {sh}-{sa} → {result} ({profit:+.2f}€)")

        if not dry_run:
            df.at[i, "result"] = result
            df.at[i, "profit_eur"] = profit

        resolved_count += 1

    if not dry_run and resolved_count > 0:
        df.to_csv(BETS_LOG, index=False)

    return {"resolved": resolved_count, "skipped": skipped_count, "details": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Résolus: {result['resolved']} | Skippés: {result['skipped']}")
    for line in result["details"]:
        print(" ", line)
