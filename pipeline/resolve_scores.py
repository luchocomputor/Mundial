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


# ── Résolution riche (score + buteurs) « comme un humain » ─────────────────────
_DC_SIDES = {
    "home_or_draw": {"home", "draw"}, "1x": {"home", "draw"},
    "home_or_away": {"home", "away"}, "12": {"home", "away"},
    "draw_or_away": {"draw", "away"}, "x2": {"draw", "away"},
}


def _scorer_query(side: str):
    """side buteur → (tokens joueurs, mode 'any'/'all', remplaçant_requis)."""
    s = side.lower().strip()
    mode = "all" if "_et_" in s else "any"
    parts = re.split(r"_et_|_ou_|_/_|_\+_", s)
    tokens = [p.strip() for p in parts if p.strip()]
    needs_sub = any(t in ("remplacant", "supersub", "rempl") for t in tokens)
    tokens = [t for t in tokens if t not in ("remplacant", "supersub", "rempl")]
    return tokens, mode, needs_sub


def resolve_bet_facts(f, market: str, side: str) -> str | None:
    """Résout un pari depuis les MatchFacts. Retourne 'W'/'L'/'V' ou None si le
    marché n'est pas résoluble de façon fiable (→ reste en attente, jamais à tort)."""
    from pipeline.match_facts import norm_player

    m, s = market.lower().strip(), side.lower().strip()
    sh, sa = f.sh, f.sa
    res_1x2 = "home" if sh > sa else ("draw" if sh == sa else "away")

    # ── basés sur le score ──
    if m in ("1x2", "1×2"):
        return "W" if s == res_1x2 else "L"
    if m == "double_chance":
        want = _DC_SIDES.get(s)
        return None if want is None else ("W" if res_1x2 in want else "L")
    if m == "btts":
        return "W" if (s == "yes") == (sh > 0 and sa > 0) else "L"
    if m == "ht_result":
        if f.sh_ht is None or f.sa_ht is None:
            return None
        ht = "home" if f.sh_ht > f.sa_ht else ("draw" if f.sh_ht == f.sa_ht else "away")
        return "W" if s == ht else "L"
    if m in ("exact_score", "score_exact"):
        mo = re.match(r"(\d+)\D+(\d+)", s)
        return None if not mo else ("W" if (sh, sa) == (int(mo.group(1)), int(mo.group(2))) else "L")
    if m == "team_total":
        return resolve_team_total(sh, sa, side, f.home_team, f.away_team)
    ou = re.match(r"(over|under)_(\d+(?:\.\d+)?)", m)
    if ou:
        over = (sh + sa) > float(ou.group(2))
        return "W" if (side.lower() == "over") == over else "L"

    # ── basés sur les buteurs ──
    if m == "penalty_scored":
        scored_pen = any(g.get("goal_type") == "penalty" for g in f.scorers)
        # side peut nommer un joueur précis, sinon n'importe quel penalty
        toks, _, _ = _scorer_query(s)
        if toks and toks != ["yes"]:
            scored_pen = any(g.get("goal_type") == "penalty"
                             and norm_player(g.get("player", "")) in toks for g in f.scorers)
        return "W" if scored_pen else "L"
    if m in ("buteur", "buteur_rempl", "buteur_supersub", "buteur_dc"):
        toks, mode, needs_sub = _scorer_query(s)
        if not toks:
            return None
        scored = f.scorer_names()  # noms normalisés des buteurs (hors csc)
        if mode == "all":  # "X et Y marquent"
            return "W" if all(t in scored for t in toks) else "L"
        # "any" : un des joueurs nommés — + leur remplaçant si "ou remplaçant"
        targets = set(toks)
        if needs_sub:
            for sub in f.subs:
                if norm_player(sub.get("player_out", "")) in toks and sub.get("player_in"):
                    targets.add(norm_player(sub["player_in"]))
        return "W" if (targets & scored) else "L"

    # marchés non résolus de façon fiable (tête, joueur décisif, période, combo…)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def _event_index(events: list[dict]) -> dict[tuple, int]:
    """(home normalisé, away normalisé, date MTL) → fixture_id, dans les 2 sens."""
    idx: dict[tuple, int] = {}
    for e in events:
        eid = e.get("id")
        if eid is None:
            continue
        d = mtl_date(e["event_date"])
        hn, an = _norm(e["home_team"]), _norm(e["away_team"])
        idx[(hn, an, d)] = int(eid)
        idx.setdefault((an, hn, d), int(eid))
    return idx


def run(dry_run: bool = False) -> dict:
    """Auto-résolution riche : pour chaque pari en attente dont le match est joué,
    on récupère les MatchFacts (score, mi-temps, buteurs, entrants) et on résout —
    1X2, double chance, BTTS, O/U, team total, score exact, mi-temps, buteur(s),
    penalty, buteur remplaçant. Les marchés ambigus (tête, combo, joueur décisif)
    restent en attente plutôt que résolus à tort."""
    from pipeline.match_facts import get_match_facts

    if not BETS_LOG.exists():
        return {"resolved": 0, "skipped": 0, "details": []}
    df = pd.read_csv(BETS_LOG)
    if df.empty:
        return {"resolved": 0, "skipped": 0, "details": []}

    events = fetch_finished_scores()
    ev_idx = _event_index(events)
    facts_cache: dict[int, object] = {}

    resolved_count = skipped_count = 0
    details = []

    for i, row in df.iterrows():
        if str(row.get("result", "")).strip() in ("W", "L", "V"):
            skipped_count += 1
            continue
        d = str(row.get("date", ""))[:10]
        ht, at = str(row.get("home_team", "")), str(row.get("away_team", ""))
        market, side = str(row.get("market", "")), str(row.get("side", ""))

        # fixture_id : du log si présent, sinon via l'index (équipes + date)
        fid = row.get("fixture_id")
        fid = int(fid) if pd.notna(fid) and str(fid).strip() not in ("", "nan") else ev_idx.get((_norm(ht), _norm(at), d))
        if fid is None:
            skipped_count += 1
            continue

        if fid not in facts_cache:
            facts_cache[fid] = get_match_facts(fid)
        facts = facts_cache[fid]
        if facts is None or not facts.finished:
            skipped_count += 1
            continue

        result = resolve_bet_facts(facts, market, side)
        if result is None:
            skipped_count += 1
            details.append(f"⏳ {ht} vs {at} | {market}/{side} → à confirmer (non auto-résoluble)")
            continue

        stake = float(row.get("stake_eur") or 0)
        cote = float(row.get("cote") or 1)
        profit = round(stake * (cote - 1) if result == "W" else (-stake if result == "L" else 0), 2)
        details.append(f"{'✅' if result=='W' else '❌'} {ht} vs {at} | {market}/{side} | "
                       f"{facts.sh}-{facts.sa} → {result} ({profit:+.2f}€)")
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
