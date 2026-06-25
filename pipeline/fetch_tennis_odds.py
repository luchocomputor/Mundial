"""Cotes tennis Betclic (même API gRPC-Web interne que le foot).

Reverse-engineeré depuis un snapshot réel (WTA, juin 2026). Marchés tennis :
  - « Vainqueur du match »   : 2-way {joueur A: cote, joueur B: cote}  ← ce que l'Elo prédit
  - « Nombre total de jeux » : O/U total jeux (+ de 17,5 / - de 17,5 …)
  - « Score final (sets) », « Écart de sets », « Vainqueur du set » : non utilisés (v1)

Les compétitions tennis sont des entiers (cid) passés au même endpoint liste que la
CDM (field1=cid). Ils TOURNENT par tournoi/semaine → on les redécouvre par signature
de marché plutôt que de les coder en dur. cid connu (grass 2026) : 177 (WTA).

    python -m pipeline.fetch_tennis_odds --discover 160 260   # trouve les cid tennis
    python -m pipeline.fetch_tennis_odds                      # liste matchs + cotes vainqueur
"""

from __future__ import annotations

import argparse

import requests

from pipeline.fetch_betclic_grpc import (HOST, LIST_PATH, _headers, _varint,
                                         fetch_match_markets, parse_match_list)

UID = "19330064"
MK_WINNER = "Vainqueur du match"
MK_GAMES = "Nombre total de jeux"
MK_SETS = "Score final (sets)"
# cid tennis connus (à rafraîchir via --discover : ils changent chaque semaine).
KNOWN_TENNIS_CIDS = [177]


def list_competition(cid: int, timeout: float = 15) -> list[dict]:
    """Matchs d'une compétition arbitraire (générique, ré-implémente fetch_match_list
    qui est codé en dur sur la CDM)."""
    payload = b"\x08" + _varint(cid) + b"\x1a\x02" + b"fr"
    frame = b"\x00" + len(payload).to_bytes(4, "big") + payload
    h = _headers("")
    h.pop("x-bg-user", None)
    r = requests.post(HOST + LIST_PATH, data=frame, headers=h, stream=True, timeout=(10, timeout))
    r.raise_for_status()
    raw, ml = b"", None
    try:
        for c in r.iter_content(8192):
            raw += c
            if ml is None and len(raw) >= 5:
                ml = int.from_bytes(raw[1:5], "big")
            if ml is not None and len(raw) >= 5 + ml:
                break
    finally:
        r.close()
    return parse_match_list(raw[5:5 + (ml or 0)])


def winner_odds(markets) -> dict | None:
    """{(A, B): {A: cote, B: cote}} depuis le marché « Vainqueur du match » (2-way)."""
    for m in markets.markets:
        if (m.get("name") or "") == MK_WINNER and len(m["outcomes"]) == 2:
            (a, oa), (b, ob) = m["outcomes"]
            return {"players": (a, b), "odds": {a: oa, b: ob}}
    return None


def total_games(markets) -> dict:
    """O/U total jeux → {17.5: {over, under}, …}."""
    import re
    out: dict[float, dict] = {}
    for m in markets.markets:
        if (m.get("name") or "") != MK_GAMES:
            continue
        for label, odd in m["outcomes"]:
            mo = re.match(r"([+-]) de (\d+),5", label)
            if mo:
                out.setdefault(float(f"{mo.group(2)}.5"), {})[
                    "over" if mo.group(1) == "+" else "under"] = odd
    return out


def is_tennis(markets) -> bool:
    """Signature tennis : un « Vainqueur du match » 2-way + un marché sets/jeux."""
    names = {(m.get("name") or "") for m in markets.markets}
    has_winner = any((m.get("name") or "") == MK_WINNER and len(m["outcomes"]) == 2
                     for m in markets.markets)
    return has_winner and bool(names & {MK_GAMES, MK_SETS})


def match_winner_odds(match_id: int, uid: str = UID) -> dict | None:
    return winner_odds(fetch_match_markets(match_id, uid))


def discover_tennis(cid_lo: int, cid_hi: int, uid: str = UID) -> list[int]:
    """Scanne [cid_lo, cid_hi] et confirme le tennis par signature de marché sur le
    1er match de chaque compétition. Lourd (2 requêtes/cid) → plages ciblées."""
    found = []
    for cid in range(cid_lo, cid_hi + 1):
        try:
            ms = list_competition(cid)
            if not ms:
                continue
            if is_tennis(fetch_match_markets(ms[0]["match_id"], uid)):
                found.append(cid)
                print(f"  ✓ cid={cid} tennis : {ms[0]['label']} ({len(ms)} matchs)")
        except Exception:
            continue
    return found


def list_tennis_matches(cids: list[int] | None = None, uid: str = UID) -> list[dict]:
    """[{cid, match_id, label, players, date, winner_odds}] pour les compétitions
    tennis données (défaut : KNOWN_TENNIS_CIDS)."""
    out = []
    for cid in (cids or KNOWN_TENNIS_CIDS):
        try:
            ms = list_competition(cid)
        except Exception:
            continue
        for m in ms:
            try:
                wo = match_winner_odds(m["match_id"], uid)
            except Exception:
                wo = None
            out.append({"cid": cid, "match_id": m["match_id"], "label": m["label"],
                        "players": m["teams_fr"], "date": m["date"],
                        "winner_odds": wo["odds"] if wo else None})
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Cotes tennis Betclic")
    ap.add_argument("--discover", nargs=2, type=int, metavar=("LO", "HI"))
    args = ap.parse_args()
    if args.discover:
        print(f"Découverte tennis cid {args.discover[0]}..{args.discover[1]} …")
        cids = discover_tennis(*args.discover)
        print(f"\ncid tennis : {cids}")
        return
    for m in list_tennis_matches():
        od = m["winner_odds"]
        s = " | ".join(f"{p} {c}" for p, c in od.items()) if od else "(cotes indispo)"
        print(f"  [{m['cid']}] {m['label']:42} {m['date'][:16]}  →  {s}")


if __name__ == "__main__":
    _cli()
