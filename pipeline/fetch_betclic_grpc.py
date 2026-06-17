"""Fetcher cotes Betclic via son API interne gRPC-Web (begmedia).

The Odds API ne donne que le 1X2 Betclic. L'API interne (offering.begmedia.com,
MatchService/GetMatchWithNotification) expose TOUS les marchés — 1X2, Plus/Moins
de buts (toutes lignes), BTTS, score exact, buteurs… Le transport est gRPC-Web
(framing 5 octets + protobuf), la réponse un STREAM (snapshot + maj live) : on lit
seulement la 1ère trame (le snapshot). Pas d'auth/token, juste l'ID compte
(x-bg-user) + headers de marque.

Reverse-engineeré sans .proto : la cote d'un outcome = champ 12 (double), son
libellé = champ 10/11 (string). Marché identifié par les libellés de ses outcomes
(robuste : 1X2 = {home, "Nul", away} ; O/U = "+ de X,5"/"- de X,5").

⚠️ Production : nécessite l'ID Betclic du match (varint dans la requête). Pour
enumérer les matchs CDM il faut l'endpoint "liste" (capture DevTools séparée).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

import requests

HOST = "https://offering.begmedia.com"
PATH = "/web/offering.access.api/offering.access.api.MatchService/GetMatchWithNotification"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/26.3.1 Safari/605.1.15")


def _headers(user_id: str) -> dict:
    return {
        "Content-Type": "application/grpc-web+proto", "Origin": "https://www.betclic.fr",
        "Referer": "https://www.betclic.fr/", "Accept": "*/*", "User-Agent": UA,
        "x-bg-ref-brand": "BETCLIC", "x-bg-ref-platform": "DESKTOP",
        "x-bg-ref-regulator-zone": "FR", "x-bg-regulation": "FR",
        "x-bg-user": str(user_id), "x-grpc-web": "1",
    }


# ── protobuf wire-format (decode_raw maison) ────────────────────────────────
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(b: bytes, i: int) -> tuple[int, int]:
    shift = res = 0
    while True:
        x = b[i]; res |= (x & 0x7F) << shift; i += 1
        if not x & 0x80:
            return res, i
        shift += 7


def _parse(b: bytes) -> list[tuple[int, int, object]]:
    """[(field_num, wire_type, value)] — value brute (int / bytes 8|4|len)."""
    i, out = 0, []
    while i < len(b):
        try:
            tag, i = _read_varint(b, i)
            fn, wt = tag >> 3, tag & 7
            if wt == 0:
                v, i = _read_varint(b, i); out.append((fn, 0, v))
            elif wt == 1:
                out.append((fn, 1, b[i:i + 8])); i += 8
            elif wt == 2:
                ln, i = _read_varint(b, i); out.append((fn, 2, b[i:i + ln])); i += ln
            elif wt == 5:
                out.append((fn, 5, b[i:i + 4])); i += 4
            else:
                break
        except Exception:
            break
    return out


def _as_str(v: bytes) -> str | None:
    try:
        s = v.decode("utf-8")
    except Exception:
        return None
    return s if s.isprintable() else None


# ── extraction des marchés ──────────────────────────────────────────────────
def _outcomes(b: bytes, depth: int = 0, acc: list | None = None) -> list[tuple[str, float]]:
    """(libellé, cote) de tout le sous-arbre : un noeud-outcome a un champ 12
    (double, la cote) + un champ 10/11 (string, le libellé)."""
    if acc is None:
        acc = []
    if depth > 10:
        return acc
    flds = _parse(b)
    label = odd = None
    for fn, wt, v in flds:
        if wt == 1 and fn == 12:
            try:
                odd = round(struct.unpack("<d", v)[0], 3)
            except Exception:
                pass
        elif wt == 2 and fn in (10, 11):
            s = _as_str(v)
            if s and not s.startswith("contestant:"):
                label = s
    if label and odd and 1.0 < odd < 10000:
        acc.append((label, odd))
    for fn, wt, v in flds:
        if wt == 2 and len(v) >= 2 and _as_str(v) is None:
            _outcomes(v, depth + 1, acc)
    return acc


def _market_name(b: bytes, depth: int = 0) -> str | None:
    """Nom-string du marché s'il existe (certains n'ont qu'un type numérique)."""
    if depth > 3:
        return None
    for fn, wt, v in _parse(b):
        if wt == 2:
            s = _as_str(v)
            if s and not s.startswith("contestant:") and 3 < len(s) < 45 and " " in s.strip():
                return s
    return None


@dataclass
class BetclicMarkets:
    teams: tuple[str, str] | None = None
    markets: list[dict] = field(default_factory=list)  # [{name, outcomes:[(label,odd)]}]

    def flat(self, home: str, away: str) -> dict:
        """→ format nested du bot : {"1X2": {...}, "over_2.5": {...}, "btts": {...}}.
        home/away = noms canoniques attendus (orientation wc_all)."""
        out: dict[str, dict] = {}
        for m in self.markets:
            od = dict(m["outcomes"])
            labels = set(od)
            name = (m.get("name") or "")
            # 1X2 : {home, Nul, away}
            if "Nul" in labels and home in labels and away in labels and "1X2" not in out:
                out["1X2"] = {"home": od[home], "draw": od["Nul"], "away": od[away]}
            # O/U total du MATCH : "Nombre total de buts" (exclut les totals par
            # équipe "Total de buts - <équipe>", qui contiennent " - ").
            if "total" in name.lower() and " - " not in name and any(l.startswith("+ de ") for l in labels):
                for l, o in od.items():
                    mo = re.match(r"([+-]) de (\d+),5", l)
                    if mo:
                        line = f"over_{mo.group(2)}.5"
                        side = "over" if mo.group(1) == "+" else "under"
                        out.setdefault(line, {})[side] = o
        return out


def parse_markets(snapshot: bytes) -> BetclicMarkets:
    res = BetclicMarkets()
    top = _parse(snapshot)
    f1 = next((v for fn, wt, v in top if fn == 1 and wt == 2), None)
    if f1 is None:
        return res
    inner = next((v for fn, wt, v in _parse(f1) if fn == 1 and wt == 2), None)
    if inner is None:
        return res
    for fn, wt, v in _parse(inner):
        if not (fn == 11 and wt == 2):
            continue
        for f, w, mk in _parse(v):
            if f != 3 or w != 2:
                continue
            outs = _outcomes(mk)
            if outs:
                res.markets.append({"name": _market_name(mk), "outcomes": outs})
    return res


def fetch_match_snapshot(match_id: int, user_id: str, timeout: float = 30) -> bytes:
    """POST gRPC-Web puis lit la 1ère trame (snapshot protobuf)."""
    payload = b"\x08" + _varint(match_id) + b"\x12\x02" + b"fr"
    frame = b"\x00" + len(payload).to_bytes(4, "big") + payload
    r = requests.post(HOST + PATH, data=frame, headers=_headers(user_id),
                      stream=True, timeout=(10, timeout))
    r.raise_for_status()
    raw = b""
    msglen = None
    try:
        for chunk in r.iter_content(8192):
            raw += chunk
            if msglen is None and len(raw) >= 5:
                msglen = int.from_bytes(raw[1:5], "big")
            if msglen is not None and len(raw) >= 5 + msglen:
                break
    finally:
        r.close()
    return raw[5:5 + (msglen or 0)]


def fetch_match_markets(match_id: int, user_id: str) -> BetclicMarkets:
    return parse_markets(fetch_match_snapshot(match_id, user_id))


if __name__ == "__main__":
    import sys
    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 543844252606823  # England-Croatia
    uid = sys.argv[2] if len(sys.argv) > 2 else "19330064"
    bm = fetch_match_markets(mid, uid)
    print(f"{len(bm.markets)} marchés")
    for m in bm.markets:
        outs = m["outcomes"]
        if len(outs) <= 14:
            print(f"  {m['name'] or '?':28s} " + ", ".join(f"{l}={o}" for l, o in outs[:6]))
