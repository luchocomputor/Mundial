"""Faits d'un match terminé — score, mi-temps, buteurs, entrants — pour
l'auto-résolution des paris « comme un humain ».

Source : API bzzoiro (league_id=27). `/events/{id}` donne le score final + le
score à la mi-temps + le penalty shootout ; `/events/{id}/incidents` donne les
buts (buteur, minute, type penalty/csc, équipe) et les remplacements (qui entre).
Une fois un match terminé ses faits ne changent plus → cache disque permanent
(data/match_facts.json), on ne re-fetch que les matchs absents du cache.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import requests

from pipeline.config import bzzoiro_headers, load_config

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "match_facts.json"
_FINISHED = {"finished", "ft", "aet", "pen"}


def norm_player(name: str) -> str:
    """Nom joueur → nom de famille normalisé (sans accent, minuscule).
    'K. Mbappé' → 'mbappe' ; 'Gyökeres' → 'gyokeres'. Sert au matching buteurs."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", " ").replace("-", " ").strip()
    parts = [p for p in s.split() if len(p) > 1]  # vire les initiales "k"
    return parts[-1] if parts else s


@dataclass
class MatchFacts:
    fixture_id: int
    home_team: str
    away_team: str
    status: str
    sh: int
    sa: int
    sh_ht: int | None = None
    sa_ht: int | None = None
    penalty_shootout: bool = False
    scorers: list[dict] = field(default_factory=list)   # {player, player_id, minute, is_home, goal_type}
    subs: list[dict] = field(default_factory=list)       # {player_in(_id), player_out(_id), is_home, minute}

    @property
    def finished(self) -> bool:
        return (self.status or "").lower() in _FINISHED

    @property
    def subs_in(self) -> set[str]:
        """Noms normalisés des joueurs ENTRÉS en jeu (remplaçants)."""
        return {norm_player(s["player_in"]) for s in self.subs if s.get("player_in")}

    def scored(self, player: str, own_goal_counts: bool = False) -> bool:
        """Le joueur (nom approx.) a-t-il marqué un but valide pour lui ?"""
        q = norm_player(player)
        for g in self.scorers:
            if g.get("goal_type") == "ownGoal" and not own_goal_counts:
                continue
            if norm_player(g.get("player", "")) == q:
                return True
        return False

    def scorer_names(self) -> set[str]:
        return {norm_player(g.get("player", "")) for g in self.scorers if g.get("goal_type") != "ownGoal"}

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MatchFacts":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


# ── Cache ────────────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text()) if CACHE.exists() else {}
    except Exception:
        return {}


def _save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=2))


# ── Fetch ────────────────────────────────────────────────────────────────────
def _fetch(fixture_id: int, cfg, h) -> MatchFacts | None:
    base = cfg.bzzoiro_base_url
    ev = requests.get(f"{base}/events/{fixture_id}", headers=h, timeout=10).json()
    if ev.get("home_score") is None or ev.get("away_score") is None:
        return None
    inc = requests.get(f"{base}/events/{fixture_id}/incidents", headers=h, timeout=10).json().get("incidents", [])
    scorers, subs = [], []
    for i in inc:
        t = i.get("type")
        if t == "goal":
            scorers.append({k: i.get(k) for k in ("player", "player_id", "minute", "added_time", "is_home", "goal_type", "assist")})
        elif t == "substitution":
            subs.append({k: i.get(k) for k in ("player_in", "player_in_id", "player_out", "player_out_id", "is_home", "minute")})
    return MatchFacts(
        fixture_id=int(fixture_id), home_team=ev.get("home_team", ""), away_team=ev.get("away_team", ""),
        status=ev.get("status", ""), sh=int(ev["home_score"]), sa=int(ev["away_score"]),
        sh_ht=ev.get("home_score_ht"), sa_ht=ev.get("away_score_ht"),
        penalty_shootout=bool(ev.get("penalty_shootout")), scorers=scorers, subs=subs,
    )


def get_match_facts(fixture_id: int, cfg=None, force: bool = False) -> MatchFacts | None:
    """Faits d'un match (caché si déjà terminé). None si introuvable/non terminé."""
    key = str(int(fixture_id))
    cache = _load_cache()
    if not force and key in cache:
        return MatchFacts.from_dict(cache[key])
    if cfg is None:
        cfg = load_config()
    try:
        facts = _fetch(fixture_id, cfg, bzzoiro_headers(cfg))
    except Exception:
        return None
    if facts is None:
        return None
    if facts.finished:  # on ne cache que le définitif
        cache[key] = facts.to_dict()
        _save_cache(cache)
    return facts


if __name__ == "__main__":
    import sys
    fid = int(sys.argv[1]) if len(sys.argv) > 1 else 8308
    f = get_match_facts(fid, force=True)
    if f:
        print(f"{f.home_team} {f.sh}-{f.sa} {f.away_team} (MT {f.sh_ht}-{f.sa_ht}) · {f.status}")
        for g in f.scorers:
            sub = " (entré)" if norm_player(g["player"]) in f.subs_in else ""
            print(f"  {g['minute']}' {g['player']} [{g['goal_type']}] {'dom' if g['is_home'] else 'ext'}{sub}")
