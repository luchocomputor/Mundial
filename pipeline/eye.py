"""Mode « Buteur à l'œil » — Louis logge ses pronos buteur repérés à l'œil
(les mobylettes obscures), le bot les RÉSOUT (a-t-il marqué ?) et les JUGE au CLV.

CLV buteur = mouvement BRUT de la cote du JOUEUR entre la prise et la clôture
Betclic (même joueur → la marge sur ce joueur est ~stable, donc le mouvement est
le signal ; ça contourne le problème de de-vig de l'anytime-buteur). Positif = la
cote a raccourci = le marché s'est rapproché de son pari.

But : tester si son flair bat le marché là où le modèle est aveugle (la technique,
la mobilité, le feu follet). C'est LUI le modèle, pas Dixon-Coles.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluation.clv import clv
from models.goalscorer import _nlast
from pipeline.fetch_betclic_grpc import match_odds
from pipeline.teams import normalize as _norm
from pipeline.value_detector import BETS_LOG

ROOT = Path(__file__).parent.parent
WC_PATH = ROOT / "data" / "raw" / "wc_all.parquet"
EYE_CLOSING = ROOT / "data" / "eye_closing.json"
PAPER_LOG = ROOT / "data" / "paper_trading.json"
UID = "19330064"          # x-bg-user (compte) pour le détail Betclic
_FINISHED_GRACE_H = 3.5    # un match dure ~2h ; après on ne re-snapshote plus


def _fixture_id(home: str, away: str, date: str) -> int | None:
    if not WC_PATH.exists():
        return None
    wc = pd.read_parquet(WC_PATH)
    hn, an, d = _norm(home), _norm(away), str(date)[:10]
    for _, r in wc.iterrows():
        if pd.isna(r.fixture_id):
            continue
        if str(pd.Timestamp(r.date).date()) == d and {_norm(r.home_team), _norm(r.away_team)} == {hn, an}:
            return int(r.fixture_id)
    return None


def log_eye_pick(home: str, away: str, date: str, player: str, odds: float,
                 stake: float, betclic_match_id: int | None = None,
                 p_model: float | None = None) -> bool:
    """Logge un prono buteur à l'œil en paper. Dédupe (fixture, joueur)."""
    fid = _fixture_id(home, away, date)
    side = _nlast(player)
    bet = {
        "fixture_id": fid, "date": str(date)[:10], "home_team": home, "away_team": away,
        "market": "buteur", "side": side, "cote": float(odds), "odds_taken": float(odds),
        "p_model": round(p_model, 4) if p_model is not None else "",
        "edge": "", "stake_eur": float(stake), "bankroll": 200.0, "confidence": 2,
        "model_version": "oeil", "anchor_weight": 0.0, "paper_trade": True,
        "betclic_match_id": int(betclic_match_id) if betclic_match_id else "",
        "notes": f"Œil — {player} buteur @ {odds}",
        "detected_at": datetime.now().isoformat(), "result": "", "profit_eur": "",
    }
    df = pd.read_csv(BETS_LOG) if BETS_LOG.exists() else pd.DataFrame()
    if not df.empty and "fixture_id" in df.columns and fid is not None:
        dup = ((df.fixture_id == fid) & (df.market == "buteur") &
               (df.side.astype(str) == side) & (df.model_version == "oeil"))
        if dup.any():
            return False
    cols = list(dict.fromkeys((list(df.columns) if not df.empty else []) + list(bet)))
    df = pd.concat([df, pd.DataFrame([bet])], ignore_index=True).reindex(columns=cols)
    df.to_csv(BETS_LOG, index=False)
    return True


# ── CLV buteur (mouvement de la cote du joueur) ─────────────────────────────
def _load(p):
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def snapshot_eye_closing() -> int:
    """Fige la cote buteur Betclic des joueurs des pronos œil en attente, tant
    que le match n'a pas démarré (converge vers la clôture)."""
    if not BETS_LOG.exists():
        return 0
    df = pd.read_csv(BETS_LOG)
    eye = df[df.get("model_version") == "oeil"] if "model_version" in df.columns else df.iloc[0:0]
    if eye.empty:
        return 0
    store = _load(EYE_CLOSING)
    now = pd.Timestamp.now(tz="UTC")
    # un seul fetch gRPC par match (plusieurs joueurs possibles)
    by_match: dict[int, list] = {}
    for _, r in eye.iterrows():
        bid = r.get("betclic_match_id")
        if pd.isna(bid) or str(bid).strip() in ("", "nan"):
            continue
        by_match.setdefault(int(bid), []).append(r)
    n = 0
    for bid, rows in by_match.items():
        try:
            scorers = match_odds(bid, UID).get("scorers") or {}
        except Exception:
            continue
        if not scorers:           # match commencé / fini → cote déjà figée, on garde
            continue
        bylast = {_nlast(p): o for p, o in scorers.items()}
        for r in rows:
            o = bylast.get(str(r["side"]))
            if o and o > 1:
                store[f"{bid}|{r['side']}"] = {"odds": float(o), "captured_at": now.isoformat()}
                n += 1
    EYE_CLOSING.parent.mkdir(parents=True, exist_ok=True)
    EYE_CLOSING.write_text(json.dumps(store, ensure_ascii=False, indent=2))
    return n


def fetch_eye_clv() -> int:
    """Remplit le CLV brut des pronos œil dont le match a démarré (cote figée)."""
    if not BETS_LOG.exists():
        return 0
    df = pd.read_csv(BETS_LOG)
    if "model_version" not in df.columns:
        return 0
    store = _load(EYE_CLOSING)
    updated = 0
    for idx in df.index:
        if df.loc[idx, "model_version"] != "oeil":
            continue
        if str(df.loc[idx].get("clv", "")).strip() not in ("", "nan"):
            continue
        bid = df.loc[idx].get("betclic_match_id")
        if pd.isna(bid) or str(bid).strip() in ("", "nan"):
            continue
        entry = store.get(f"{int(bid)}|{df.loc[idx, 'side']}")
        if not entry:
            continue
        taken = float(df.loc[idx, "odds_taken"])
        close = float(entry["odds"])
        df.loc[idx, "odds_close"] = close
        df.loc[idx, "clv"] = clv(taken, close)
        updated += 1
    if updated:
        df.to_csv(BETS_LOG, index=False)
    return updated
