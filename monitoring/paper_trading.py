"""Paper trading — enregistre paris simulés, CLV post-clôture, rolling 7j/30j."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from evaluation.clv import bootstrap_clv_significance, clv
from pipeline.value_detector import BETS_LOG, log_bets

ROOT = Path(__file__).parent.parent
PAPER_LOG = ROOT / "data" / "paper_trading.json"
WC_PATH = ROOT / "data" / "raw" / "wc_all.parquet"

# Lignes de clôture : on fige la dernière cote SHARP (Pinnacle) AVANT le coup
# d'envoi de chaque match → odds_close fiable pour le CLV, sans dépendre d'un
# appel API live au kickoff (l'ancienne voie bzzoiro était muette → CLV jamais
# peuplé). Le CLV vs Pinnacle est le KPI d'edge réel du bot.
SHARP_FLAT = ROOT / "data" / "raw" / "pinnacle_odds_flat.json"
SOFT_FLAT = ROOT / "data" / "raw" / "betclic_odds_flat.json"
CLOSING_LINES = ROOT / "data" / "closing_lines.json"


def _logged_keys() -> set:
    """(fixture_id, market, side) déjà présents dans le bets_log."""
    if not BETS_LOG.exists():
        return set()
    try:
        df = pd.read_csv(BETS_LOG)
    except Exception:
        return set()
    keys = set()
    for _, r in df.iterrows():
        fid = r.get("fixture_id")
        if pd.isna(fid):
            continue
        keys.add((str(int(fid)), str(r.get("market")), str(r.get("side"))))
    return keys


def record_paper_bets(bets: list[dict]) -> int:
    """Enregistre les paris paper en DÉDUPLIQUANT sur (fixture_id, market, side) :
    on garde la première détection (odds_taken = le prix qu'on aurait pris), les
    re-détections des refresh suivants sont ignorées — sinon le bets_log double à
    chaque run et le CLV/ROI sont fausses. Retourne le nombre de nouveaux paris."""
    seen = _logged_keys()
    new = []
    for bet in bets:
        fid = bet.get("fixture_id")
        if fid is None:
            continue
        key = (str(int(fid)), str(bet.get("market")), str(bet.get("side")))
        if key in seen:
            continue
        seen.add(key)
        new.append(bet)
    if not new:
        return 0

    for bet in new:
        bet["paper_trade"] = True
        bet["recorded_at"] = datetime.now().isoformat()
    log_bets(new)

    existing = []
    if PAPER_LOG.exists():
        existing = json.loads(PAPER_LOG.read_text())
    existing.extend(new)
    PAPER_LOG.write_text(json.dumps(existing, indent=2, default=str))
    return len(new)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


def _kickoffs() -> dict[int, pd.Timestamp]:
    """fixture_id → datetime du coup d'envoi (depuis wc_all)."""
    if not WC_PATH.exists():
        return {}
    wc = pd.read_parquet(WC_PATH)
    wc = wc[wc["fixture_id"].notna()]
    ko = pd.to_datetime(wc["date"], utc=True)
    return {int(f): t for f, t in zip(wc["fixture_id"], ko)}


def snapshot_closing_lines() -> int:
    """Fige la cote sharp (Pinnacle, fallback soft) de chaque match À VENIR.
    Appelé à chaque refresh → la valeur converge vers la cote de clôture ; dès que
    le match démarre on cesse de le re-snapshoter, la dernière valeur reste = close.
    Retourne le nombre de matchs (re)snapshotés."""
    kicks = _kickoffs()
    now = pd.Timestamp.now(tz="UTC")
    pinn, betc = _load_json(SHARP_FLAT), _load_json(SOFT_FLAT)
    store = _load_json(CLOSING_LINES)

    n = 0
    for fid in set(pinn) | set(betc):
        if not str(fid).lstrip("-").isdigit():
            continue
        ko = kicks.get(int(fid))
        if ko is None or ko <= now:        # kickoff inconnu ou déjà passé → on ne touche pas (gel)
            continue
        if not (pinn.get(fid) or betc.get(fid)):
            continue
        store[str(fid)] = {
            "pinnacle": pinn.get(fid),   # ancre sharp → CLV d'edge
            "betclic": betc.get(fid),    # book de mise → CLV de ligne
            "captured_at": now.isoformat(),
            "kickoff": ko.isoformat(),
        }
        n += 1

    CLOSING_LINES.parent.mkdir(parents=True, exist_ok=True)
    CLOSING_LINES.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    return n


def update_clv_after_close(fixture_id: int, odds_close: float) -> None:
    if not BETS_LOG.exists():
        return
    df = pd.read_csv(BETS_LOG)
    mask = (df["fixture_id"] == fixture_id) & (df["odds_close"].isna() | (df["odds_close"] == ""))
    if not mask.any():
        return
    for idx in df[mask].index:
        odds_taken = float(df.loc[idx, "odds_taken"] or df.loc[idx, "cote"])
        df.loc[idx, "odds_close"] = odds_close
        df.loc[idx, "clv"] = clv(odds_taken, odds_close)
    df.to_csv(BETS_LOG, index=False)


def fetch_and_update_clv(fixture_ids: list[int] | None = None) -> int:
    """Remplit odds_close + CLV des paris en attente dont le match a démarré,
    depuis les lignes de clôture figées par snapshot_closing_lines. Le CLV est
    mesuré contre la cote sharp (Pinnacle) — le KPI d'edge réel du bot."""
    if not BETS_LOG.exists():
        return 0
    df = pd.read_csv(BETS_LOG)
    # Besoin du fixture_id pour rattacher la ligne de clôture. Les anciens logs
    # n'en ont pas → on saute proprement (le CLV se peuplera sur les paris récents).
    if "clv" not in df.columns or "fixture_id" not in df.columns:
        return 0
    store = _load_json(CLOSING_LINES)
    if not store:
        return 0
    now = pd.Timestamp.now(tz="UTC")

    def _empty(col):
        return df[col].isna() | (df[col].astype(str).str.strip() == "") if col in df.columns else True
    # En attente tant qu'AUCUN des deux CLV (Pinnacle/Betclic) n'est rempli.
    mask = _empty("clv") & _empty("clv_betclic")
    pending = df[mask] if mask is not True else df
    if fixture_ids:
        pending = pending[pending["fixture_id"].isin(fixture_ids)]
    if pending.empty:
        return 0

    def _close(entry, book, market, side):
        o = ((entry.get(book) or {}).get(market) or {}).get(side)
        return float(o) if o and float(o) > 1.0 else None

    updated = 0
    for idx in pending.index:
        fid = df.loc[idx, "fixture_id"]
        if pd.isna(fid):
            continue
        entry = store.get(str(int(fid)))
        if not entry or pd.Timestamp(entry["kickoff"]) > now:  # close pas encore figée
            continue
        market, side = str(df.loc[idx, "market"]), str(df.loc[idx, "side"])
        taken = df.loc[idx, "odds_taken"]
        taken = float(taken) if pd.notna(taken) and str(taken).strip() else float(df.loc[idx, "cote"])

        pinn = _close(entry, "pinnacle", market, side)  # CLV d'edge (sharp)
        betc = _close(entry, "betclic", market, side)   # CLV de ligne (book de mise)
        if pinn is None and betc is None:
            continue
        if pinn is not None:
            df.loc[idx, "odds_close"] = pinn
            df.loc[idx, "clv"] = clv(taken, pinn)
        if betc is not None:
            df.loc[idx, "betclic_close"] = betc
            df.loc[idx, "clv_betclic"] = clv(taken, betc)
        updated += 1

    if updated:
        df.to_csv(BETS_LOG, index=False)
    return updated


def _rolling_clv(df: pd.DataFrame, days: int) -> dict:
    if "detected_at" not in df.columns or "clv" not in df.columns:
        return {"n": 0, "clv_mean": None}
    df = df.copy()
    df["detected_at"] = pd.to_datetime(df["detected_at"], errors="coerce")
    df["clv"] = pd.to_numeric(df["clv"], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days)
    recent = df[(df["detected_at"] >= cutoff) & df["clv"].notna()]
    if recent.empty:
        return {"n": 0, "clv_mean": None}
    return {"n": len(recent), "clv_mean": round(float(recent["clv"].mean()), 4)}


def paper_trading_report() -> dict:
    if not BETS_LOG.exists():
        return {"status": "no_data"}

    df = pd.read_csv(BETS_LOG)
    if "paper_trade" in df.columns:
        paper = df[df["paper_trade"] == True]
    else:
        paper = df

    clv_vals = pd.to_numeric(paper.get("clv", pd.Series()), errors="coerce").dropna()
    result: dict = {
        "n_bets": len(paper),
        "clv_7d": _rolling_clv(paper, 7),
        "clv_30d": _rolling_clv(paper, 30),
    }

    # CLV vs Betclic (book de mise) — secondaire, mouvement de ligne au book.
    betc_vals = pd.to_numeric(paper.get("clv_betclic", pd.Series()), errors="coerce").dropna()
    if not betc_vals.empty:
        result["clv_betclic_mean"] = round(float(betc_vals.mean()), 4)
        result["clv_betclic_n"] = int(len(betc_vals))

    if clv_vals.empty:
        result["clv"] = "pending"
        return result

    report = bootstrap_clv_significance(clv_vals)  # CLV vs Pinnacle = edge réel
    result.update({
        "clv_mean": report.mean_clv,
        "beat_close_pct": report.beat_close_pct,
        "significant": report.significant,
        "ci": [report.ci_lower, report.ci_upper],
    })
    return result


def run_clv_cycle() -> dict:
    """Fige les lignes de clôture, remplit le CLV des paris dont le match a
    démarré, puis renvoie le rapport. Le gate : clv_mean > 0 et significant=True."""
    n_snap = snapshot_closing_lines()
    n_clv = fetch_and_update_clv()
    rep = paper_trading_report()
    rep["_snapshot"] = n_snap
    rep["_clv_updated"] = n_clv
    return rep


if __name__ == "__main__":
    r = run_clv_cycle()
    print(f"Lignes de clôture figées : {r.pop('_snapshot', 0)} matchs à venir")
    print(f"CLV rempli pour          : {r.pop('_clv_updated', 0)} paris")
    n = r.get("n_bets", 0)
    cm = r.get("clv_mean")
    bm = r.get("clv_betclic_mean")
    if cm is None:
        print(f"Paris paper : {n} · CLV : en attente (aucun match clôturé)")
    else:
        ci = r.get("ci", [None, None])
        verdict = "EDGE confirmé ✅" if r.get("significant") else "non significatif (échantillon ↑)"
        print(f"Paris paper : {n}")
        print(f"CLV Pinnacle (edge) : {cm:+.4f}  ({r.get('beat_close_pct', 0)*100:.0f}% battent la clôture)")
        print(f"IC 95%              : [{ci[0]:+.4f}, {ci[1]:+.4f}] → {verdict}")
        if bm is not None:
            print(f"CLV Betclic (ligne) : {bm:+.4f}  (n={r.get('clv_betclic_n', 0)})")
        print(f"CLV 7j : {r['clv_7d']}  · 30j : {r['clv_30d']}")
