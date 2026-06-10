"""Paper trading — enregistre paris simulés, CLV post-clôture, rolling 7j/30j."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from evaluation.clv import bootstrap_clv_significance, clv
from pipeline.fetch_odds import fetch_odds_for_event
from pipeline.value_detector import BETS_LOG, log_bets

ROOT = Path(__file__).parent.parent
PAPER_LOG = ROOT / "data" / "paper_trading.json"


def record_paper_bets(bets: list[dict]) -> None:
    for bet in bets:
        bet["paper_trade"] = True
        bet["recorded_at"] = datetime.now().isoformat()
    log_bets(bets)

    existing = []
    if PAPER_LOG.exists():
        existing = json.loads(PAPER_LOG.read_text())
    existing.extend(bets)
    PAPER_LOG.write_text(json.dumps(existing, indent=2, default=str))


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
    """Fetch cotes close et met à jour CLV pour paris en attente."""
    if not BETS_LOG.exists():
        return 0
    df = pd.read_csv(BETS_LOG)
    pending = df[df["clv"].isna() | (df["clv"] == "")]
    if fixture_ids:
        pending = pending[pending["fixture_id"].isin(fixture_ids)]
    if pending.empty:
        return 0

    updated = 0
    for fid in pending["fixture_id"].unique():
        rows = fetch_odds_for_event(int(fid))
        if not rows:
            continue
        # Prendre cote 1X2 home comme proxy close (marché le plus liquide)
        close_odds = {r["market"]: r for r in rows}
        for idx in pending[pending["fixture_id"] == fid].index:
            market = df.loc[idx, "market"]
            side = df.loc[idx, "side"]
            match_rows = [r for r in rows if r["market"] == market and r["side"] == side]
            if match_rows:
                odds_close = match_rows[0]["odds_decimal"]
                odds_taken = float(df.loc[idx, "odds_taken"] or df.loc[idx, "cote"])
                df.loc[idx, "odds_close"] = odds_close
                df.loc[idx, "clv"] = clv(odds_taken, odds_close)
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

    if clv_vals.empty:
        result["clv"] = "pending"
        return result

    report = bootstrap_clv_significance(clv_vals)
    result.update({
        "clv_mean": report.mean_clv,
        "beat_close_pct": report.beat_close_pct,
        "significant": report.significant,
        "ci": [report.ci_lower, report.ci_upper],
    })
    return result
