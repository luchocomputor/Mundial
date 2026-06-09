"""
Détection de value bets — edge no-vig, Kelly avec incertitude, garde-fous.
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from models.base import MatchContext, Prediction
from odds.devig import devig
from odds.edge import compute_edge_novig
from pipeline.config import load_config
from pipeline.features import get_altitude_adjustment
from pipeline.model_loader import load_calibrator
from pipeline.risk import BetCandidate, kelly_with_uncertainty, optimize_daily_portfolio
from pipeline.signal_guards import GuardStats, evaluate_signal

ROOT = Path(__file__).parent.parent
BETS_LOG = ROOT / "data" / "bets_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_edge(p_model: float, cote: float) -> float:
    if cote <= 1:
        return -1.0
    return p_model - (1.0 / cote)


def kelly_size(p_model: float, cote: float, fraction: float = 0.25) -> float:
    return kelly_with_uncertainty(p_model, cote, variance=0.0, fraction=fraction)


def compute_confidence(n_home: int, n_away: int, is_friendly: bool) -> int:
    base = min(5, (n_home + n_away) // 10)
    if is_friendly:
        base = max(1, base - 1)
    return max(1, base)


def _bzzoiro_to_market_odds(raw: dict) -> dict[str, dict[str, float]]:
    mapping = {
        "home_win": ("1X2", "home"),
        "draw": ("1X2", "draw"),
        "away_win": ("1X2", "away"),
        "over_25_goals": ("over_2.5", "over"),
        "under_25_goals": ("over_2.5", "under"),
        "btts_yes": ("btts", "yes"),
        "btts_no": ("btts", "no"),
    }
    result: dict[str, dict[str, float]] = {}
    for key, val in raw.items():
        if val is None or val <= 1:
            continue
        m = mapping.get(key)
        if m:
            market, side = m
            result.setdefault(market, {})[side] = val
    return result


def _predict(model, home, away, ctx, bzzo=None) -> dict:
    try:
        if hasattr(model, "predict_outcomes"):
            try:
                pred = model.predict_outcomes(home, away, context=ctx)
            except TypeError:
                try:
                    pred = model.predict_outcomes(home, away, ctx, bzzoiro_features=bzzo)
                except TypeError:
                    pred = model.predict_outcomes(
                        home, away,
                        altitude_adj=ctx.altitude_adj,
                        is_neutral=ctx.is_neutral,
                    )
        else:
            raise RuntimeError("Modèle sans predict_outcomes")
        return pred.to_dict() if hasattr(pred, "to_dict") else pred
    except Exception as e:
        logger.warning(f"Prédiction impossible {home} vs {away}: {e}")
        return {}


def scan_value_bets(
    matches: list[dict],
    model,
    bookmaker_odds: dict,
    cfg,
    bankroll: float,
    bzzoiro_predictions: dict | None = None,
    calibrator=None,
    team_match_counts: dict[str, int] | None = None,
) -> tuple[list[dict], GuardStats]:
    if calibrator is None:
        calibrator = load_calibrator()
    if team_match_counts is None:
        team_match_counts = {}

    mcfg = cfg.model if hasattr(cfg, "model") else cfg["model"]
    threshold = mcfg.min_edge_threshold if hasattr(mcfg, "min_edge_threshold") else mcfg["min_edge_threshold"]
    kelly_fraction = mcfg.kelly_fraction if hasattr(mcfg, "kelly_fraction") else mcfg["kelly_fraction"]
    max_kelly = mcfg.max_kelly_bet if hasattr(mcfg, "max_kelly_bet") else mcfg["max_kelly_bet"]
    devig_method = mcfg.devig_method if hasattr(mcfg, "devig_method") else mcfg.get("devig_method", "shin")
    max_divergence = mcfg.max_market_divergence if hasattr(mcfg, "max_market_divergence") else mcfg.get("max_market_divergence", 0.15)
    anchor_weight = mcfg.market_anchor_weight if hasattr(mcfg, "market_anchor_weight") else mcfg.get("market_anchor_weight", 0.3)
    sparse_threshold = mcfg.sparse_match_threshold if hasattr(mcfg, "sparse_match_threshold") else mcfg.get("sparse_match_threshold", 5)
    sparse_anchor = mcfg.sparse_anchor_weight if hasattr(mcfg, "sparse_anchor_weight") else mcfg.get("sparse_anchor_weight", 0.5)
    max_daily = cfg.risk.max_daily_exposure if hasattr(cfg, "risk") else cfg.get("risk", {}).get("max_daily_exposure", 0.15)

    value_bets = []
    candidates: list[BetCandidate] = []
    stats = GuardStats()

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        fid = match.get("fixture_id")
        venue = match.get("venue", "")
        is_friendly = match.get("is_friendly", False)
        is_neutral = match.get("is_neutral", False)
        altitude_adj = get_altitude_adjustment(venue)

        ctx = MatchContext(
            is_neutral=is_neutral,
            venue=venue,
            is_friendly=is_friendly,
            altitude_adj=altitude_adj,
        )

        bzzo = (bzzoiro_predictions or {}).get(fid)
        preds = _predict(model, home, away, ctx, bzzo)
        if not preds:
            continue

        if calibrator:
            preds = calibrator.calibrate(preds)

        raw_odds = bookmaker_odds.get(fid, {})
        odds_nested = _bzzoiro_to_market_odds(raw_odds) if "1X2" not in raw_odds else raw_odds

        min_matches = min(
            team_match_counts.get(home, 0),
            team_match_counts.get(away, 0),
        )

        market_map = [
            ("1X2", "home", "home_win", preds.get("home_win", 0)),
            ("1X2", "draw", "draw", preds.get("draw", 0)),
            ("1X2", "away", "away_win", preds.get("away_win", 0)),
            ("over_2.5", "over", "over_2.5", preds.get("over_2.5", 0)),
            ("btts", "yes", "btts", preds.get("btts", 0)),
        ]

        for market, side, pred_key, p_raw in market_map:
            market_odds = odds_nested.get(market, {})
            cote = market_odds.get(side)
            if not cote or cote <= 1:
                continue

            novig_probs = devig(market_odds, devig_method)
            p_novig = novig_probs.get(side, 1 / cote)

            guard = evaluate_signal(
                p_raw, p_novig,
                max_divergence=max_divergence,
                anchor_weight=anchor_weight,
                min_team_matches=min_matches,
            )
            if not guard.accepted:
                if guard.reason == "divergence":
                    stats.rejected_divergence += 1
                else:
                    stats.rejected_bounds += 1
                continue

            p_model = guard.p_final
            edge = p_model - p_novig
            if edge < threshold:
                continue

            stats.accepted += 1
            uncertainty = (preds.get("uncertainty") or {})
            var = uncertainty.get(pred_key, uncertainty.get(market, 0.0))

            candidates.append(BetCandidate(
                fixture_id=fid,
                market=market,
                side=side,
                p_model=p_model,
                odds=cote,
                edge=edge,
                variance=var,
            ))

            value_bets.append({
                "fixture_id": fid,
                "date": match.get("date"),
                "home_team": home,
                "away_team": away,
                "venue": venue,
                "is_friendly": is_friendly,
                "market": market,
                "side": side,
                "p_model": round(p_model, 4),
                "p_raw": round(p_raw, 4),
                "p_implicit": round(1 / cote, 4),
                "p_novig": round(p_novig, 4),
                "edge": round(edge, 4),
                "cote": cote,
                "odds_taken": cote,
                "odds_close": "",
                "clv": "",
                "kelly_pct": 0,
                "stake_eur": 0,
                "bankroll": bankroll,
                "confidence": compute_confidence(
                    team_match_counts.get(home, 0),
                    team_match_counts.get(away, 0),
                    is_friendly,
                ),
                "altitude_adj": altitude_adj,
                "model_version": preds.get("source", "unknown"),
                "anchor_weight": round(guard.anchor_weight, 3),
                "paper_trade": True,
            })

    if candidates and value_bets:
        stakes = optimize_daily_portfolio(
            candidates, bankroll, max_daily, kelly_fraction, max_kelly
        )
        for bet, stake in zip(value_bets, stakes):
            bet["stake_eur"] = stake
            bet["kelly_pct"] = round(stake / bankroll * 100, 2) if bankroll > 0 else 0

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets, stats


def log_bets(bets: list[dict]) -> None:
    if not bets:
        return
    BETS_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BETS_LOG.exists()
    extra = ["detected_at", "result", "profit_eur"]
    fieldnames = list(dict.fromkeys(list(bets[0].keys()) + extra))
    with open(BETS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for bet in bets:
            row = dict(bet)
            row["detected_at"] = datetime.now().isoformat()
            row.setdefault("result", "")
            row.setdefault("profit_eur", "")
            writer.writerow(row)
    logger.info(f"{len(bets)} bets loggés dans {BETS_LOG}")


def format_alert(bet: dict) -> str:
    stars = "★" * bet["confidence"] + "☆" * (5 - bet["confidence"])
    friendly_note = " (Amical)" if bet.get("is_friendly") else ""
    paper_note = " [PAPER]" if bet.get("paper_trade") else ""
    return (
        f"🎯 VALUE BET DÉTECTÉE{paper_note}{friendly_note}\n\n"
        f"⚽ {bet['home_team']} vs {bet['away_team']}\n"
        f"📅 {bet['date']}\n"
        f"🏟️ {bet.get('venue', '')}\n\n"
        f"Marché : {bet['market']} — {bet['side']}\n"
        f"Cote : {bet['cote']:.2f}\n"
        f"Probabilité modèle : {bet['p_model']*100:.1f}%\n"
        f"Probabilité no-vig : {bet.get('p_novig', bet['p_implicit'])*100:.1f}%\n"
        f"Edge : +{bet['edge']*100:.1f}%\n\n"
        f"Kelly : {bet['kelly_pct']:.1f}% | Mise : {bet['stake_eur']:.2f}€\n"
        f"Modèle : {bet.get('model_version', '?')}\n"
        f"Confiance : {stars}"
    )


def compute_roi_from_log() -> dict:
    if not BETS_LOG.exists():
        return {"n_bets": 0, "roi": 0.0, "profit": 0.0, "total_staked": 0.0}
    df = pd.read_csv(BETS_LOG)
    df = df[df["result"].isin(["W", "L", "V"])]
    if df.empty:
        return {"n_bets": 0, "roi": 0.0, "profit": 0.0, "total_staked": 0.0}
    df["profit_eur"] = pd.to_numeric(df["profit_eur"], errors="coerce").fillna(0)
    total_staked = df["stake_eur"].sum()
    total_profit = df["profit_eur"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0.0
    clv_mean = 0.0
    if "clv" in df.columns:
        clv_vals = pd.to_numeric(df["clv"], errors="coerce").dropna()
        clv_mean = float(clv_vals.mean()) if len(clv_vals) > 0 else 0.0
    return {
        "n_bets": len(df),
        "roi": round(roi * 100, 2),
        "profit": round(total_profit, 2),
        "total_staked": round(total_staked, 2),
        "clv_mean": round(clv_mean * 100, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", action="store_true")
    args = parser.parse_args()
    if args.roi:
        print(compute_roi_from_log())
