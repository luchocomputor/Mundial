"""
Backtest walk-forward rigoureux sur cotes réelles.
Remplace backtest_wc2022.py (cotes synthétiques).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.go_nogo import GO_NOGO_CRITERIA, evaluate, GoNoGoDecision
from evaluation.clv import bootstrap_clv_significance, clv
from evaluation.metrics import compute_all_metrics
from evaluation.report import generate_report
from evaluation.walkforward import WalkForwardConfig, WalkForwardRunner, compute_outcomes
from models.base import MatchContext
from models.ratings import EloRating
from odds.devig import devig
from pipeline.config import load_config
from pipeline.features import build_training_data, load_raw_matches
from pipeline.model_loader import load_calibrator
from pipeline.risk import kelly_with_uncertainty
from pipeline.signal_guards import evaluate_signal

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "data" / "processed" / "backtest"
GO_NOGO_REPORT = ROOT / "data" / "processed" / "go_nogo_report.json"


@dataclass
class BacktestConfig:
    train_end_year: int = 2021
    test_year: int = 2022
    bankroll: float = 100.0
    kelly_fraction: float = 0.25
    max_kelly: float = 0.05
    edge_threshold: float = 0.05
    commission: float = 0.02


def load_matches_and_odds() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_path = DATA_RAW / "all_matches.parquet"
    odds_path = DATA_RAW / "odds_history.parquet"

    if not matches_path.exists():
        raise FileNotFoundError(f"Lance fetch_data + build_dataset : {matches_path}")

    matches = pd.read_parquet(matches_path)
    matches["date"] = pd.to_datetime(matches["date"], utc=True)
    matches = matches[matches["status"] == "finished"]

    odds = pd.DataFrame()
    if odds_path.exists():
        odds = pd.read_parquet(odds_path)

    return matches, odds


def simulate_bets_with_real_odds(
    predictions: list[dict],
    test_rows: pd.DataFrame,
    odds: pd.DataFrame,
    cfg: BacktestConfig,
    app_cfg=None,
) -> pd.DataFrame:
    results = []
    bankroll = cfg.bankroll
    peak = bankroll
    calibrator = load_calibrator()
    mcfg = app_cfg.model if app_cfg else None
    max_div = mcfg.max_market_divergence if mcfg else 0.15
    anchor_w = mcfg.market_anchor_weight if mcfg else 0.3
    threshold = cfg.edge_threshold

    odds_by_fixture = {}
    if not odds.empty:
        for fid, grp in odds.groupby("fixture_id"):
            close = grp[grp["snapshot_type"] == "close"]
            if close.empty:
                close = grp
            odds_by_fixture[fid] = close

    for pred, (_, row) in zip(predictions, test_rows.iterrows()):
        fid = row["fixture_id"]
        if fid not in odds_by_fixture:
            continue

        if calibrator:
            pred = calibrator.calibrate(pred)

        fixture_odds = odds_by_fixture[fid]
        markets_odds: dict[str, dict[str, float]] = {}
        for _, o in fixture_odds.iterrows():
            markets_odds.setdefault(o["market"], {})[o["side"]] = o["odds_decimal"]

        outcome = compute_outcomes(row)
        market_map = [
            ("1X2", "home", "home_win"),
            ("1X2", "draw", "draw"),
            ("1X2", "away", "away_win"),
            ("over_2.5", "over", "over_2.5"),
            ("btts", "yes", "btts"),
        ]

        for market, side, pred_key in market_map:
            cote = markets_odds.get(market, {}).get(side)
            if not cote or cote <= 1:
                continue

            p_raw = pred.get(pred_key, 0)
            novig = devig(markets_odds.get(market, {}))
            p_novig = novig.get(side, 1 / cote)

            guard = evaluate_signal(p_raw, p_novig, max_div, anchor_w)
            if not guard.accepted:
                continue

            p_model = guard.p_final
            edge = p_model - p_novig
            if edge < threshold:
                continue

            kelly = kelly_with_uncertainty(p_model, cote, 0, cfg.kelly_fraction, cfg.max_kelly)
            stake = bankroll * kelly
            won = outcome.get(pred_key, 0) == 1
            profit = stake * (cote * (1 - cfg.commission) - 1) if won else -stake
            bankroll = max(0.01, bankroll + profit)
            peak = max(peak, bankroll)

            open_odds = cote
            close_odds = cote
            clv_val = clv(open_odds, close_odds)

            results.append({
                "fixture_id": fid,
                "market": market,
                "side": side,
                "p_model": p_model,
                "edge": edge,
                "cote": cote,
                "stake": stake,
                "outcome": int(won),
                "profit": profit,
                "bankroll_after": bankroll,
                "clv": clv_val,
            })

    return pd.DataFrame(results)


def run_full_backtest(config: BacktestConfig | None = None) -> dict:
    config = config or BacktestConfig()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    app_cfg = load_config()

    print("Chargement données...")
    matches, odds = load_matches_and_odds()

    train = matches[matches["date"].dt.year <= config.train_end_year]
    test = matches[matches["date"].dt.year == config.test_year]

    if train.empty or test.empty:
        print("Données insuffisantes pour backtest.")
        return {"error": "insufficient_data"}

    ref_date = pd.Timestamp(f"{config.train_end_year}-12-31", tz="UTC")
    training = build_training_data(
        train,
        friendly_weight=app_cfg.model.friendly_weight,
        reference_date=ref_date,
        decay=app_cfg.model.decay,
    )

    print(f"Train: {len(training)} | Test: {len(test)}")

    model = EloRating()
    model.fit(training, reference_date=ref_date)

    predictions = []
    for _, row in test.iterrows():
        ctx = MatchContext(
            is_neutral=bool(row.get("is_neutral", False)),
            venue=str(row.get("venue", "")),
            is_friendly=bool(row.get("is_friendly", False)),
        )
        pred = model.predict_outcomes(row["home_team"], row["away_team"], context=ctx)
        predictions.append(pred.to_dict() if hasattr(pred, "to_dict") else pred)

    outcomes = [compute_outcomes(row) for _, row in test.iterrows()]
    metrics = compute_all_metrics(predictions, outcomes)

    bets_df = simulate_bets_with_real_odds(predictions, test, odds, config, app_cfg)

    report = {
        "n_test_matches": len(test),
        "metrics": {
            "rps": metrics.rps,
            "ece": metrics.ece,
            "log_loss": metrics.log_loss_binary,
        },
        "n_bets": len(bets_df),
    }

    if not bets_df.empty:
        total_staked = bets_df["stake"].sum()
        total_profit = bets_df["profit"].sum()
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0
        clv_report = bootstrap_clv_significance(bets_df["clv"])
        dd = (bets_df["bankroll_after"].cummax() - bets_df["bankroll_after"]).max()
        dd_pct = dd / config.bankroll if config.bankroll > 0 else 0

        report.update({
            "roi_pct": round(roi, 2),
            "total_profit": round(total_profit, 2),
            "hit_rate": round(bets_df["outcome"].mean() * 100, 1),
            "clv_mean": clv_report.mean_clv,
            "clv_significant": clv_report.significant,
            "beat_close_pct": clv_report.beat_close_pct,
            "max_drawdown_pct": round(dd_pct * 100, 1),
        })

        bets_df.to_csv(REPORT_DIR / "simulated_bets.csv", index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(bets_df["bankroll_after"].values)
        plt.axhline(config.bankroll, color="gray", linestyle="--", alpha=0.5)
        plt.title("Bankroll — Walk-forward backtest")
        plt.xlabel("Bet #")
        plt.ylabel("Bankroll (€)")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "bankroll_evolution.png", dpi=150)
        plt.close()

        decision = evaluate(report)
        report["go_nogo"] = decision.value
        GO_NOGO_REPORT.parent.mkdir(parents=True, exist_ok=True)
        GO_NOGO_REPORT.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nDécision : {decision.value}")
    else:
        report["go_nogo"] = GoNoGoDecision.NO_GO.value
        GO_NOGO_REPORT.parent.mkdir(parents=True, exist_ok=True)
        GO_NOGO_REPORT.write_text(json.dumps(report, indent=2, default=str))
        print("Aucun pari simulé (edge insuffisant ou cotes manquantes).")

    (REPORT_DIR / "backtest_report.json").write_text(json.dumps(report, indent=2, default=str))
    generate_report(metrics, output_dir=REPORT_DIR)
    print(f"Rapport : {REPORT_DIR}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", type=int, default=2021)
    parser.add_argument("--test-year", type=int, default=2022)
    args = parser.parse_args()
    run_full_backtest(BacktestConfig(train_end_year=args.train_end, test_year=args.test_year))
