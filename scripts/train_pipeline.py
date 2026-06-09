#!/usr/bin/env python3
"""
Pipeline d'entraînement : OOF walk-forward → calibration → gates → artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.go_nogo import GoNoGoDecision
from evaluation.metrics import compute_all_metrics, ranked_probability_score
from evaluation.walkforward import WalkForwardConfig, WalkForwardRunner, compute_outcomes
from models.base import MatchContext
from models.calibration import MarketCalibrator, CALIB_PATH
from models.dixon_coles import DixonColesModel
from models.ensemble import StackingEnsemble
from models.ratings import EloRating, ARTIFACT_PATH as ELO_PATH
from pipeline.config import load_config
from pipeline.features import build_training_data, load_raw_matches

ARTIFACTS = ROOT / "models" / "artifacts"
GATES_REPORT = ARTIFACTS / "train_gates_report.json"


def _oof_predictions(matches: pd.DataFrame, years: list[int], max_per_year: int = 200) -> tuple[list[dict], list[dict]]:
    """Walk-forward OOF par année (échantillonné pour perf)."""
    preds, outcomes = [], []
    for year in years:
        train = matches[matches["date"].dt.year < year]
        test = matches[(matches["date"].dt.year == year) & (matches["status"] == "finished")]
        if len(test) > max_per_year:
            test = test.sample(max_per_year, random_state=42)
        if train.empty or test.empty:
            continue
        ref = pd.Timestamp(f"{year - 1}-12-31", tz="UTC")
        training = build_training_data(train, reference_date=ref)
        model = EloRating()
        model.fit(training, reference_date=ref)
        for _, row in test.iterrows():
            ctx = MatchContext(
                is_neutral=bool(row.get("is_neutral", False)),
                is_friendly=bool(row.get("is_friendly", False)),
            )
            pred = model.predict_outcomes(row["home_team"], row["away_team"], context=ctx)
            preds.append(pred.to_dict())
            outcomes.append(compute_outcomes(row))
    return preds, outcomes


def evaluate_gates(
    model_name: str,
    rps: float,
    ece: dict,
    elo_rps: float,
    market_rps: float | None = None,
) -> dict:
    gates = {
        "rps_vs_elo": rps <= elo_rps + 0.01,
        "rps_vs_market": market_rps is None or rps <= market_rps + 0.02,
        "ece_ok": all(v < 0.03 for v in ece.values()) if ece else False,
    }
    gates["passed"] = gates["rps_vs_elo"] and gates["ece_ok"]
    return {"model": model_name, "rps": rps, "ece": ece, "gates": gates}


def run_train_pipeline(train_end_year: int = 2025) -> dict:
    cfg = load_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    matches = load_raw_matches()
    finished = matches[matches["status"] == "finished"]
    ref_date = pd.Timestamp(f"{train_end_year}-12-31", tz="UTC")
    training = build_training_data(
        finished[finished["date"].dt.year <= train_end_year],
        friendly_weight=cfg.model.friendly_weight,
        reference_date=ref_date,
        decay=cfg.model.decay,
    )

    report: dict = {"train_end_year": train_end_year, "n_training": len(training)}

    # 1. Elo (baseline prod)
    elo = EloRating()
    elo.fit(training, reference_date=ref_date)
    elo.save(ELO_PATH)
    report["elo"] = {"status": "saved", "n_teams": len(elo.ratings)}

    # 2. Dixon-Coles (top équipes pour perf MLE)
    team_counts = pd.concat([training["home_team"], training["away_team"]]).value_counts()
    top_teams = set(team_counts.head(40).index)
    dc_train = training[
        training["home_team"].isin(top_teams) & training["away_team"].isin(top_teams)
    ]
    if len(dc_train) > 400:
        dc_train = dc_train.sample(400, random_state=42)
    dc = DixonColesModel(decay=cfg.model.decay, friendly_weight=cfg.model.friendly_weight)
    if len(dc_train) >= 100:
        dc.fit(dc_train, reference_date=ref_date)
        dc.save()
        report["dixon_coles"] = {"status": "saved", "n_confederations": len(dc.conf_attack), "n_train": len(dc_train)}
    else:
        report["dixon_coles"] = {"status": "skipped", "reason": "insufficient DC training rows"}
        dc = None

    # 3. OOF pour calibration
    oof_years = [2018, 2019, 2020, 2021, 2022]
    oof_years = [y for y in oof_years if y in finished["date"].dt.year.unique()]
    oof_preds, oof_outcomes = _oof_predictions(finished, oof_years)
    report["oof"] = {"n_predictions": len(oof_preds), "years": oof_years}

    calibrator = MarketCalibrator()
    if len(oof_preds) >= 50:
        calibrator.fit_all_markets(oof_preds, oof_outcomes)
        calibrator.save()
        report["calibrator"] = {"status": "saved", "markets": list(calibrator.calibrators.keys())}
    else:
        report["calibrator"] = {"status": "skipped", "reason": "insufficient OOF"}

    # 4. Gates RPS/ECE sur CDM 2022
    test_2022 = finished[finished["date"].dt.year == 2022]
    elo_preds, elo_outcomes = [], []
    for _, row in test_2022.iterrows():
        ctx = MatchContext(is_neutral=bool(row.get("is_neutral", False)))
        p = elo.predict_outcomes(row["home_team"], row["away_team"], context=ctx)
        elo_preds.append(p.to_dict())
        elo_outcomes.append(compute_outcomes(row))

    dc_preds = []
    if dc is not None:
        for _, row in test_2022.iterrows():
            ctx = MatchContext(is_neutral=bool(row.get("is_neutral", False)))
            p = dc.predict_outcomes(row["home_team"], row["away_team"], context=ctx)
            dc_preds.append(p.to_dict() if hasattr(p, "to_dict") else p)

    elo_metrics = compute_all_metrics(elo_preds, elo_outcomes)
    gates = [evaluate_gates("elo", elo_metrics.rps, elo_metrics.ece, elo_metrics.rps)]
    if dc_preds:
        dc_metrics = compute_all_metrics(dc_preds, elo_outcomes)
        gates.append(evaluate_gates("dixon_coles", dc_metrics.rps, dc_metrics.ece, elo_metrics.rps))
    else:
        dc_metrics = None
    report["gates"] = gates

    # 5. Ensemble meta seulement si DC passe gate RPS vs Elo
    dc_passed = len(gates) > 1 and gates[1]["gates"]["passed"]
    if dc_passed and dc is not None and len(oof_preds) >= 100:
        ensemble = StackingEnsemble()
        ensemble.add_model(elo, "elo")
        ensemble.add_model(dc, "dixon_coles")
        ensemble.fit(training, reference_date=ref_date)
        # Meta simplifié sur OOF 1X2
        oof_df = pd.DataFrame([
            {**{f"elo_{k}": p[k] for k in ("home_win", "draw", "away_win")}, **p}
            for p in oof_preds[:500]
        ])
        out_df = pd.DataFrame(oof_outcomes[:500])
        if len(oof_df) >= 50:
            ensemble.fit_meta(oof_df, out_df)
        ensemble.save()
        report["ensemble"] = {"status": "saved", "has_meta": ensemble.meta_model is not None}
    else:
        report["ensemble"] = {"status": "skipped", "reason": "DC gate failed or insufficient OOF"}

    report["production_model"] = "elo"
    report["production_mode"] = cfg.model.production_mode

    GATES_REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", type=int, default=2025)
    args = parser.parse_args()
    run_train_pipeline(args.train_end)
