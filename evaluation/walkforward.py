"""Walk-forward validation (rolling-origin, no look-ahead)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from evaluation.clv import CLVReport, bootstrap_clv_significance, clv
from evaluation.metrics import MetricsReport, compute_all_metrics


@dataclass
class WalkForwardConfig:
    train_min_days: int = 365 * 3
    retrain_every_days: int = 30
    prediction_horizon_days: int = 7


@dataclass
class WalkForwardResult:
    predictions: list[dict] = field(default_factory=list)
    outcomes: list[dict] = field(default_factory=list)
    bets: list[dict] = field(default_factory=list)
    metrics: MetricsReport | None = None
    clv_report: CLVReport | None = None


def compute_outcomes(row: pd.Series) -> dict:
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    return {
        "home_win": int(hg > ag),
        "draw": int(hg == ag),
        "away_win": int(hg < ag),
        "over_2.5": int(hg + ag > 2.5),
        "btts": int(hg > 0 and ag > 0),
    }


class WalkForwardRunner:
    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()

    def run(
        self,
        model_factory: Callable,
        matches: pd.DataFrame,
        odds: pd.DataFrame | None = None,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> WalkForwardResult:
        result = WalkForwardResult()
        df = matches[matches["status"] == "finished"].copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date")

        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        if df.empty:
            return result

        dates = pd.date_range(
            df["date"].min() + pd.Timedelta(days=self.config.train_min_days),
            df["date"].max(),
            freq=f"{self.config.retrain_every_days}D",
        )

        odds_map = {}
        if odds is not None and not odds.empty:
            for fid, grp in odds.groupby("fixture_id"):
                close = grp[grp["snapshot_type"] == "close"]
                if close.empty:
                    close = grp
                odds_map[fid] = close

        for ref_date in dates:
            train = df[df["date"] < ref_date]
            test_end = ref_date + pd.Timedelta(days=self.config.prediction_horizon_days)
            test = df[(df["date"] >= ref_date) & (df["date"] < test_end)]

            if len(train) < 50 or test.empty:
                continue

            model = model_factory()
            model.fit(train, reference_date=ref_date)

            for _, row in test.iterrows():
                home, away = row["home_team"], row["away_team"]
                try:
                    from models.base import MatchContext

                    ctx = MatchContext(
                        is_neutral=bool(row.get("is_neutral", False)),
                        venue=str(row.get("venue", "")),
                        is_friendly=bool(row.get("is_friendly", False)),
                        date=row["date"],
                    )
                    pred = model.predict_outcomes(home, away, ctx)
                    pred_dict = pred.to_dict() if hasattr(pred, "to_dict") else pred
                except Exception:
                    continue

                result.predictions.append(pred_dict)
                result.outcomes.append(compute_outcomes(row))

                fid = row.get("fixture_id")
                if odds is not None and fid in odds_map:
                    close_odds = odds_map[fid]
                    for _, o_row in close_odds.iterrows():
                        result.bets.append({
                            "fixture_id": fid,
                            "market": o_row["market"],
                            "side": o_row["side"],
                            "odds_close": o_row["odds_decimal"],
                        })

        if result.predictions:
            result.metrics = compute_all_metrics(
                result.predictions, result.outcomes
            )

        if result.bets:
            clv_vals = pd.Series([b.get("clv", 0) for b in result.bets])
            result.clv_report = bootstrap_clv_significance(clv_vals)

        return result
