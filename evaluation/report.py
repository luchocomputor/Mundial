"""Génération de rapports d'évaluation."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.clv import CLVReport
from evaluation.metrics import MetricsReport


def generate_report(
    metrics: MetricsReport,
    clv: CLVReport | None = None,
    output_dir: Path | None = None,
) -> dict:
    report = {
        "n_samples": metrics.n_samples,
        "rps": metrics.rps,
        "log_loss_binary": metrics.log_loss_binary,
        "brier": metrics.brier,
        "ece": metrics.ece,
    }
    if clv:
        report["clv"] = {
            "mean": clv.mean_clv,
            "beat_close_pct": clv.beat_close_pct,
            "n_bets": clv.n_bets,
            "ci_lower": clv.ci_lower,
            "ci_upper": clv.ci_upper,
            "significant": clv.significant,
        }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "evaluation_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    return report
