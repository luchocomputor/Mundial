"""Utilitaires production : Go/No-Go, alertes autorisées."""

from __future__ import annotations

import json
from pathlib import Path

from backtest.go_nogo import GoNoGoDecision, evaluate

ROOT = Path(__file__).parent.parent
GO_NOGO_REPORT = ROOT / "data" / "processed" / "go_nogo_report.json"


def load_latest_go_nogo() -> GoNoGoDecision:
    if not GO_NOGO_REPORT.exists():
        return GoNoGoDecision.NO_GO
    try:
        report = json.loads(GO_NOGO_REPORT.read_text())
        return evaluate(report)
    except Exception:
        return GoNoGoDecision.NO_GO


def alerts_allowed(cfg) -> bool:
    """Alertes Telegram uniquement si mode live ET Go/No-Go OK."""
    mode = cfg.model.production_mode if hasattr(cfg, "model") else cfg.get("model", {}).get("production_mode", "paper_only")
    if mode == "paper_only":
        return False
    return load_latest_go_nogo() == GoNoGoDecision.GO_PAPER_TRADING
