"""Backtest legacy — redirige vers run_walkforward."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.run_walkforward import run_full_backtest, BacktestConfig

if __name__ == "__main__":
    run_full_backtest(BacktestConfig())
