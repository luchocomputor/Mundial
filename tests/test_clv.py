from evaluation.clv import clv, bootstrap_clv_significance
import pandas as pd


def test_clv_positive_when_beat_close():
    assert clv(2.10, 2.00) > 0


def test_clv_negative_when_worse_than_close():
    assert clv(1.90, 2.00) < 0


def test_bootstrap_clv():
    series = pd.Series([0.05, 0.03, 0.02, -0.01, 0.04])
    report = bootstrap_clv_significance(series, n_bootstrap=1000)
    assert report.n_bets == 5
    assert report.mean_clv > 0
