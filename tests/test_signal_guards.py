"""Tests garde-fous signaux."""

from pipeline.signal_guards import (
    GuardStats,
    anchor_to_market,
    evaluate_signal,
    is_plausible_signal,
)


def test_reject_high_divergence():
    assert not is_plausible_signal(0.61, 0.09, max_divergence=0.15)


def test_accept_close_probs():
    assert is_plausible_signal(0.45, 0.40, max_divergence=0.15)


def test_anchor_pulls_toward_market():
    p_final, w = anchor_to_market(0.61, 0.09, anchor_weight=0.3)
    assert p_final < 0.61
    assert p_final > 0.09


def test_evaluate_signal_rejects_panama():
    result = evaluate_signal(0.61, 0.09, max_divergence=0.15, anchor_weight=0.3)
    assert not result.accepted
    assert result.reason == "divergence"


def test_evaluate_signal_accepts_reasonable():
    result = evaluate_signal(0.48, 0.42, max_divergence=0.15, anchor_weight=0.3)
    assert result.accepted
    assert result.p_final > 0
