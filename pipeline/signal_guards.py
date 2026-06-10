"""Garde-fous pour filtrer les signaux non plausibles."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GuardStats:
    accepted: int = 0
    rejected_divergence: int = 0
    rejected_bounds: int = 0
    rejected_low_matches: int = 0


@dataclass
class GuardResult:
    accepted: bool
    reason: str = ""
    p_final: float = 0.0
    anchor_weight: float = 0.0


def is_plausible_signal(
    p_model: float,
    p_market_novig: float,
    max_divergence: float = 0.15,
) -> bool:
    """Rejette si |p_model - p_marché| > max_divergence."""
    return abs(p_model - p_market_novig) <= max_divergence


def anchor_to_market(
    p_model: float,
    p_market_novig: float,
    anchor_weight: float = 0.3,
    min_matches: int = 999,
    sparse_threshold: int = 5,
    sparse_anchor: float = 0.5,
) -> tuple[float, float]:
    """
    Ancre la proba modèle vers le marché.
    Plus de poids marché si équipe peu observée.
    Retourne (p_final, weight_used).
    """
    weight = anchor_weight
    if min_matches < sparse_threshold:
        weight = max(weight, sparse_anchor)
    weight = min(max(weight, 0.0), 1.0)
    p_final = (1 - weight) * p_model + weight * p_market_novig
    return p_final, weight


def apply_probability_bounds(p: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, p))


def evaluate_signal(
    p_model: float,
    p_market_novig: float,
    max_divergence: float = 0.15,
    anchor_weight: float = 0.3,
    min_team_matches: int = 999,
) -> GuardResult:
    """Pipeline complet : ancrage → plausibilité → bounds."""
    p_anchored, w = anchor_to_market(
        p_model, p_market_novig, anchor_weight, min_team_matches
    )
    p_final = apply_probability_bounds(p_anchored)

    if not is_plausible_signal(p_final, p_market_novig, max_divergence):
        return GuardResult(
            accepted=False,
            reason="divergence",
            p_final=p_final,
            anchor_weight=w,
        )

    return GuardResult(
        accepted=True,
        p_final=p_final,
        anchor_weight=w,
    )
