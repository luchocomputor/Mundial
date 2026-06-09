"""Détection des équipes placeholder (Winner Group A, etc.)."""

from __future__ import annotations

import re

PLACEHOLDER_PATTERNS = [
    re.compile(r"^Winner Group", re.I),
    re.compile(r"^Runner.?up Group", re.I),
    re.compile(r"^[WL]\d Group", re.I),
    re.compile(r"^3rd place Group", re.I),
    re.compile(r"^Best 3rd", re.I),
    re.compile(r"^Group \d+ (Winner|Runner)", re.I),
    re.compile(r"^TBD$", re.I),
    re.compile(r"^To Be Determined", re.I),
]


def is_placeholder_team(name: str | None) -> bool:
    if not name or not isinstance(name, str):
        return True
    name = name.strip()
    if not name:
        return True
    return any(p.search(name) for p in PLACEHOLDER_PATTERNS)


def filter_real_teams(df, home_col="home_team", away_col="away_team"):
    """Filtre les matchs avec équipes réelles (pas de placeholders)."""
    mask = (
        ~df[home_col].apply(is_placeholder_team)
        & ~df[away_col].apply(is_placeholder_team)
    )
    return df[mask].copy()
