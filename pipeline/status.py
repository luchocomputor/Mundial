"""Normalisation des statuts de match API → canonique."""

from __future__ import annotations

from typing import Literal

MatchStatus = Literal["finished", "upcoming", "cancelled"]

_STATUS_MAP: dict[str, MatchStatus] = {
    "finished": "finished",
    "ft": "finished",
    "complete": "finished",
    "notstarted": "upcoming",
    "scheduled": "upcoming",
    "upcoming": "upcoming",
    "ns": "upcoming",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "postponed": "cancelled",
    "abandoned": "cancelled",
}


def normalize_status(raw: str | None) -> MatchStatus:
    if not raw:
        return "upcoming"
    key = raw.strip().lower()
    return _STATUS_MAP.get(key, "upcoming")


def is_finished(raw: str | None) -> bool:
    return normalize_status(raw) == "finished"


def is_upcoming(raw: str | None) -> bool:
    return normalize_status(raw) == "upcoming"
