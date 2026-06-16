"""
Fuseau horaire unique : tout est affiché et ordonné en heure de Montréal
(Louis suit la CDM 2026 depuis Montréal). Les sources de données — wc_all.parquet
et l'API bzzoiro — sont toutes en UTC. Un tiers des matchs (coups d'envoi en
soirée aux US/Canada) tombent le lendemain en UTC : sans conversion, ils
s'affichent au mauvais jour et les scores ne se rattachent pas.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

MONTREAL = ZoneInfo("America/Montreal")
_UTC = ZoneInfo("UTC")


def to_montreal(dt) -> datetime:
    """datetime / pandas.Timestamp / str ISO (UTC) → datetime en heure de Montréal."""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(MONTREAL)


def mtl_date(dt) -> str:
    """Date locale Montréal au format YYYY-MM-DD (clé de regroupement des matchs)."""
    return to_montreal(dt).date().isoformat()


def mtl_time(dt) -> str:
    """Heure locale Montréal au format HH:MM (coup d'envoi)."""
    return to_montreal(dt).strftime("%H:%M")
