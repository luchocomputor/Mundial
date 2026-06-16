"""
Source unique pour les noms d'équipes : normalisation canonique + drapeaux.

Avant, `_norm`/`FLAGS` étaient dupliqués (dashboard.py, resolve_scores.py) avec
des jeux d'alias divergents → scores parfois non rattachés (ex. la même équipe
écrite "Bosnia & Herzegovina" d'un côté et "Bosnia-Herzegovina" de l'autre).

Drapeaux : générés depuis le code ISO 3166-1 alpha-2 via les "regional
indicators" Unicode (aucun emoji écrit à la main → zéro typo, et toute nation
mappée a son drapeau). Les nations UK (Angleterre/Écosse/Pays de Galles) ont
des drapeaux à séquence de tags, gérés à part.
"""

from __future__ import annotations

# ── Alias (minuscule) → forme canonique ─────────────────────────────────────
_ALIASES: dict[str, str] = {
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "korea": "south korea",
    "czech republic": "czechia",
    "bosnia and herzegovina": "bosnia-herzegovina",
    "bosnia & herzegovina": "bosnia-herzegovina",
    "bosnia herzegovina": "bosnia-herzegovina",
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "ivory coast": "côte d'ivoire",
    "cote d'ivoire": "côte d'ivoire",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "drc": "dr congo",
    "democratic republic of congo": "dr congo",
    "dr congo": "dr congo",
    "cape verde": "cabo verde",
    "curacao": "curaçao",
}


def normalize(name: str) -> str:
    """Nom → forme canonique minuscule (pour matcher entre sources)."""
    n = (name or "").lower().strip()
    return _ALIASES.get(n, n)


# ── Nom canonique → ISO 3166-1 alpha-2 ──────────────────────────────────────
_ISO: dict[str, str] = {
    # CONMEBOL
    "argentina": "AR", "brazil": "BR", "colombia": "CO", "uruguay": "UY",
    "ecuador": "EC", "paraguay": "PY", "chile": "CL", "peru": "PE",
    "venezuela": "VE", "bolivia": "BO",
    # UEFA
    "france": "FR", "spain": "ES", "germany": "DE", "portugal": "PT",
    "netherlands": "NL", "belgium": "BE", "italy": "IT", "croatia": "HR",
    "austria": "AT", "switzerland": "CH", "denmark": "DK", "poland": "PL",
    "serbia": "RS", "turkey": "TR", "bosnia-herzegovina": "BA", "sweden": "SE",
    "czechia": "CZ", "ukraine": "UA", "norway": "NO", "slovakia": "SK",
    "romania": "RO", "albania": "AL", "north macedonia": "MK", "hungary": "HU",
    "greece": "GR", "ireland": "IE", "slovenia": "SI",
    # AFC
    "japan": "JP", "south korea": "KR", "australia": "AU", "iran": "IR",
    "saudi arabia": "SA", "qatar": "QA", "jordan": "JO", "uzbekistan": "UZ",
    "iraq": "IQ",
    # CAF
    "morocco": "MA", "senegal": "SN", "egypt": "EG", "algeria": "DZ",
    "ghana": "GH", "côte d'ivoire": "CI", "tunisia": "TN", "south africa": "ZA",
    "cabo verde": "CV", "dr congo": "CD", "nigeria": "NG", "cameroon": "CM",
    "mali": "ML",
    # CONCACAF
    "united states": "US", "mexico": "MX", "canada": "CA", "costa rica": "CR",
    "panama": "PA", "honduras": "HN", "haiti": "HT", "curaçao": "CW",
    "jamaica": "JM",
    # OFC
    "new zealand": "NZ",
}

# Nations UK : pas d'ISO pays → drapeaux à séquence de tags (rendent sur Apple)
_SPECIAL: dict[str, str] = {
    "england": "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "scotland": "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "wales": "🏴\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f",
}


def _iso_to_emoji(iso2: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in iso2.upper())


def flag(name: str) -> str:
    """Nom d'équipe → drapeau emoji. 🏳 seulement si la nation est inconnue
    (ne devrait pas arriver pour les vraies équipes)."""
    n = normalize(name)
    if n in _SPECIAL:
        return _SPECIAL[n]
    iso = _ISO.get(n)
    return _iso_to_emoji(iso) if iso else "🏳"
