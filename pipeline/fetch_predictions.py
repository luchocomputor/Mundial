"""
Récupère les prédictions CatBoost de bzzoiro pour les matchs CDM/amicaux.

Ces prédictions servent de second signal aux côtés de Dixon-Coles.
Stratégie de combinaison : moyenne pondérée (Dixon-Coles 60%, bzzoiro 40%)
— ajustable si le backtest montre une meilleure performance bzzoiro.
"""

import time
from pathlib import Path

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
CONFIG_PATH = ROOT / "config.yaml"
BASE_URL = "https://sports.bzzoiro.com/api/v2"
LEAGUE_WC = 27
LEAGUE_FRIENDLY = 31


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _headers(cfg: dict) -> dict:
    return {"Authorization": f"Token {cfg['bzzoiro']['token']}"}


def _get_all_pages(path: str, params: dict, cfg: dict, sleep: float = 0.2) -> list:
    results = []
    offset = 0
    limit = 100
    while True:
        resp = requests.get(
            BASE_URL + path,
            headers=_headers(cfg),
            params={**params, "limit": limit, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("results", [])
        results.extend(batch)
        if not data.get("next"):
            break
        offset += limit
        time.sleep(sleep)
    return results


def _parse_prediction(p: dict) -> dict | None:
    ev = p.get("event", {})
    if not ev:
        return None
    markets = p.get("markets", {})
    mr = markets.get("match_result", {})
    ou = markets.get("over_under", {})
    btts = markets.get("btts", {})
    xg = markets.get("expected_goals", {})
    model = p.get("model", {})

    if not mr.get("prob_home"):
        return None

    return {
        "event_id": ev["id"],
        "event_date": ev.get("event_date"),
        "home_team": ev.get("home_team"),
        "away_team": ev.get("away_team"),
        "league_id": ev.get("league_id"),
        "league_name": ev.get("league_name"),
        "status": ev.get("status"),
        "prob_home": mr.get("prob_home", 0) / 100,
        "prob_draw": mr.get("prob_draw", 0) / 100,
        "prob_away": mr.get("prob_away", 0) / 100,
        "predicted_result": mr.get("predicted"),
        "xg_home": xg.get("home"),
        "xg_away": xg.get("away"),
        "prob_over_15": ou.get("prob_over_15", 0) / 100,
        "prob_over_25": ou.get("prob_over_25", 0) / 100,
        "prob_over_35": ou.get("prob_over_35", 0) / 100,
        "prob_btts": btts.get("prob_yes", 0) / 100,
        "model_confidence": model.get("confidence"),
        "model_version": model.get("version"),
    }


def fetch_all_predictions(cfg: dict | None = None) -> pd.DataFrame:
    """Toutes les prédictions disponibles (toutes ligues)."""
    if cfg is None:
        cfg = load_config()
    raw = _get_all_pages("/predictions/", {}, cfg)
    rows = [_parse_prediction(p) for p in raw]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
    return df


def fetch_international_predictions(cfg: dict | None = None) -> pd.DataFrame:
    """Prédictions filtrées sur WC + amicaux internationaux uniquement."""
    df = fetch_all_predictions(cfg)
    if df.empty:
        return df
    return df[df["league_id"].isin([LEAGUE_WC, LEAGUE_FRIENDLY])].copy()


def fetch_prediction_for_event(event_id: int, cfg: dict | None = None) -> dict | None:
    """Prédiction pour un event_id précis (recherche dans le flux global)."""
    if cfg is None:
        cfg = load_config()
    raw = _get_all_pages("/predictions/", {}, cfg)
    for p in raw:
        if p.get("event", {}).get("id") == event_id:
            return _parse_prediction(p)
    return None


def blend_predictions(
    dc_probs: dict,
    bzzo_probs: dict | None,
    dc_weight: float = 0.6,
) -> dict:
    """
    Combine Dixon-Coles et bzzoiro CatBoost par moyenne pondérée.

    dc_probs   : {"home_win": p, "draw": p, "away_win": p, "over_2.5": p, "btts": p}
    bzzo_probs : {"prob_home": p, "prob_draw": p, "prob_away": p, "prob_over_25": p, "prob_btts": p}
    """
    if bzzo_probs is None:
        return dc_probs

    bzzo_w = 1 - dc_weight

    blended = {
        "home_win": dc_weight * dc_probs["home_win"] + bzzo_w * bzzo_probs.get("prob_home", dc_probs["home_win"]),
        "draw":     dc_weight * dc_probs["draw"]     + bzzo_w * bzzo_probs.get("prob_draw", dc_probs["draw"]),
        "away_win": dc_weight * dc_probs["away_win"] + bzzo_w * bzzo_probs.get("prob_away", dc_probs["away_win"]),
        "over_2.5": dc_weight * dc_probs["over_2.5"] + bzzo_w * bzzo_probs.get("prob_over_25", dc_probs["over_2.5"]),
        "btts":     dc_weight * dc_probs["btts"]     + bzzo_w * bzzo_probs.get("prob_btts", dc_probs["btts"]),
        "expected_home": dc_probs.get("expected_home", bzzo_probs.get("xg_home", 1.3)),
        "expected_away": dc_probs.get("expected_away", bzzo_probs.get("xg_away", 1.0)),
        "source": "blended",
    }

    # Renormaliser 1X2
    total = blended["home_win"] + blended["draw"] + blended["away_win"]
    if total > 0:
        for k in ["home_win", "draw", "away_win"]:
            blended[k] /= total

    return blended


def save_predictions(df: pd.DataFrame):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / "predictions_bzzoiro.parquet"
    df.to_parquet(path, index=False)
    print(f"Prédictions sauvegardées : {path} ({len(df)} lignes)")


if __name__ == "__main__":
    cfg = load_config()
    print("Fetching prédictions internationales...")
    df = fetch_international_predictions(cfg)
    print(f"  {len(df)} prédictions récupérées")
    if not df.empty:
        print(df[["event_date", "home_team", "away_team", "prob_home", "prob_draw", "prob_away", "xg_home", "xg_away"]].head(10).to_string(index=False))
        save_predictions(df)
