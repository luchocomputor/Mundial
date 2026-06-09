"""
Calibration empirique du poids DC vs bzzoiro sur WC 2022.
Compare Brier Score de DC seul, bzzoiro seul, et blend à différents ratios.
Écrit le meilleur ratio dans config.yaml.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.dixon_coles import DixonColesModel
from pipeline.features import build_training_data, get_altitude_adjustment
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def brier(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return brier_score_loss(y_true, y_pred)


def compute_outcomes(row):
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    return {
        "home_win": int(hg > ag),
        "draw": int(hg == ag),
        "away_win": int(hg < ag),
        "over_2.5": int(hg + ag > 2.5),
        "btts": int(hg > 0 and ag > 0),
    }


def run_calibration():
    print("Calibration du blend DC vs bzzoiro sur WC 2022...\n")

    # Charger modèle DC (entraîné sur tout sauf WC 2022)
    raw = ROOT / "data" / "raw"
    all_df = pd.read_parquet(raw / "all_matches.parquet")
    all_df["date"] = pd.to_datetime(all_df["date"], utc=True)
    pre2022 = all_df[all_df["date"].dt.year < 2022].copy()
    training = build_training_data(pre2022, reference_date=pd.Timestamp("2022-11-20", tz="UTC"))
    model = DixonColesModel()
    model.fit(training, reference_date=pd.Timestamp("2022-11-20", tz="UTC"))

    wc = pd.read_parquet(raw / "wc_all.parquet")
    wc["date"] = pd.to_datetime(wc["date"], utc=True)
    wc2022 = wc[(wc["date"].dt.year == 2022) & (wc["home_goals"].notna())].copy()

    if wc2022.empty:
        print("Pas de données WC 2022 disponibles pour calibrer.")
        return

    # Prédictions bzzoiro pour WC 2022 (si dispo dans predictions_bzzoiro.parquet)
    bzzo_path = raw / "predictions_bzzoiro.parquet"
    bzzo_map = {}
    if bzzo_path.exists():
        bp = pd.read_parquet(bzzo_path)
        bp = bp[bp["league_id"] == 27]
        for _, r in bp.iterrows():
            key = (r["home_team"], r["away_team"])
            bzzo_map[key] = r.to_dict()

    dc_preds, bzzo_preds, actuals = [], [], []
    skipped = 0

    for _, row in wc2022.iterrows():
        home, away = row["home_team"], row["away_team"]
        try:
            dc = model.predict_outcomes(home, away)
        except Exception:
            skipped += 1
            continue

        bzzo = bzzo_map.get((home, away))

        dc_preds.append(dc)
        bzzo_preds.append(bzzo)
        actuals.append(compute_outcomes(row))

    print(f"  Matchs évalués: {len(actuals)} (skipped: {skipped})")
    if len(actuals) < 5:
        print("  Pas assez de matchs pour calibrer.")
        return

    markets = ["home_win", "draw", "away_win", "over_2.5", "btts"]
    bzzo_keys = ["prob_home", "prob_draw", "prob_away", "prob_over_25", "prob_btts"]

    has_bzzo = sum(1 for b in bzzo_preds if b is not None)
    print(f"  Matchs avec prédictions bzzoiro: {has_bzzo}/{len(bzzo_preds)}\n")

    print(f"{'Ratio DC':>10} | {'home_win':>9} {'draw':>9} {'away_win':>9} {'over_2.5':>9} {'btts':>9} | {'MEAN':>9}")
    print("─" * 80)

    best_ratio, best_score = 1.0, float("inf")

    for dc_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        bzzo_w = 1.0 - dc_w
        scores = []
        for market, bzzo_key in zip(markets, bzzo_keys):
            y_true = [a[market] for a in actuals]
            y_pred = []
            for dc, bzzo in zip(dc_preds, bzzo_preds):
                p_dc = dc.get(market, 0.33)
                p_bzzo = bzzo.get(bzzo_key, p_dc) if bzzo else p_dc
                y_pred.append(dc_w * p_dc + bzzo_w * p_bzzo)
            scores.append(brier(y_true, y_pred))

        mean_bs = np.mean(scores)
        if mean_bs < best_score:
            best_score = mean_bs
            best_ratio = dc_w

        mark = " ← optimal" if dc_w == best_ratio else ""
        print(f"  DC={dc_w:.1f}     | {scores[0]:9.4f} {scores[1]:9.4f} {scores[2]:9.4f} {scores[3]:9.4f} {scores[4]:9.4f} | {mean_bs:9.4f}{mark}")

    print(f"\nMeilleur ratio DC: {best_ratio:.1f} (Brier moyen: {best_score:.4f})")

    # Écrire dans config.yaml
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["blend"]["dc_weight"] = best_ratio
    with open(ROOT / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"config.yaml mis à jour: blend.dc_weight = {best_ratio}")


if __name__ == "__main__":
    run_calibration()
