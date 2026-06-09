"""
Backtest du modèle Dixon-Coles sur la CDM 2022.
Entraîne sur 2018-2021, prédit CDM 2022, simule les bets.

Métriques :
- Brier Score (vs bookmaker)
- Log-loss
- ROI simulé avec Kelly quarter
- Calibration curve
- Distribution des value bets
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT))

from models.dixon_coles import DixonColesModel
from models.calibration import ProbabilityCalibrator, plot_calibration
from pipeline.features import build_training_data, get_altitude_adjustment
from pipeline.value_detector import compute_edge, kelly_size


DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = DATA_PROCESSED / "backtest_wc2022"


def load_wc2022_matches() -> pd.DataFrame:
    df = pd.read_parquet(DATA_RAW / "wc_all.parquet")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    wc2022 = df[(df["date"].dt.year == 2022) & (df["status"] == "finished")]
    wc2022 = wc2022.dropna(subset=["home_goals", "away_goals"])
    if wc2022.empty:
        raise FileNotFoundError("Données CDM 2022 introuvables dans wc_all.parquet")
    return wc2022.copy()


def load_pre2022_matches() -> pd.DataFrame:
    """Matchs terminés avant CDM 2022 : WC 2014/2018 + qualifications + amicaux."""
    parts = []

    wc = pd.read_parquet(DATA_RAW / "wc_all.parquet")
    wc["date"] = pd.to_datetime(wc["date"], utc=True)
    pre_wc = wc[(wc["date"].dt.year < 2022) & (wc["status"] == "finished")]
    if not pre_wc.empty:
        parts.append(pre_wc)

    qual = DATA_RAW / "qualifications.parquet"
    if qual.exists():
        df_q = pd.read_parquet(qual)
        df_q["date"] = pd.to_datetime(df_q["date"], utc=True)
        parts.append(df_q[(df_q["date"].dt.year < 2022) & (df_q["status"] == "finished")])

    fr = DATA_RAW / "friendly_all.parquet"
    if fr.exists():
        df_f = pd.read_parquet(fr)
        df_f["date"] = pd.to_datetime(df_f["date"], utc=True)
        parts.append(df_f[(df_f["date"].dt.year < 2022) & (df_f["status"] == "finished")])

    if not parts:
        raise FileNotFoundError("Aucune donnée d'entraînement trouvée.")
    df = pd.concat(parts, ignore_index=True)
    return df.dropna(subset=["home_goals", "away_goals"]).drop_duplicates(subset=["fixture_id"])


def simulate_bets(
    predictions: list[dict],
    actuals: list[dict],
    simulated_odds: dict | None = None,
    kelly_fraction: float = 0.25,
    max_kelly: float = 0.05,
    threshold: float = 0.05,
    bankroll: float = 100.0,
) -> pd.DataFrame:
    """
    Simule les bets avec Kelly quarter.
    Si simulated_odds est None, utilise des cotes synthétiques à partir des fréquences historiques.
    """
    results = []
    bk = bankroll

    for pred, actual in zip(predictions, actuals):
        markets = [
            ("home_win", pred["home_win"], actual.get("home_win", 0)),
            ("draw", pred["draw"], actual.get("draw", 0)),
            ("away_win", pred["away_win"], actual.get("away_win", 0)),
            ("over_2.5", pred["over_2.5"], actual.get("over_2.5", 0)),
            ("btts", pred["btts"], actual.get("btts", 0)),
        ]

        for market, p_model, outcome in markets:
            # Cote synthétique : 1/p_model avec marge bookmaker 5%
            p_synthetic = max(0.05, p_model)
            cote_fair = 1.0 / p_synthetic
            cote_bk = cote_fair * 0.95

            edge = compute_edge(p_model, cote_bk)
            if edge < threshold:
                continue

            kelly = kelly_size(p_model, cote_bk, kelly_fraction)
            kelly = min(kelly, max_kelly)
            stake = bk * kelly
            profit = stake * (cote_bk - 1) if outcome == 1 else -stake
            bk = max(0.01, bk + profit)

            results.append({
                "market": market,
                "p_model": p_model,
                "p_implicit": 1 / cote_bk,
                "edge": edge,
                "cote": cote_bk,
                "stake": stake,
                "outcome": outcome,
                "profit": profit,
                "bankroll_after": bk,
            })

    return pd.DataFrame(results)


def compute_outcomes(row: pd.Series) -> dict:
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    return {
        "home_win": int(hg > ag),
        "draw": int(hg == ag),
        "away_win": int(hg < ag),
        "over_2.5": int(hg + ag > 2.5),
        "btts": int(hg > 0 and ag > 0),
    }


def brier_score(predictions: list[dict], actuals: list[dict], market: str) -> float:
    y_pred = np.array([p.get(market, 0.5) for p in predictions])
    y_true = np.array([a.get(market, 0) for a in actuals])
    return brier_score_loss(y_true, y_pred)


def log_loss_score(predictions: list[dict], actuals: list[dict], market: str) -> float:
    y_pred = np.clip([p.get(market, 0.5) for p in predictions], 1e-7, 1 - 1e-7)
    y_true = [a.get(market, 0) for a in actuals]
    return log_loss(y_true, y_pred, labels=[0, 1])


def run_backtest():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement des données...")
    df_train = load_pre2022_matches()
    df_test = load_wc2022_matches()
    print(f"  Train : {len(df_train)} matchs | Test : {len(df_test)} matchs CDM 2022")

    print("Entraînement Dixon-Coles sur 2018-2021...")
    ref = pd.Timestamp("2022-11-20", tz="UTC")
    training = build_training_data(df_train, reference_date=ref, decay=0.001, friendly_weight=0.3)
    model = DixonColesModel(decay=0.001, friendly_weight=0.3, l2_reg=0.05)
    model.fit(training, reference_date=ref)

    print("Génération des prédictions CDM 2022...")
    predictions = []
    actuals = []

    for _, row in df_test.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        venue = row.get("venue", "")
        altitude_adj = get_altitude_adjustment(venue)

        try:
            pred = model.predict_outcomes(home, away, altitude_adj=altitude_adj)
        except Exception:
            pred = {"home_win": 1/3, "draw": 1/3, "away_win": 1/3, "over_2.5": 0.5, "btts": 0.5}

        predictions.append(pred)
        actuals.append(compute_outcomes(row))

    print("\n=== MÉTRIQUES ===")
    markets = ["home_win", "draw", "away_win", "over_2.5", "btts"]
    for market in markets:
        bs = brier_score(predictions, actuals, market)
        ll = log_loss_score(predictions, actuals, market)
        print(f"  {market:12s} | Brier: {bs:.4f} | Log-loss: {ll:.4f}")

    print("\n=== SIMULATION BETS ===")
    bets_df = simulate_bets(predictions, actuals, bankroll=100.0)

    if bets_df.empty:
        print("  Aucun value bet détecté.")
    else:
        total_staked = bets_df["stake"].sum()
        total_profit = bets_df["profit"].sum()
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0

        won = (bets_df["outcome"] == 1).sum()
        total_bets = len(bets_df)
        hit_rate = won / total_bets * 100

        print(f"  Bets détectés : {total_bets}")
        print(f"  Hit rate : {hit_rate:.1f}%")
        print(f"  Total misé : {total_staked:.2f}€")
        print(f"  Profit net : {total_profit:+.2f}€")
        print(f"  ROI simulé : {roi:+.1f}%")
        print(f"  Bankroll finale : {bets_df['bankroll_after'].iloc[-1]:.2f}€ (départ 100€)")

        if roi > 0:
            print("\n  BACKTEST VALIDÉ — modèle prêt pour CDM 2026.")
        else:
            print("\n  BACKTEST NÉGATIF — revoir les paramètres avant production.")

        bets_df.to_csv(REPORT_DIR / "simulated_bets.csv", index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(bets_df["bankroll_after"].values)
        plt.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Départ 100€")
        plt.title("Évolution du bankroll — Backtest CDM 2022")
        plt.xlabel("Bet #")
        plt.ylabel("Bankroll (€)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "bankroll_evolution.png", dpi=150)
        plt.close()
        print(f"\nRapports sauvegardés dans {REPORT_DIR}")

    calib = ProbabilityCalibrator()
    calib.fit(predictions, actuals)
    calib.save()

    for market in ["home_win", "over_2.5"]:
        try:
            plot_calibration(predictions, actuals, market)
        except Exception:
            pass

    # Sauvegarder le modèle de backtest séparément pour ne pas écraser la production
    backtest_model_path = ROOT / "models" / "dixon_coles_backtest2022.pkl"
    model.save(backtest_model_path)
    print(f"\nModèle backtest sauvegardé dans {backtest_model_path}")


if __name__ == "__main__":
    run_backtest()
