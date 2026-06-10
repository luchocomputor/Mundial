#!/usr/bin/env python3
"""
Audit complet : tests modèles, backtest, scan live, simulation bankroll.
Génère data/processed/full_audit_report.json et .md
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.go_nogo import GO_NOGO_CRITERIA, evaluate
from backtest.run_walkforward import BacktestConfig, run_full_backtest, simulate_bets_with_real_odds
from evaluation.clv import bootstrap_clv_significance, clv
from evaluation.metrics import compute_all_metrics, ranked_probability_score
from evaluation.walkforward import compute_outcomes
from models.base import MatchContext
from models.dixon_coles import DixonColesModel
from models.ratings import EloRating
from models.bivariate_poisson import BivariatePoissonModel
from models.ensemble import StackingEnsemble, build_default_ensemble
from models.calibration import MarketCalibrator
from odds.devig import devig
from pipeline.config import load_config
from pipeline.features import build_training_data, load_raw_matches
from pipeline.model_loader import load_production_model
from pipeline.placeholder import filter_real_teams
from pipeline.signal_guards import GuardStats
from pipeline.value_detector import scan_value_bets

REPORT_DIR = ROOT / "data" / "processed" / "audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> list[str]:
    return ["", f"## {title}", ""]


def train_all_models(matches: pd.DataFrame, ref_date: pd.Timestamp) -> dict:
    """Entraîne tous les modèles et retourne métriques de fit."""
    training = build_training_data(
        matches[matches["status"] == "finished"],
        reference_date=ref_date,
    )
    results = {}

    models = {
        "elo": EloRating(),
        "dixon_coles": DixonColesModel(),
        "bivariate_poisson": BivariatePoissonModel(),
    }

    for name, model in models.items():
        try:
            model.fit(training, reference_date=ref_date)
            results[name] = {"status": "ok", "n_teams": len(getattr(model, "teams", model.ratings))}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    # Ensemble
    try:
        ensemble = build_default_ensemble()
        ensemble.fit(training, reference_date=ref_date)
        ensemble.save()
        results["ensemble"] = {"status": "ok", "models": ensemble.model_names}
    except Exception as e:
        results["ensemble"] = {"status": "error", "error": str(e)}

    # Save DC
    try:
        dc = models["dixon_coles"]
        if results["dixon_coles"]["status"] == "ok":
            dc.save()
    except Exception:
        pass

    return results, models


def evaluate_models_on_test(
    models: dict,
    test_df: pd.DataFrame,
) -> dict:
    """Évalue RPS/log-loss de chaque modèle sur matchs test."""
    eval_results = {}

    for name, model in models.items():
        predictions = []
        outcomes = []
        for _, row in test_df.iterrows():
            ctx = MatchContext(
                is_neutral=bool(row.get("is_neutral", False)),
                venue=str(row.get("venue", "")),
                is_friendly=bool(row.get("is_friendly", False)),
            )
            try:
                pred = model.predict_outcomes(row["home_team"], row["away_team"], context=ctx)
                pred_dict = pred.to_dict() if hasattr(pred, "to_dict") else pred
                predictions.append(pred_dict)
                outcomes.append(compute_outcomes(row))
            except Exception:
                continue

        if predictions:
            metrics = compute_all_metrics(predictions, outcomes)
            eval_results[name] = {
                "n_predictions": len(predictions),
                "rps": metrics.rps,
                "ece": metrics.ece,
                "log_loss": metrics.log_loss_binary,
            }
        else:
            eval_results[name] = {"n_predictions": 0}

    return eval_results


def scan_live_signals(cfg) -> list[dict]:
    """Scan CDM 2026 avec cotes existantes."""
    wc_path = ROOT / "data" / "raw" / "wc_all.parquet"
    all_path = ROOT / "data" / "raw" / "all_matches.parquet"
    path = wc_path if wc_path.exists() else all_path
    if not path.exists():
        return []

    wc = pd.read_parquet(path)
    wc2026 = wc[wc["date"].dt.year == 2026]
    matches = filter_real_teams(wc2026).sort_values("date")

    odds_path = ROOT / "data" / "raw" / "odds_wc2026.json"
    odds = {}
    if odds_path.exists():
        raw = json.loads(odds_path.read_text())
        odds = {int(k): v for k, v in raw.items()}

    bzzo_map = {}
    preds_path = ROOT / "data" / "raw" / "predictions_wc2026.parquet"
    if preds_path.exists():
        bdf = pd.read_parquet(preds_path)
        bzzo_map = {int(r["event_id"]): r.to_dict() for _, r in bdf.iterrows()}

    # Charger modèle Elo (prod par défaut)
    model = load_production_model(prefer_elo=True)

    match_list = []
    for _, row in matches.iterrows():
        fid = int(row["fixture_id"])
        if fid not in odds:
            continue
        match_list.append({
            "fixture_id": fid,
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "venue": row.get("venue", ""),
            "is_friendly": bool(row.get("is_friendly", False)),
            "is_neutral": bool(row.get("is_neutral", False)),
            "group": row.get("group_name", ""),
        })

    bets, guard_stats = scan_value_bets(
        match_list, model, odds, cfg, cfg.bankroll_initial, bzzo_map
    )
    return bets, guard_stats


def simulate_bankroll_scenario(bets: list[dict], starting_bankroll: float = 200.0) -> dict:
    """Simule l'évolution du bankroll si tous les signaux avaient été joués."""
    bankroll = starting_bankroll
    peak = bankroll
    history = []
    total_staked = 0

    for i, bet in enumerate(bets):
        stake = bet.get("stake_eur", 0)
        if stake <= 0 or stake > bankroll:
            stake = min(stake, bankroll)
        total_staked += stake
        # Résultat inconnu → on simule en espérance (EV)
        p = bet.get("p_model", 0.5)
        cote = bet.get("cote", 2.0)
        ev_profit = stake * (p * (cote - 1) - (1 - p))  # espérance mathématique
        bankroll += ev_profit
        peak = max(peak, bankroll)
        history.append({
            "bet_num": i + 1,
            "match": f"{bet['home_team']} vs {bet['away_team']}",
            "date": str(bet.get("date", ""))[:16],
            "market": f"{bet['market']} {bet['side']}",
            "cote": bet["cote"],
            "p_model": round(bet["p_model"], 4),
            "edge": round(bet["edge"], 4),
            "kelly_pct": bet.get("kelly_pct", 0),
            "stake_eur": stake,
            "ev_profit": round(ev_profit, 2),
            "bankroll_after": round(bankroll, 2),
        })

    return {
        "starting_bankroll": starting_bankroll,
        "final_bankroll_ev": round(bankroll, 2),
        "ev_return_pct": round((bankroll - starting_bankroll) / starting_bankroll * 100, 2),
        "total_staked": round(total_staked, 2),
        "n_bets": len(bets),
        "max_exposure_pct": round(total_staked / starting_bankroll * 100, 1) if starting_bankroll else 0,
        "history": history,
    }


def run_audit():
    cfg = load_config()
    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "bankroll": cfg.bankroll_initial,
            "min_edge": cfg.model.min_edge_threshold,
            "kelly_fraction": cfg.model.kelly_fraction,
            "max_kelly_bet": cfg.model.max_kelly_bet,
            "devig_method": cfg.model.devig_method,
            "max_market_divergence": cfg.model.max_market_divergence,
            "market_anchor_weight": cfg.model.market_anchor_weight,
            "production_mode": cfg.model.production_mode,
        },
        "sections": {},
    }
    lines = [
        "# Rapport d'audit complet — Mundial CDM 2026",
        f"*Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}*",
        "",
    ]

    # --- 1. Données ---
    lines += section("1. État des données")
    try:
        matches = load_raw_matches()
        odds_path = ROOT / "data" / "raw" / "odds_history.parquet"
        odds_df = pd.read_parquet(odds_path) if odds_path.exists() else pd.DataFrame()
        odds_json = ROOT / "data" / "raw" / "odds_wc2026.json"
        n_live_odds = len(json.loads(odds_json.read_text())) if odds_json.exists() else 0

        data_stats = {
            "n_matches": len(matches),
            "n_finished": int((matches["status"] == "finished").sum()),
            "n_upcoming": int((matches["status"] == "upcoming").sum()),
            "n_teams": len(set(matches["home_team"]) | set(matches["away_team"])),
            "date_min": str(matches["date"].min()),
            "date_max": str(matches["date"].max()),
            "competition_types": matches["competition_type"].value_counts().to_dict() if "competition_type" in matches.columns else {},
            "n_odds_history": len(odds_df),
            "n_live_odds_wc2026": n_live_odds,
        }
        report["sections"]["data"] = data_stats
        lines.append(f"- **{data_stats['n_matches']}** matchs ({data_stats['n_finished']} terminés, {data_stats['n_upcoming']} à venir)")
        lines.append(f"- **{data_stats['n_teams']}** équipes distinctes")
        lines.append(f"- Période : {data_stats['date_min'][:10]} → {data_stats['date_max'][:10]}")
        lines.append(f"- Cotes historiques backtest : **{data_stats['n_odds_history']}** lignes")
        lines.append(f"- Cotes live CDM 2026 : **{data_stats['n_live_odds_wc2026']}** matchs")
        if data_stats["n_odds_history"] < 100:
            lines.append("- ⚠ Couverture cotes historiques **insuffisante** pour backtest fiable (besoin ≥500 paris simulés)")
    except Exception as e:
        report["sections"]["data"] = {"error": str(e)}
        lines.append(f"Erreur données : {e}")

    # --- 2. Entraînement ---
    lines += section("2. Entraînement des modèles")
    try:
        matches = load_raw_matches()
        ref_date = pd.Timestamp("2025-12-31", tz="UTC")
        train_results, models = train_all_models(matches, ref_date)
        report["sections"]["training"] = train_results
        for name, res in train_results.items():
            status = "✓" if res.get("status") == "ok" else "✗"
            lines.append(f"- {status} **{name}** : {res}")
    except Exception as e:
        report["sections"]["training"] = {"error": str(e), "trace": traceback.format_exc()}
        lines.append(f"Erreur entraînement : {e}")

    # --- 3. Évaluation out-of-sample CDM 2022 ---
    lines += section("3. Évaluation out-of-sample (CDM 2022)")
    try:
        test_2022 = matches[
            (matches["date"].dt.year == 2022) & (matches["status"] == "finished")
        ]
        if len(test_2022) > 0 and models:
            eval_results = evaluate_models_on_test(models, test_2022)
            report["sections"]["eval_2022"] = eval_results
            lines.append(f"Matchs test CDM 2022 : **{len(test_2022)}**")
            lines.append("")
            lines.append("| Modèle | N pred | RPS | ECE over_2.5 | ECE btts |")
            lines.append("|--------|--------|-----|--------------|----------|")
            for name, res in eval_results.items():
                rps = f"{res.get('rps', 0):.4f}" if res.get("rps") else "—"
                ece_ou = res.get("ece", {}).get("over_2.5", "—")
                ece_bt = res.get("ece", {}).get("btts", "—")
                ece_ou = f"{ece_ou:.4f}" if isinstance(ece_ou, float) else "—"
                ece_bt = f"{ece_bt:.4f}" if isinstance(ece_bt, float) else "—"
                lines.append(f"| {name} | {res.get('n_predictions', 0)} | {rps} | {ece_ou} | {ece_bt} |")
        else:
            lines.append("Pas assez de matchs CDM 2022 pour évaluation.")
    except Exception as e:
        lines.append(f"Erreur évaluation : {e}")

    # --- 4. Backtest walk-forward ---
    lines += section("4. Backtest walk-forward (train ≤2021, test 2022)")
    try:
        bt_report = run_full_backtest(BacktestConfig(train_end_year=2021, test_year=2022))
        report["sections"]["backtest"] = bt_report
        if "error" not in bt_report:
            lines.append(f"- Matchs test : **{bt_report.get('n_test_matches', 0)}**")
            lines.append(f"- Paris simulés : **{bt_report.get('n_bets', 0)}**")
            lines.append(f"- ROI simulé : **{bt_report.get('roi_pct', 'N/A')}%**")
            lines.append(f"- Hit rate : **{bt_report.get('hit_rate', 'N/A')}%**")
            lines.append(f"- CLV moyen : **{bt_report.get('clv_mean', 'N/A')}**")
            lines.append(f"- Beat the close : **{bt_report.get('beat_close_pct', 'N/A')}**")
            lines.append(f"- Drawdown max : **{bt_report.get('max_drawdown_pct', 'N/A')}%**")
            lines.append(f"- Décision Go/No-Go : **{bt_report.get('go_nogo', 'N/A')}**")
            if bt_report.get("n_bets", 0) < GO_NOGO_CRITERIA["min_bets"]:
                lines.append(f"- ⚠ Seulement {bt_report.get('n_bets')} paris (< {GO_NOGO_CRITERIA['min_bets']} requis) — résultats non conclusifs")
        else:
            lines.append(f"Backtest : {bt_report['error']}")
    except Exception as e:
        report["sections"]["backtest"] = {"error": str(e), "trace": traceback.format_exc()}
        lines.append(f"Erreur backtest : {e}")

    # --- 5. Scan live CDM 2026 ---
    lines += section("5. Signaux value bets CDM 2026 (cotes actuelles)")
    try:
        bets, guard_stats = scan_live_signals(cfg)
        report["sections"]["live_signals"] = {
            "bets": bets,
            "guard_stats": {
                "accepted": guard_stats.accepted,
                "rejected_divergence": guard_stats.rejected_divergence,
            },
        }
        lines.append(
            f"**{len(bets)}** signaux plausibles (edge > {cfg.model.min_edge_threshold*100:.0f}%, "
            f"garde-fous actifs, divergence max {cfg.model.max_market_divergence:.0%})"
        )
        lines.append(
            f"- Garde-fous : {guard_stats.accepted} acceptés, "
            f"**{guard_stats.rejected_divergence}** rejetés (divergence > {cfg.model.max_market_divergence:.0%})"
        )
        lines.append("")
        if bets:
            lines.append("| Match | Date | Marché | Cote | Modèle | Edge | Kelly | Mise |")
            lines.append("|-------|------|--------|------|--------|------|-------|------|")
            for b in bets:
                match = f"{b['home_team']} vs {b['away_team']}"[:25]
                date = str(b.get("date", ""))[:10]
                lines.append(
                    f"| {match} | {date} | {b['market']} {b['side']} | {b['cote']:.2f} | "
                    f"{b['p_model']*100:.1f}% | +{b['edge']*100:.1f}% | {b['kelly_pct']:.1f}% | {b['stake_eur']:.2f}€ |"
                )
        else:
            lines.append("Aucun signal au seuil actuel.")
    except Exception as e:
        report["sections"]["live_signals"] = {"error": str(e)}
        lines.append(f"Erreur scan : {e}")
        bets = []
        guard_stats = GuardStats()

    # --- 6. Simulation bankroll ---
    lines += section(f"6. Simulation bankroll (départ {cfg.bankroll_initial}€)")
    try:
        if bets:
            sim = simulate_bankroll_scenario(bets, cfg.bankroll_initial)
            report["sections"]["bankroll_simulation"] = sim
            lines.append(f"- Bankroll départ : **{sim['starting_bankroll']}€**")
            lines.append(f"- Mise totale suggérée : **{sim['total_staked']:.2f}€** ({sim['max_exposure_pct']:.1f}% du bankroll)")
            lines.append(f"- Bankroll final (espérance math.) : **{sim['final_bankroll_ev']:.2f}€** ({sim['ev_return_pct']:+.1f}%)")
            lines.append("")
            lines.append("### Détail par pari")
            lines.append("")
            lines.append("| # | Match | Marché | Cote | Edge | Mise | EV profit | Bankroll |")
            lines.append("|---|-------|--------|------|------|------|-----------|----------|")
            for h in sim["history"]:
                lines.append(
                    f"| {h['bet_num']} | {h['match'][:22]} | {h['market']} | {h['cote']:.2f} | "
                    f"+{h['edge']*100:.1f}% | {h['stake_eur']:.2f}€ | {h['ev_profit']:+.2f}€ | {h['bankroll_after']:.2f}€ |"
                )
            lines.append("")
            lines.append("*Note : EV profit = espérance mathématique (pas le résultat réel). Les matchs n'ont pas encore eu lieu.*")
            lines.append("*⚠ Simulation sans gate de validation — NE PAS interpréter comme ROI réel.*")
        else:
            lines.append("Pas de signaux → pas de simulation.")
    except Exception as e:
        lines.append(f"Erreur simulation : {e}")

    # --- 7. Verdict ---
    lines += section("7. Verdict et recommandations")
    go_nogo = report.get("sections", {}).get("backtest", {}).get("go_nogo", "NO_GO")
    n_bets_bt = report.get("sections", {}).get("backtest", {}).get("n_bets", 0)
    lines.append(f"- **Décision backtest** : {go_nogo}")
    lines.append(f"- **Mode recommandé** : {'Paper trading uniquement' if go_nogo != 'GO_PAPER_TRADING' else 'Paper trading → argent réel après 4-6 sem CLV+'}")
    lines.append("- **Ne pas miser d'argent réel** tant que CLV live n'est pas positif sur ≥4 semaines")
    if n_bets_bt < 50:
        lines.append("- **Action prioritaire** : élargir les ligues (qualifs, Nations League) et backfill cotes historiques")
    lines.append("- **CDM 2026** : utiliser `python refresh.py --full` quotidiennement + `python scan_now.py --paper`")

    # Save
    (REPORT_DIR / "full_audit_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    (REPORT_DIR / "full_audit_report.md").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\n\nRapports sauvegardés dans {REPORT_DIR}/")
    return report


if __name__ == "__main__":
    run_audit()
