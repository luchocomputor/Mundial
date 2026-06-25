"""Backtest honnête de l'Elo tennis vs la cote de CLÔTURE Pinnacle.

Pinnacle = bookmaker le plus sharp (faible marge, pas de limite sérieuse) → sa cote
de clôture de-viggée est le proxy standard de la « vraie » proba. Si l'Elo ne bat pas
Pinnacle en Brier/log-loss, il n'y a pas d'alpha (cf leçon du marché CDM efficace).

Walk-forward SANS lookahead : l'Elo est séquentiel — pour chaque match on prédit avec
les ratings courants PUIS on met à jour. On n'évalue que les matchs récents (start) où
les deux joueurs ont assez d'historique et où la cote Pinnacle existe.

Réutilise les formules du modèle de prod (models.tennis_elo) — pas de divergence.

    python -m backtest.backtest_tennis --tour atp --start 20240101
"""

from __future__ import annotations

import argparse
import math
from glob import glob

import numpy as np
import pandas as pd

from models.tennis_elo import (DATA_DIR, INITIAL_RATING, SURFACE_BLEND, _NO_CONTEST,
                               _SURFACES, _expected, _k, player_key)


def _load_with_odds(tour: str) -> pd.DataFrame:
    """Matchs tennis-data avec cote de clôture Pinnacle (PSW/PSL) + segments."""
    keep = ["Date", "Surface", "Winner", "Loser", "Comment", "PSW", "PSL",
            "WRank", "LRank", "Series", "Round"]
    frames = []
    for f in sorted(glob(str(DATA_DIR / tour / "*.xlsx"))):
        df = pd.read_excel(f)
        if not {"Winner", "Loser", "Date", "PSW", "PSL"}.issubset(df.columns):
            continue
        for c in keep:
            if c not in df.columns:
                df[c] = None
        frames.append(df[keep].copy())
    d = pd.concat(frames, ignore_index=True)
    d["date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["yyyymmdd"] = d["date"].dt.strftime("%Y%m%d").astype(float).fillna(0).astype(int)
    return d.sort_values(["yyyymmdd"], kind="stable").reset_index(drop=True)


def _eff(ratings, by_surf, key, surface):
    overall = ratings.get(key, INITIAL_RATING)
    if not surface:
        return overall
    sr = by_surf.get((key, surface), overall)
    a = SURFACE_BLEND.get(surface, 0.5)
    return a * sr + (1 - a) * overall


def records(tour: str = "atp", start: int = 20240101, min_matches: int = 15) -> pd.DataFrame:
    """Walk-forward sans lookahead → un record par match éligible :
    p_model / p_market (proba que le VAINQUEUR réel gagne) + segments (rang, série…)."""
    df = _load_with_odds(tour)
    ratings: dict[str, float] = {}
    by_surf: dict[tuple, float] = {}
    n_over: dict[str, int] = {}
    n_surf: dict[tuple, int] = {}
    rows = []

    for r in df.itertuples(index=False):
        if isinstance(r.Comment, str) and _NO_CONTEST.search(r.Comment):
            continue
        wk, lk = player_key(r.Winner), player_key(r.Loser)
        if not wk or not lk or wk == lk:
            continue
        surface = _SURFACES.get(str(r.Surface).strip().lower(), "")
        pw = _expected(_eff(ratings, by_surf, wk, surface), _eff(ratings, by_surf, lk, surface))

        ok_odds = (isinstance(r.PSW, (int, float)) and isinstance(r.PSL, (int, float))
                   and r.PSW > 1 and r.PSL > 1 and not math.isnan(r.PSW) and not math.isnan(r.PSL))
        if (r.yyyymmdd >= start and n_over.get(wk, 0) >= min_matches
                and n_over.get(lk, 0) >= min_matches and ok_odds):
            iw, il = 1 / r.PSW, 1 / r.PSL
            try:
                maxrank = max(float(r.WRank), float(r.LRank))
            except (TypeError, ValueError):
                maxrank = float("nan")
            rows.append({"tour": tour, "date": int(r.yyyymmdd), "p_model": pw,
                         "p_market": iw / (iw + il), "psw": float(r.PSW), "psl": float(r.PSL),
                         "maxrank": maxrank, "series": r.Series, "round": r.Round})

        # mise à jour Elo (après prédiction → pas de lookahead)
        ratings[wk] = ratings.get(wk, INITIAL_RATING) + _k(n_over.get(wk, 0)) * (1 - _expected(ratings.get(wk, INITIAL_RATING), ratings.get(lk, INITIAL_RATING)))
        ratings[lk] = ratings.get(lk, INITIAL_RATING) + _k(n_over.get(lk, 0)) * (_expected(ratings.get(lk, INITIAL_RATING), ratings.get(wk, INITIAL_RATING)) - 1)
        n_over[wk] = n_over.get(wk, 0) + 1
        n_over[lk] = n_over.get(lk, 0) + 1
        if surface:
            rw = by_surf.get((wk, surface), INITIAL_RATING)
            rl = by_surf.get((lk, surface), INITIAL_RATING)
            es = _expected(rw, rl)
            by_surf[(wk, surface)] = rw + _k(n_surf.get((wk, surface), 0)) * (1 - es)
            by_surf[(lk, surface)] = rl + _k(n_surf.get((lk, surface), 0)) * (es - 1)
            n_surf[(wk, surface)] = n_surf.get((wk, surface), 0) + 1
            n_surf[(lk, surface)] = n_surf.get((lk, surface), 0) + 1

    return pd.DataFrame(rows)


def summarize(rec: pd.DataFrame, edge_thr: float = 0.05, kelly_frac: float = 0.25) -> dict:
    """Métriques (Brier/log-loss modèle vs Pinnacle + ROI simulé) sur un sous-ensemble."""
    if rec.empty:
        return {"samples": 0}
    pm, mk = rec["p_model"].to_numpy(), rec["p_market"].to_numpy()
    eps = 1e-9
    pnl = n_bets = 0
    for p, m, psw, psl in zip(pm, mk, rec["psw"], rec["psl"]):
        if p - m > edge_thr:                              # parier le vainqueur (gagne)
            pnl += max(0, ((psw - 1) * p - (1 - p)) / (psw - 1)) * kelly_frac * (psw - 1); n_bets += 1
        elif (1 - p) - (1 - m) > edge_thr:                # parier le perdant (perd)
            pl = 1 - p
            pnl += -max(0, ((psl - 1) * pl - (1 - pl)) / (psl - 1)) * kelly_frac; n_bets += 1
    return {
        "samples": len(rec), "acc": float(np.mean(pm > 0.5)),
        "brier_model": float(np.mean((pm - 1) ** 2)), "brier_market": float(np.mean((mk - 1) ** 2)),
        "logloss_model": float(-np.mean(np.log(np.clip(pm, eps, 1)))),
        "logloss_market": float(-np.mean(np.log(np.clip(mk, eps, 1)))),
        "n_bets": n_bets, "pnl_units": round(pnl, 2),
        "roi": round(pnl / n_bets, 4) if n_bets else 0.0,
    }


def backtest(tour: str = "atp", start: int = 20240101, min_matches: int = 15,
             edge_thr: float = 0.05, kelly_frac: float = 0.25) -> dict:
    r = summarize(records(tour, start, min_matches), edge_thr, kelly_frac)
    r["tour"] = tour
    return r


def segment_report(start: int = 20240101, min_matches: int = 15) -> None:
    """Stratifie l'écart modèle-vs-Pinnacle par classement et par série, pour TESTER
    si l'edge vit dans les coins moins surveillés (bas-classés, petits tournois).
    Verdict empirique : NON — Pinnacle reste sharp partout, et l'Elo se dégrade sur
    les bas-classés (data joueur trop fine). Le coin mou est le PIRE pour le modèle."""
    rec = pd.concat([records("atp", start, min_matches),
                     records("wta", start, min_matches)], ignore_index=True)

    def line(name, sub):
        s = summarize(sub)
        if s["samples"] < 50:
            print(f"  {name:22} n={s['samples']:>4}  (trop peu)"); return
        d = s["brier_market"] - s["brier_model"]  # >0 ⇒ modèle meilleur que Pinnacle
        print(f"  {name:22} n={s['samples']:>4}  Brier {s['brier_model']:.3f} vs "
              f"{s['brier_market']:.3f} (Δ{d:+.3f})  ROI {s['roi']:+.1%}"
              f"{'  << modèle bat marché' if d > 0 else ''}")

    print(f"=== Stratification modèle vs Pinnacle (ATP+WTA, ≥ {start}) ===")
    print("Δ = Brier_marché − Brier_modèle (positif = le modèle bat Pinnacle)\n")
    print("Par classement max du match :")
    line("les 2 top-20",     rec[rec.maxrank <= 20])
    line("max rang 21-50",   rec[(rec.maxrank > 20) & (rec.maxrank <= 50)])
    line("max rang 51-100",  rec[(rec.maxrank > 50) & (rec.maxrank <= 100)])
    line("max rang 101-200", rec[(rec.maxrank > 100) & (rec.maxrank <= 200)])
    line("max rang >200",    rec[rec.maxrank > 200])
    print("\nPar série :")
    for s in ["ATP250", "ATP500", "Masters 1000", "Grand Slam"]:
        line(s, rec[rec.series == s])
    print("\n→ Verdict : aucun segment où l'Elo bat Pinnacle. Le coin 'mou' (bas-classés) "
          "est le PIRE pour le modèle (data trop fine). Filon mort sur le match-winner.")


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _fit_platt(p_model: np.ndarray) -> tuple[float, float]:
    """Platt scaling : fit p_cal = σ(a + b·logit(p)) par MLE. Jeu d'entraînement
    SYMÉTRIQUE — chaque match donne 2 points : (logit(pw),1) et (logit(1−pw),0) —
    sinon tous les outcomes valent 1 (perspective vainqueur) et c'est dégénéré."""
    from scipy.optimize import minimize
    x = np.concatenate([_logit(p_model), _logit(1 - p_model)])
    y = np.concatenate([np.ones_like(p_model), np.zeros_like(p_model)])

    def nll(ab):
        z = ab[0] + ab[1] * x
        return float(np.mean(np.logaddexp(0, z) - y * z))

    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def calibration_test(tour: str = "both", split: int = 20240101) -> None:
    """Décompose le déficit du modèle : calibration vs discrimination. Fit Platt sur
    < split, évalue OOS sur ≥ split. Si le calibré rejoint Pinnacle → seule la
    calibration clochait ; sinon l'Elo est intrinsèquement moins informé."""
    tours = ["atp", "wta"] if tour == "both" else [tour]
    rec = pd.concat([records(t, start=20190101) for t in tours], ignore_index=True)
    tr, te = rec[rec.date < split], rec[rec.date >= split]
    a, b = _fit_platt(tr["p_model"].to_numpy())

    pm, mk = te["p_model"].to_numpy(), te["p_market"].to_numpy()
    pc = 1 / (1 + np.exp(-(a + b * _logit(pm))))
    eps = 1e-9

    def brier(p): return float(np.mean((p - 1) ** 2))
    def ll(p): return float(-np.mean(np.log(np.clip(p, eps, 1))))

    print(f"=== Calibration test ({'+'.join(tours)}, fit <{split}, test ≥{split} : "
          f"{len(te)} matchs) ===")
    print(f"  Platt : a={a:+.3f}, b={b:.3f}  (b<1 ⇒ modèle surconfiant, on tasse vers 0.5)\n")
    print(f"  {'':14}{'Brier':>9}{'LogLoss':>10}")
    print(f"  {'Elo brut':14}{brier(pm):>9.4f}{ll(pm):>10.4f}")
    print(f"  {'Elo calibré':14}{brier(pc):>9.4f}{ll(pc):>10.4f}")
    print(f"  {'Pinnacle':14}{brier(mk):>9.4f}{ll(mk):>10.4f}")
    gap0 = brier(pm) - brier(mk)
    gap1 = brier(pc) - brier(mk)
    closed = (1 - gap1 / gap0) if gap0 else 0
    print(f"\n  Écart Brier au marché : brut {gap0:+.4f} → calibré {gap1:+.4f} "
          f"({closed:.0%} de l'écart fermé par la seule calibration).")

    # ROI simulé vs Pinnacle : brut vs calibré (la vraie question pour le value-betting).
    roi_raw = summarize(te)                     # te["p_model"] = proba brute
    roi_cal = summarize(te.assign(p_model=pc))  # proba calibrée
    print(f"  ROI simulé (edge>5% vs Pinnacle, quart-Kelly) : "
          f"brut {roi_raw['roi']:+.1%} ({roi_raw['n_bets']}) → calibré {roi_cal['roi']:+.1%} ({roi_cal['n_bets']})")
    print("  → reste >0 en Brier ET ROI≤0 ⇒ l'Elo discrimine moins que Pinnacle (pas qu'un souci de calibration).")


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--start", type=int, default=20240101)
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--edge", type=float, default=0.05)
    ap.add_argument("--segments", action="store_true", help="stratifie par classement/série")
    ap.add_argument("--calibration", action="store_true", help="test calibration vs discrimination")
    args = ap.parse_args()
    if args.segments:
        segment_report(args.start, args.min_matches)
        return
    if args.calibration:
        calibration_test(split=args.start)
        return
    r = backtest(args.tour, args.start, args.min_matches, args.edge)
    print(f"=== Backtest {r['tour'].upper()}  (≥ {args.start}, {r['samples']} matchs évalués) ===")
    print(f"  Accuracy modèle (favori) : {r['acc']:.1%}")
    print(f"  Brier   modèle {r['brier_model']:.4f}  vs  Pinnacle {r['brier_market']:.4f}  "
          f"({'modèle MEILLEUR' if r['brier_model'] < r['brier_market'] else 'marché meilleur'})")
    print(f"  LogLoss modèle {r['logloss_model']:.4f}  vs  Pinnacle {r['logloss_market']:.4f}")
    print(f"  Simu paris (edge>{args.edge:.0%} vs Pinnacle, quart-Kelly) : "
          f"{r['n_bets']} paris, PnL {r['pnl_units']:+.2f} u, ROI {r['roi']:+.1%}")
    print("  → Si le modèle ne bat pas Pinnacle en Brier ET ROI≤0 : pas d'alpha (attendu sur tour principal).")


if __name__ == "__main__":
    _cli()
