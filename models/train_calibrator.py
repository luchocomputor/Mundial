"""Calibre les probas du modèle (isotonic par marché) sur prédictions OOS.

Le modèle sur-cote les outsiders (Qatar 22% vs Canada) → faux edge. On génère des
prédictions out-of-sample en walk-forward (prédire AVANT de mettre à jour les
ratings → pas de lookahead) sur les 3671 matchs, puis on fit une calibration
isotonic par marché. Split temporel pour mesurer l'ECE honnêtement, puis fit final
sur tout → models/artifacts/calibrator.pkl (chargé auto par value_detector).

Usage : python -m models.train_calibrator
"""

from __future__ import annotations

import numpy as np

from models.base import MatchContext
from models.calibration import MarketCalibrator
from models.ratings import EloRating
from models.train_elo import build_training

# On ne calibre QUE ce qui est cassé. Diagnostic OOS : le 1X2 home/away sur-cote
# les outsiders (ECE 0.15/0.13) et BTTS dérive un peu (0.056) → à calibrer. En
# revanche le nul (ECE 0.015) et les O/U Poisson (over_2.5 à 0.008 !) sont déjà
# excellents → les calibrer ne fait que les dégrader (overfit). On les laisse bruts.
MARKETS = ["home_win", "away_win", "btts"]
WARMUP = 600


def gen_oos():
    """Walk-forward : (pred OOS, outcome réalisé) pour chaque match après warmup."""
    train = build_training()
    m = EloRating()
    preds, outs = [], []
    for i, row in enumerate(train.itertuples(index=False)):
        h, a = row.home_team, row.away_team
        sh, sa = int(row.home_goals), int(row.away_goals)
        neu = bool(getattr(row, "is_neutral", False))
        if i >= WARMUP:
            p = m.predict_outcomes(h, a, MatchContext(is_neutral=neu))
            pd_ = p.to_dict() if hasattr(p, "to_dict") else dict(
                home_win=p.home_win, draw=p.draw, away_win=p.away_win)
            tot = sh + sa
            outs.append({
                "home_win": 1.0 if sh > sa else 0.0,
                "away_win": 1.0 if sh < sa else 0.0,
                "over_1.5": 1.0 if tot > 1.5 else 0.0,
                "over_2.5": 1.0 if tot > 2.5 else 0.0,
                "over_3.5": 1.0 if tot > 3.5 else 0.0,
                "btts": 1.0 if (sh > 0 and sa > 0) else 0.0,
            })
            preds.append(pd_)
        m.update(h, a, sh, sa, is_neutral=neu)
    return preds, outs


def ece(y_true, y_prob, bins=10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    tot, n = 0.0, len(y_true)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        msk = (y_prob >= lo) & ((y_prob < hi) if b < bins - 1 else (y_prob <= hi))
        if msk.sum():
            tot += abs(y_prob[msk].mean() - y_true[msk].mean()) * msk.sum()
    return tot / n if n else float("nan")


def run():
    preds, outs = gen_oos()
    print(f"Prédictions OOS : {len(preds)} matchs")
    cut = int(len(preds) * 0.8)
    tr_p, tr_o = preds[:cut], outs[:cut]
    te_p, te_o = preds[cut:], outs[cut:]

    cal = MarketCalibrator()
    for mk in MARKETS:
        cal.fit(tr_p, tr_o, mk, method="isotonic")

    print("\nECE sur le set test (20% le plus récent) — avant → après :")
    for mk in MARKETS:
        yp = np.array([p.get(mk, np.nan) for p in te_p])
        yt = np.array([o[mk] for o in te_o])
        msk = ~np.isnan(yp)
        before = ece(yt[msk], yp[msk])
        ypc = np.array([cal.calibrate(p).get(mk, p.get(mk, np.nan)) for p in te_p])[msk]
        after = ece(yt[msk], ypc)
        arrow = "✓" if after < before else "≈"
        print(f"  {mk:10s} {before:.3f} → {after:.3f}  {arrow}")

    cal_full = MarketCalibrator()
    for mk in MARKETS:
        cal_full.fit(preds, outs, mk, method="isotonic")
    cal_full.save()
    print("\n✅ calibrator.pkl sauvegardé (chargé auto par le value detector)")


if __name__ == "__main__":
    run()
