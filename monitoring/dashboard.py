"""Dashboard CLI — CLV roulant 7j/30j, ROI, alertes."""

from __future__ import annotations

import argparse

from monitoring.paper_trading import paper_trading_report
from pipeline.production import load_latest_go_nogo
from pipeline.value_detector import compute_roi_from_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", action="store_true", help="Affiche CLV roulant")
    args = parser.parse_args()

    roi = compute_roi_from_log()
    paper = paper_trading_report()
    go = load_latest_go_nogo()

    print("=== Dashboard Mundial CDM 2026 ===")
    print(f"\nGo/No-Go : {go.value}")
    print(f"Mode : paper_only (voir config.yaml)")
    print(f"\nROI log : {roi}")
    print(f"\nPaper trading : {paper}")

    if args.rolling or True:
        clv7 = paper.get("clv_7d", {})
        clv30 = paper.get("clv_30d", {})
        print(f"\nCLV roulant 7j  : n={clv7.get('n', 0)}, mean={clv7.get('clv_mean')}")
        print(f"CLV roulant 30j : n={clv30.get('n', 0)}, mean={clv30.get('clv_mean')}")

        if clv7.get("clv_mean") is not None and clv7["clv_mean"] < 0:
            print("\n⚠ CLV 7j glissant négatif — observation uniquement")

    if paper.get("significant"):
        print("\n✓ CLV positif significatif — candidat après 4-6 semaines paper")
    elif paper.get("clv") == "pending":
        print("\n⏳ CLV en attente (cotes de clôture)")
    else:
        print("\n✗ CLV non significatif — pas d'argent réel")


if __name__ == "__main__":
    main()
