"""Value detector tennis : Elo surface × cotes Betclic « Vainqueur du match ».

Réutilise la logique edge/Kelly du foot (sport-agnostique). Spécificités tennis :
  - Marché 2-way (pas de nul) → on de-vig les 2 cotes pour la proba marché « fair »
    (l'edge brut `p_model − 1/cote` garde la vig → conservateur, c'est ce qu'on logge).
  - Tour ATP/WTA auto-détecté par couverture des noms dans chaque Elo.
  - Surface = gazon par défaut (saison Wimbledon) ; paramétrable.

⚠️ Comme pour le foot (cf marché CDM efficace), un edge affiché n'est PAS un edge réel
tant que le CLV n'est pas vert. Ici en plus : couverture data faible sur les bas-classés
(qualifs) → `known=False` ou `n` bas = proba peu fiable, à ne pas miser.

    python -m pipeline.tennis_value                 # scan des matchs tennis live
    python -m pipeline.tennis_value --min-edge 0.04
"""

from __future__ import annotations

import argparse

from models.tennis_elo import TennisElo, player_key
from pipeline.fetch_tennis_odds import list_tennis_matches
from pipeline.value_detector import compute_edge, kelly_size


def _load_models() -> dict[str, TennisElo]:
    out = {}
    for tour in ("atp", "wta"):
        try:
            out[tour] = TennisElo.load(tour)
        except Exception:
            pass
    return out


def _pick_tour(models: dict[str, TennisElo], a: str, b: str) -> tuple[str | None, TennisElo | None]:
    """Choisit l'Elo (atp/wta) où les deux joueurs sont les mieux couverts."""
    best, best_score = None, -1
    for tour, m in models.items():
        ka, kb = player_key(a), player_key(b)
        score = (ka in m.players) + (kb in m.players)
        if score > best_score:
            best, best_score = tour, score
    return (best, models.get(best)) if best_score > 0 else (None, None)


def _devig_2way(oa: float, ob: float) -> tuple[float, float]:
    ia, ib = 1.0 / oa, 1.0 / ob
    s = ia + ib
    return ia / s, ib / s


def scan(matches: list[dict] | None = None, surface: str = "Grass",
         min_edge: float = 0.03, kelly_fraction: float = 0.25) -> list[dict]:
    models = _load_models()
    if not models:
        raise RuntimeError("Aucun artefact Elo tennis. Lance d'abord "
                           "`python -m models.tennis_elo --tour atp|wta`.")
    rows = matches or list_tennis_matches()
    out = []
    for m in rows:
        od = m.get("winner_odds")
        if not od or len(od) != 2:
            continue
        (a, oa), (b, ob) = list(od.items())
        tour, model = _pick_tour(models, a, b)
        if model is None:
            continue
        pr = model.predict(a, b, surface=surface)
        fair_a, fair_b = _devig_2way(oa, ob)
        for side, p_model, cote, fair, n, ns, known in [
            (a, pr["p_a"], oa, fair_a, pr["n_a"], pr["n_a_surface"], pr["known_a"]),
            (b, pr["p_b"], ob, fair_b, pr["n_b"], pr["n_b_surface"], pr["known_b"])]:
            edge = compute_edge(p_model, cote)        # vs cote brute (vig incluse, conservateur)
            edge_fair = p_model - fair                # vs proba marché de-viggée
            if edge >= min_edge and known:
                out.append({
                    "tour": tour, "surface": surface, "match": m["label"],
                    "date": m["date"][:16], "side": side, "cote": cote,
                    "p_model": round(p_model, 3), "p_market": round(fair, 3),
                    "edge": round(edge, 3), "edge_fair": round(edge_fair, 3),
                    "kelly": round(kelly_size(p_model, cote, kelly_fraction), 4),
                    "n": n, "n_surface": ns,
                })
    return sorted(out, key=lambda r: r["edge"], reverse=True)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Value bets tennis (Elo × Betclic)")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--surface", default="Grass")
    args = ap.parse_args()
    bets = scan(surface=args.surface, min_edge=args.min_edge)
    if not bets:
        print("Aucune value bet sous le seuil (ou cotes/joueurs indisponibles).")
        return
    print(f"{len(bets)} value bet(s) — edge brut ≥ {args.min_edge:.0%} :\n")
    for b in bets:
        print(f"  [{b['tour'].upper()}] {b['match']}  ({b['date']})")
        print(f"     {b['side']} @ {b['cote']}  | modèle {b['p_model']:.1%} vs marché "
              f"{b['p_market']:.1%}  | edge brut +{b['edge']:.1%} / de-vig +{b['edge_fair']:.1%}"
              f"  | Kelly {b['kelly']:.2%}  | n={b['n']} ({b['n_surface']} gazon)")


if __name__ == "__main__":
    _cli()
