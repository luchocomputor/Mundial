"""Screener PÉPITES : classe les joueurs CDM par output offensif SOUS LE RADAR —
buts+passes/90 pondérés par la force de la ligue (ÉLEVÉS) vs valeur marchande
(FAIBLE). Surface les finisseurs/passeurs obscurs que le marché peut sous-coter →
candidats pour le mode Œil.

⚠️ Capte l'OUTPUT (buts/passes), pas le STYLE (dribble/mobilité/feu follet) — un
pur dribbleur type Fayzullaev passe sous le radar ; ça, c'est l'œil de Louis (ou,
plus tard, des stats FBref : dribbles/xA/progressif).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.goalscorer import _nlast, league_strength
from pipeline.teams import normalize as _norm

ROOT = Path(__file__).parent.parent
FORM = ROOT / "data" / "raw" / "transfermarkt" / "wc2026_player_form.csv"
SQUADS = ROOT / "data" / "raw" / "transfermarkt" / "wc2026_squads.csv"
FBREF = ROOT / "data" / "raw" / "fbref" / "wc2026_player_advanced.csv"

# Colonnes TM « étendues » (présentes seulement après re-scrape — cf. scrapers).
# minutes-pondérées comme start11. Absentes → ignorées proprement.
_TM_PCT = {"minutes_pct": "mpm", "gc_pct": "gcm", "gc_pct_nopen": "gcnm"}
# Colonnes avancées attendues (par 90, normalisées par le builder). Source-agnostique :
# Understat remplit npxg90/xa90/sh90/kp90 ; FBref ajoute sot90/sca90/prog90/box/drib.
_FBREF_COLS = ["npxg90", "xa90", "sh90", "sot90", "sca90", "kp90", "prog90", "box_touch90",
               "drib90", "act_g90"]


def _wmean(num: pd.Series, minutes: pd.Series) -> pd.Series:
    """Moyenne pondérée par les minutes (somme pondérée / minutes)."""
    return num / minutes.clip(lower=1)


def screen_players(min_minutes: int = 600) -> pd.DataFrame:
    """Tous les joueurs (assez de temps de jeu) classés par gem_score.

    Input enrichi par joueur, par couches :
      • OUTPUT club  : buts/90 + passes déc./90 (pondérés force-de-ligue), min/but
      • RÉGULARITÉ   : % titu pondéré minutes (et minutes% / dépendance penalty si TM étendu)
      • SÉLECTION    : age, caps, buts intl, buts/cap
      • MARCHÉ       : valeur marchande
      • QUALITÉ/STYLE: npxG/90, xA/90, tirs/90, SCA/90, progressif, touches surface (FBref, si dispo)
    age est le discriminant jeune-pépite vs vieux-décoté ; gem_score escompte l'output
    par l'âge et la régularité pour pousser les vraies pépites devant les vétérans décotés.
    """
    if not FORM.exists():
        return pd.DataFrame()
    pf = pd.read_csv(FORM)
    pf["wg"] = pf["goals"].fillna(0) * pf["competition"].map(league_strength)
    pf["wa"] = pf["assists"].fillna(0) * pf["competition"].map(league_strength)
    pf["s11m"] = pd.to_numeric(pf["start11_pct"], errors="coerce").fillna(0) * pf["minutes"]
    aggspec = dict(
        player=("player", "first"), team=("team", "first"), position=("position", "first"),
        wg=("wg", "sum"), wa=("wa", "sum"), goals=("goals", "sum"),
        assists=("assists", "sum"), minutes=("minutes", "sum"),
        games=("games", "sum"), s11m=("s11m", "sum"),
    )
    for col, w in _TM_PCT.items():  # TM étendu : pré-pondère puis somme
        if col in pf.columns:
            pf[w] = pd.to_numeric(pf[col], errors="coerce").fillna(0) * pf["minutes"]
            aggspec[w] = (w, "sum")
    agg = pf.groupby("player_id").agg(**aggspec).reset_index()
    agg = agg[agg["minutes"] >= min_minutes].copy()
    # force de la ligue de club DOMINANTE (comp où il joue le + de minutes) → sert à
    # pondérer le xG brut (un npxG en Saudi vaut moins qu'en PL), cohérent avec out90.
    dom = pf.loc[pf.groupby("player_id")["minutes"].idxmax()].set_index("player_id")["competition"]
    agg["ls_main"] = agg["player_id"].map(dom).map(league_strength).fillna(0.5)
    n90 = (agg["minutes"] / 90.0).clip(lower=0.1)
    agg["g90"] = agg["wg"] / n90        # finisseur (buts/90 pondéré ligue)
    agg["a90"] = agg["wa"] / n90        # créateur (passes déc./90 pondéré ligue)
    agg["out90"] = agg["g90"] + agg["a90"]
    agg["min_per_goal"] = agg["minutes"] / agg["goals"].where(agg["goals"] > 0)
    agg["start_share"] = _wmean(agg["s11m"], agg["minutes"]) / 100.0
    if "mpm" in agg:
        agg["minutes_share"] = _wmean(agg["mpm"], agg["minutes"]) / 100.0
    if "gcm" in agg and "gcnm" in agg:  # part des contributions venant des penaltys
        agg["penalty_reliance"] = (_wmean(agg["gcm"], agg["minutes"])
                                   - _wmean(agg["gcnm"], agg["minutes"])) / 100.0

    if SQUADS.exists():
        sq = pd.read_csv(SQUADS)
        sq["mv"] = pd.to_numeric(sq["market_value_eur"], errors="coerce").fillna(0)
        sq["caps"] = pd.to_numeric(sq["caps"], errors="coerce")
        sq["intl_goals"] = pd.to_numeric(sq["intl_goals"], errors="coerce")
        squad_cols = sq[["player_id", "age", "club", "caps", "intl_goals", "mv"]].drop_duplicates("player_id")
        agg = agg.merge(squad_cols, on="player_id", how="left")
        agg["intl_per_cap"] = agg["intl_goals"] / agg["caps"].clip(lower=1)
    else:
        for c in ("age", "club", "caps", "intl_goals", "intl_per_cap"):
            agg[c] = pd.NA
        agg["mv"] = 0
    agg["mv"] = agg["mv"].fillna(0)

    # ── Couche STYLE/QUALITÉ : FBref (xG/xA/tirs/progressif), si le CSV existe ──
    agg = _merge_fbref(agg)

    # ── gem_score : AGRÉGAT INTELLIGENT DE TOUS LES SIGNAUX ──────────────────────
    # Chaque signal → rang-percentile cross-sectionnel (0..1, joueurs de champ), puis
    # blend pondéré transparent → un "talent offensif" 0..1, escompté par âge+régularité.
    # xG league-pondéré (ls_main) pour comparer équitablement Saudi vs PL. Repli propre :
    # le poids des composantes manquantes (pas de xG) est redistribué (num/den).
    of = ~agg["position"].isin(["Goalkeeper"])
    ls = agg["ls_main"]

    def pctl(s: pd.Series) -> pd.Series:
        """Rang-percentile (0..1) parmi les joueurs de champ ; NaN préservés."""
        return pd.to_numeric(s, errors="coerce").where(of).rank(pct=True)

    p_out = pctl(agg["out90"])                            # ancre : output réel (toujours là)
    a90w = pctl(agg["a90"])                               # repli création quand pas de xA
    sig = {  # (percentile, poids) — somme des poids = 1.00
        "xout":   (pctl((agg["npxg90"] + agg["xa90"]) * ls), 0.34),  # output ATTENDU (cœur)
        "out":    (p_out,                                    0.20),  # output RÉEL (confirme)
        "threat": (pctl((agg["sh90"] + agg["kp90"]) * ls),  0.16),  # volume tirs + créa
        "create": (pctl(agg["xa90"] * ls).fillna(a90w),     0.12),  # création (xA, repli A)
        "drib":   (pctl(agg["drib90"] * ls),                0.10),  # feu follet (dribbles)
        "intl":   (pctl(agg["intl_per_cap"]).fillna(p_out), 0.08),  # buteur en sélection
    }
    # composantes manquantes (pas de xG) imputées au NIVEAU D'OUTPUT du joueur : le xG
    # ne fait que monter/descendre ceux qu'on mesure, il n'avantage pas les non-mesurés.
    agg["talent"] = sum(p.fillna(p_out) * w for p, w in sig.values()).where(of, 0.0)

    age = pd.to_numeric(agg["age"], errors="coerce")
    age_mult = (1 + (27 - age) * 0.03).clip(0.6, 1.4).fillna(1.0)
    reg_mult = (0.7 + 0.3 * agg["start_share"].fillna(0)).clip(0.7, 1.0)
    agg["gem_score"] = agg["talent"] * age_mult * reg_mult

    # signaux dérivés exposés
    agg["xout90"] = (agg["npxg90"] + agg["xa90"]) * ls          # output attendu pondéré
    agg["finishing90"] = agg["act_g90"] - agg["npxg90"]         # >0 chaud (régression), <0 dû
    agg["finishing"] = pd.cut(agg["finishing90"], [-99, -0.12, 0.12, 99],
                              labels=["froid", "neutre", "chaud"]).astype("object").fillna("")
    # breakout DURCI : jeune + pas cher + output réel ET talent (agrégat soutenable,
    # via xOut) tous deux top-quartile → exclut les profils « chauds » (out élevé mais
    # xOut faible) et les « potentiels » non encore producteurs (talent haut, out bas).
    out_thr = agg.loc[of, "out90"].quantile(0.75)
    tal_thr = agg.loc[of, "talent"].quantile(0.75)
    agg["breakout"] = ((age <= 23) & (agg["out90"] >= out_thr)
                       & (agg["talent"] >= tal_thr) & (agg["mv"] < 20e6))

    agg["tnorm"] = agg["team"].map(_norm)
    agg["last"] = agg["player"].map(_nlast)
    return agg.sort_values("gem_score", ascending=False).reset_index(drop=True)


def _merge_fbref(agg: pd.DataFrame) -> pd.DataFrame:
    """Greffe les stats avancées FBref (par player_id TM) si le CSV est là."""
    if FBREF.exists():
        fb = pd.read_csv(FBREF)
        keep = ["player_id"] + [c for c in _FBREF_COLS if c in fb.columns]
        agg = agg.merge(fb[keep].drop_duplicates("player_id"), on="player_id", how="left")
    for c in _FBREF_COLS:  # garantit la présence des colonnes (NaN si non couvert)
        if c not in agg.columns:
            agg[c] = pd.NA
    return agg


def gems_for_match(home: str, away: str, screen: pd.DataFrame | None = None,
                   top_n: int = 6, max_mv: float = 25e6, max_age: int | None = None) -> dict:
    """{équipe: DataFrame des pépites} pour un match (output élevé, VM < max_mv).

    max_age : plafond d'âge optionnel pour isoler les VRAIES pépites (jeune + cheap +
    productif) des vétérans simplement décotés par l'âge.
    """
    if screen is None:
        screen = screen_players()
    out: dict[str, pd.DataFrame] = {}
    for team in (home, away):
        tn = _norm(team)
        sub = screen[(screen["tnorm"] == tn) & (screen["mv"] < max_mv)
                     & (~screen["position"].isin(["Goalkeeper"]))]
        if max_age is not None:
            sub = sub[sub["age"].notna() & (sub["age"] <= max_age)]
        out[team] = sub.head(top_n)
    return out


if __name__ == "__main__":
    s = screen_players()
    cov = s["npxg90"].notna().mean()
    print(f"{len(s)} joueurs screenés · xG {cov*100:.0f}% · "
          f"top 15 pépites VM<15M (gem_score = agrégat intelligent) :")
    fin = {"chaud": "🔥", "froid": "🧊", "neutre": "", "": ""}
    for _, r in s[s["mv"] < 15e6].head(15).iterrows():
        age = f"{int(r.age)}a" if pd.notna(r.age) else " ?"
        xo = f"xOut {r.xout90:.2f}" if pd.notna(r.xout90) else "xOut  — "
        bo = " 🚀" if r.breakout else ""
        print(f"  {r.player[:18]:18s} {r.team[:10]:10s} {age:>4s} "
              f"out {r.out90:.2f} {xo} titu{r.start_share*100:3.0f}% "
              f"gem {r.gem_score:.2f}{fin.get(r.finishing,''):2s}  VM {r.mv/1e6:.0f}M€{bo}")
    print(f"\nBreakouts (jeune + output ET talent top-quartile + pas cher) : {int(s['breakout'].sum())}")
    for _, r in s[s["breakout"]].head(8).iterrows():
        print(f"  {r.player[:18]:18s} {r.team[:10]:10s} {int(r.age):>2d}a "
              f"talent {r.talent:.2f}  gem {r.gem_score:.2f}  VM {r.mv/1e6:.0f}M€")
