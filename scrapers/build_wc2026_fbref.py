"""Couche STYLE/QUALITÉ FBref → data/raw/fbref/wc2026_player_advanced.csv.

FBref bloque le scraping (403 Cloudflare). On passe donc par EXPORT MANUEL : sur
fbref.com, ouvrir les tables joueurs « Big 5 European Leagues » (ou une ligue), cliquer
« Get table as CSV », coller le contenu dans un fichier sous data/raw/fbref/raw/.
Tables utiles (déposer celles qu'on veut, l'absence dégrade proprement) :

    Standard Stats   → npxg90, xa90      (colonnes 90s, npxG, xAG)
    Shooting         → sh90, sot90       (90s, Sh, SoT)
    Goal/Shot Creat. → sca90             (90s, SCA)
    Possession       → prog90, box_touch90, drib90  (PrgC, Att Pen, Take-Ons Att)
    Passing          → prog90 (renfort)  (PrgP)

Ce script auto-détecte le type de chaque CSV par ses colonnes, agrège les totaux par
joueur, calcule les per-90 (total / 90s), puis matche le joueur sur le player_id
Transfermarkt (nom normalisé + désambiguïsation par club). Couverture = top
championnats only (assumé) ; les joueurs hors Big-5 restent sans xG → NaN côté gems.

Usage : python -m scrapers.build_wc2026_fbref
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.goalscorer import _nfull, _nlast  # noqa: E402

SQUADS = ROOT / "data" / "raw" / "transfermarkt" / "wc2026_squads.csv"
RAWDIR = ROOT / "data" / "raw" / "fbref" / "raw"
OUT = ROOT / "data" / "raw" / "fbref" / "wc2026_player_advanced.csv"
MIN_S90 = 4.0  # minimum de 90s (~360 min) dans la ligue pour des per-90 stables

# Mots de club ignorés pour la désambiguïsation (tokens trop communs).
_CLUB_STOP = {"fc", "cf", "ac", "sc", "club", "de", "the", "city", "united", "real",
              "athletic", "atletico", "calcio", "ssd", "ssc", "afc", "cd", "if", "sk"}

# Signatures de table (colonne discriminante) + extraction des totaux. Deux familles :
# FBref (export « Get table as CSV ») ET Understat (snippet console, cf. README ci-dessous).
# Chaque entrée : (set de colonnes requises) → {cible_total: nom_colonne_source}.
TABLES = [
    # — Understat (1 CSV/ligue, déjà par joueur×saison) —
    ("understat", {"xG", "xA", "time"},
     {"npxg_tot": "npxG", "xag_tot": "xA", "sh_tot": "shots", "kp_tot": "key_passes",
      "actg_tot": "goals"}),
    # — ASA / American Soccer Analysis (MLS, via scrapers.fetch_asa_mls) —
    ("asa", {"xgoals", "xassists", "minutes_played"},
     {"npxg_tot": "xgoals", "xag_tot": "xassists", "sh_tot": "shots",
      "sot_tot": "shots_on_target", "kp_tot": "key_passes", "actg_tot": "goals"}),
    # — Sofascore (mondial : Golfe/Liga MX…, via sofascore_console_snippet.js) —
    ("sofascore", {"expectedGoals", "expectedAssists", "minutesPlayed"},
     {"npxg_tot": "expectedGoals", "xag_tot": "expectedAssists", "sh_tot": "totalShots",
      "sot_tot": "shotsOnTarget", "kp_tot": "keyPasses", "drib_tot": "successfulDribbles",
      "actg_tot": "goals"}),
    # — FBref (1 CSV/table, via snippet console → colonnes = attributs data-stat FBref) —
    ("fb_standard", {"npxg", "xg_assist"},
     {"npxg_tot": "npxg", "xag_tot": "xg_assist", "actg_tot": "goals"}),
    ("fb_shooting", {"shots", "shots_on_target"},
     {"sh_tot": "shots", "sot_tot": "shots_on_target"}),
    ("fb_gca", {"sca"}, {"sca_tot": "sca"}),
    ("fb_possession", {"touches_att_pen_area", "take_ons"},
     {"box_tot": "touches_att_pen_area", "drib_tot": "take_ons",
      "prgc_tot": "progressive_carries"}),
    ("fb_passing", {"progressive_passes"},
     {"prgp_tot": "progressive_passes", "kp_tot": "assisted_shots"}),
]

# Per-90 finales (consommées par models.gems._FBREF_COLS).
PER90 = {
    "npxg90": "npxg_tot", "xa90": "xag_tot", "sh90": "sh_tot", "sot90": "sot_tot",
    "sca90": "sca_tot", "kp90": "kp_tot", "box_touch90": "box_tot", "drib90": "drib_tot",
    "act_g90": "actg_tot",  # buts réels/90 (ligue xG) → flag finition chaud/froid vs npxg90
}  # prog90 traité à part (PrgC + PrgP)


def _norm_club(s: str) -> set[str]:
    return {t for t in _nfull(s).split() if t not in _CLUB_STOP and len(t) > 1}


def _read_fbref_csv(path: Path) -> tuple[str, pd.DataFrame] | None:
    """Lit un CSV FBref, identifie son type, renvoie (type, df de totaux par joueur)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"player_name": "Player", "team_title": "Squad",  # Understat/ASA
                            "player": "Player", "team": "Squad"})            # FBref data-stat
    if "Player" not in df.columns:
        return None
    df = df[df["Player"].astype(str).str.strip().ne("Player")].copy()  # ré-en-têtes FBref
    cols = set(df.columns)
    for name, sig, mapping in TABLES:
        if sig.issubset(cols):
            keep = {"Player": "player", "Squad": "squad"}
            keep.update({src: tgt for tgt, src in mapping.items()})  # col source → cible
            sub = df[[c for c in keep if c in df.columns]].rename(columns=keep)
            if "90s" in df.columns:
                sub["s90"] = pd.to_numeric(df["90s"], errors="coerce")
            elif "time" in df.columns:        # Understat : minutes jouées
                sub["s90"] = pd.to_numeric(df["time"], errors="coerce") / 90.0
            elif "minutes_played" in df.columns:  # ASA
                sub["s90"] = pd.to_numeric(df["minutes_played"], errors="coerce") / 90.0
            elif "minutesPlayed" in df.columns:   # Sofascore
                sub["s90"] = pd.to_numeric(df["minutesPlayed"], errors="coerce") / 90.0
            elif "minutes_90s" in df.columns:     # FBref data-stat (« 90s »)
                sub["s90"] = pd.to_numeric(df["minutes_90s"], errors="coerce")
            elif "minutes" in df.columns:         # FBref repli : minutes brutes
                sub["s90"] = pd.to_numeric(df["minutes"], errors="coerce") / 90.0
            elif "Min" in df.columns:
                sub["s90"] = pd.to_numeric(df["Min"], errors="coerce") / 90.0
            for c in sub.columns:
                if c not in ("player", "squad"):
                    sub[c] = pd.to_numeric(sub[c], errors="coerce")
            return name, sub
    return None


def _build_tm_index() -> tuple[dict, dict]:
    """{nom_norm: [(player_id, club_tokens)]} et {(last, club_tok frozenset): id}."""
    sq = pd.read_csv(SQUADS)
    full: dict[str, list] = {}
    for _, r in sq.iterrows():
        if pd.isna(r.player_id):
            continue
        full.setdefault(_nfull(r.player), []).append(
            (int(r.player_id), _norm_club(str(r.get("club", "")))))
    last: dict[str, list] = {}
    for nf, lst in full.items():
        ln = _nlast(nf)
        last.setdefault(ln, []).extend(lst)
    return full, last


def _match_id(player: str, squad: str, full: dict, last: dict) -> int | None:
    """player_id TM, HAUTE PRÉCISION : nom complet exact unique, OU nom de famille
    CORROBORÉ par le club (le club source recoupe le club TM). Sinon None — on préfère
    rater un joueur que fabriquer un faux positif (un homonyme hors-CDM en MLS, p.ex.)."""
    nf = _nfull(player)
    cands = full.get(nf)
    if cands and len(cands) == 1:
        return cands[0][0]                       # nom complet exact unique = sûr
    pool = cands or last.get(_nlast(nf))
    if not pool:
        return None
    sq_tok = _norm_club(squad)
    if not sq_tok:                               # pas de club → pas de corroboration
        return None
    best, best_ov = None, 0
    for pid, club_tok in pool:
        ov = len(sq_tok & club_tok)
        if ov > best_ov:
            best, best_ov = pid, ov
    return best if best_ov > 0 else None         # exige un recoupement de club


def run() -> pd.DataFrame:
    if not RAWDIR.exists() or not any(RAWDIR.glob("*.csv")):
        print(f"Aucun CSV FBref dans {RAWDIR}/ — exporter les tables (voir docstring).")
        return pd.DataFrame()
    full, last = _build_tm_index()

    # Agrège les totaux de toutes les tables par (player, squad).
    acc: dict[tuple, dict] = {}
    seen_types = []
    for f in sorted(RAWDIR.glob("*.csv")):
        res = _read_fbref_csv(f)
        if not res:
            print(f"  ? {f.name} : type FBref non reconnu, ignoré.")
            continue
        name, sub = res
        seen_types.append(name)
        for _, r in sub.iterrows():
            key = (_nfull(r.player), _nfull(str(r.get("squad", ""))))
            rec = acc.setdefault(key, {"player": r.player, "squad": r.get("squad", "")})
            for c in sub.columns:
                if c not in ("player", "squad") and pd.notna(r[c]):
                    rec[c] = r[c]  # garde la dernière valeur vue (tables cohérentes)
        print(f"  ✓ {f.name} → table « {name} » ({len(sub)} joueurs)")

    if not acc:
        print("Aucune table FBref exploitable.")
        return pd.DataFrame()

    rows = []
    for rec in acc.values():
        s90 = rec.get("s90")
        if not s90 or s90 < MIN_S90:  # trop peu de minutes → per-90 = bruit, on n'émet pas
            continue
        out = {"player": rec["player"], "squad": rec.get("squad", "")}
        for col, tot in PER90.items():
            if tot in rec and pd.notna(rec[tot]):
                out[col] = round(rec[tot] / s90, 3)
        prg = sum(rec[t] for t in ("prgc_tot", "prgp_tot") if t in rec and pd.notna(rec[t]))
        if any(t in rec for t in ("prgc_tot", "prgp_tot")):
            out["prog90"] = round(prg / s90, 3)
        out["player_id"] = _match_id(rec["player"], str(rec.get("squad", "")), full, last)
        rows.append(out)

    df = pd.DataFrame(rows)
    matched = df["player_id"].notna()
    df = df[matched].copy()
    df["player_id"] = df["player_id"].astype(int)
    df = df.drop_duplicates("player_id")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["player_id", "player", "squad", "npxg90", "xa90", "sh90", "sot90",
            "sca90", "kp90", "prog90", "box_touch90", "drib90", "act_g90"]
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df.to_csv(OUT, index=False)
    print(f"\n✅ tables vues : {seen_types}")
    print(f"   {int(matched.sum())}/{len(rows)} joueurs FBref matchés sur un player_id TM → {OUT}")
    if "npxg90" in df.columns:
        top = df.dropna(subset=["npxg90"]).sort_values("npxg90", ascending=False).head(8)
        print("\nTop npxG/90 (matchés) :")
        for _, r in top.iterrows():
            print(f"  {str(r.player)[:22]:22s} {str(r.squad)[:16]:16s} npxG/90 {r.npxg90:.2f}")
    return df


if __name__ == "__main__":
    run()
