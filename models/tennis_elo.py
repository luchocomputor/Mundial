"""Elo tennis surface-ajusté — modèle de proba vainqueur (1v1, pas de nul).

Pendant tennis du `EloRating` foot, adapté au sport :
  - Pas de match nul → l'Elo prédit directement P(A bat B).
  - K-factor DÉCROISSANT avec l'expérience (façon FiveThirtyEight tennis) :
    K = 250 / (n + 5)^0.4. Débutant bouge vite, vétéran stable. Mieux calibré
    qu'un K constant sur des carrières longues.
  - Elo PAR SURFACE en plus de l'Elo global. Wimbledon = gazon (Grass). La proba
    se calcule sur un blend `α·surface + (1−α)·global` : sur gazon il y a peu de
    matchs → le blend ramène vers le global et évite le bruit (clé pour les qualifs,
    peuplées de bas-classés à faible historique surface).

CLÉ JOUEUR unifiée `surname + initiale` (player_key) : réconcilie les 3 formats de
noms rencontrés — tennis-data « Alcaraz C. », Betclic « Carlos Alcaraz », Sackmann
« Carlos Alcaraz ». Sans ça, l'Elo entraîné sur un format ne matche pas les cotes
d'un autre. Collisions surname+initiale rares (acceptées en v1).

Données (deux sources, schémas auto-détectés dans data/raw/tennis/<tour>/) :
  - tennis-data.co.uk : xlsx ATP/WTA avec cotes de clôture (Pinnacle) → Elo + backtest.
  - Jeff Sackmann (CSV) : tour principal + qualif/challenger (couvre les bas-classés).

Usage :
    python -m models.tennis_elo --tour atp          # build + sauvegarde l'artefact
    python -m models.tennis_elo --predict "Carlos Alcaraz" "Jannik Sinner" --surface Grass
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import unicodedata
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "tennis"
ARTIFACT_DIR = ROOT / "models" / "artifacts"

# K-factor décroissant (FiveThirtyEight tennis) : n = matchs déjà joués par le joueur.
K0, K_OFFSET, K_SHAPE = 250.0, 5.0, 0.4
INITIAL_RATING = 1500.0
# Poids de l'Elo surface dans le blend de prédiction (le reste = global).
SURFACE_BLEND = {"Grass": 0.5, "Clay": 0.6, "Hard": 0.6, "Carpet": 0.4}
DEFAULT_BLEND = 0.5

_SURFACES = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}
# Scores/commentaires qui ne sont PAS un match joué (RET/abandon = compte, lui).
_NO_CONTEST = re.compile(r"w/o|walkover|def\.|def$", re.IGNORECASE)
_INITIAL = re.compile(r"[a-z]\.?$")  # token initiale type « c. » (format Surname I.)


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", _strip_accents(name)).strip().lower()


def player_key(name: str) -> str:
    """Clé canonique `surname initiale`, robuste au format du nom :
      « Alcaraz C. »      → 'alcaraz c'   (Surname Initiale, tennis-data)
      « Carlos Alcaraz »  → 'alcaraz c'   (Prénom Nom, Betclic/Sackmann)
      « Auger-Aliassime F.» / « Felix Auger-Aliassime » → 'auger-aliassime f'
    """
    toks = norm_name(name).split()
    if not toks:
        return ""
    if len(toks) >= 2 and _INITIAL.fullmatch(toks[-1]):   # « Surname... I. »
        return f"{' '.join(toks[:-1])} {toks[-1][0]}"
    if len(toks) == 1:
        return toks[0]
    return f"{' '.join(toks[1:])} {toks[0][0]}"            # « First ... Last »


@dataclass
class _PlayerState:
    overall: float = INITIAL_RATING
    by_surface: dict[str, float] = field(default_factory=dict)
    n_overall: int = 0
    n_surface: dict[str, int] = field(default_factory=dict)
    last_date: int = 0  # YYYYMMDD du dernier match


def _k(n: int) -> float:
    return K0 / ((n + K_OFFSET) ** K_SHAPE)


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


# Calibration Platt : l'Elo brut est surconfiant (b≈0.42 en backtest) → il sur-détecte
# des « edges » fantômes. On tasse p vers 0.5. Fit walk-forward pendant le build.
CALIB_MIN_HISTORY = 15  # apprend la calibration sur des matchs au prior fiable (méthodo backtest)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def _fit_platt(pw: list[float]) -> tuple[float, float] | None:
    """(a, b) de p_cal = σ(a + b·logit(p)). Jeu symétrique : chaque match = 2 points
    (logit(pw),1) et (logit(1−pw),0), sinon tous les outcomes valent 1 (dégénéré)."""
    if len(pw) < 200:
        return None
    import numpy as np
    from scipy.optimize import minimize
    x = np.array([_logit(p) for p in pw] + [_logit(1 - p) for p in pw])
    y = np.concatenate([np.ones(len(pw)), np.zeros(len(pw))])

    def nll(ab):
        z = ab[0] + ab[1] * x
        return float(np.mean(np.logaddexp(0, z) - y * z))

    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return (float(res.x[0]), float(res.x[1])) if res.success else None


class TennisElo:
    def __init__(self, tour: str = "atp"):
        self.tour = tour
        self.players: dict[str, _PlayerState] = {}
        self.names: dict[str, str] = {}      # player_key → nom d'affichage (dernier vu)
        self.n_matches = 0
        self.date_max = 0
        self.calibration: tuple[float, float] | None = None  # Platt (a, b)

    # ── build ────────────────────────────────────────────────────────────────
    def _state(self, key: str) -> _PlayerState:
        st = self.players.get(key)
        if st is None:
            st = _PlayerState()
            self.players[key] = st
        return st

    def _update(self, w_name: str, l_name: str, surface: str, date: int) -> None:
        wk, lk = player_key(w_name), player_key(l_name)
        if not wk or not lk or wk == lk:
            return
        self.names[wk], self.names[lk] = w_name, l_name
        w, l = self._state(wk), self._state(lk)

        ew = _expected(w.overall, l.overall)
        w.overall += _k(w.n_overall) * (1.0 - ew)
        l.overall += _k(l.n_overall) * (ew - 1.0)
        w.n_overall += 1
        l.n_overall += 1

        if surface:
            wr = w.by_surface.get(surface, INITIAL_RATING)
            lr = l.by_surface.get(surface, INITIAL_RATING)
            es = _expected(wr, lr)
            w.by_surface[surface] = wr + _k(w.n_surface.get(surface, 0)) * (1.0 - es)
            l.by_surface[surface] = lr + _k(l.n_surface.get(surface, 0)) * (es - 1.0)
            w.n_surface[surface] = w.n_surface.get(surface, 0) + 1
            l.n_surface[surface] = l.n_surface.get(surface, 0) + 1

        w.last_date = l.last_date = max(w.last_date, date)
        self.n_matches += 1
        self.date_max = max(self.date_max, date)

    def fit(self, matches: pd.DataFrame) -> "TennisElo":
        """matches : colonnes canoniques winner_name, loser_name, surface,
        tourney_date (YYYYMMDD), score (option., pour détecter walkover)."""
        df = matches.copy()
        df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce").fillna(0).astype(int)
        df["match_num"] = pd.to_numeric(df.get("match_num", 0), errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["tourney_date", "match_num"], kind="stable")

        train_pw: list[float] = []  # probas pré-update du vainqueur (walk-forward) → calibration
        for r in df.itertuples(index=False):
            score = getattr(r, "score", "") or ""
            if isinstance(score, str) and _NO_CONTEST.search(score):
                continue  # walkover / forfait
            surface = _SURFACES.get(str(getattr(r, "surface", "")).strip().lower(), "")
            wk, lk = player_key(r.winner_name), player_key(r.loser_name)
            if wk and lk and wk != lk:
                w, l = self.players.get(wk), self.players.get(lk)
                if w and l and min(w.n_overall, l.n_overall) >= CALIB_MIN_HISTORY:
                    train_pw.append(_expected(self._eff_rating(wk, surface),
                                              self._eff_rating(lk, surface)))
            self._update(r.winner_name, r.loser_name, surface, int(r.tourney_date))

        self.calibration = _fit_platt(train_pw)  # walk-forward → pas de fuite
        return self

    # ── predict ────────────────────────────────────────────────────────────────
    def _eff_rating(self, key: str, surface: str) -> float:
        st = self.players.get(key)
        if st is None:
            return INITIAL_RATING
        if not surface:
            return st.overall
        sr = st.by_surface.get(surface, st.overall)  # jamais joué la surface → global
        a = SURFACE_BLEND.get(surface, DEFAULT_BLEND)
        return a * sr + (1 - a) * st.overall

    def _calibrate(self, p: float) -> float:
        if not self.calibration:
            return p
        a, b = self.calibration
        return 1.0 / (1.0 + math.exp(-(a + b * _logit(p))))

    def predict(self, player_a: str, player_b: str, surface: str = "Grass",
                calibrated: bool = True) -> dict:
        """P(A bat B) sur `surface` + diagnostic de confiance (nb de matchs).
        calibrated=True applique le Platt (recommandé : l'Elo brut est surconfiant)."""
        surface = _SURFACES.get(surface.strip().lower(), surface.title())
        ka, kb = player_key(player_a), player_key(player_b)
        ra, rb = self._eff_rating(ka, surface), self._eff_rating(kb, surface)
        p_a = self._calibrate(_expected(ra, rb)) if calibrated else _expected(ra, rb)
        sa, sb = self.players.get(ka), self.players.get(kb)
        return {
            "p_a": p_a, "p_b": 1 - p_a,
            "rating_a": ra, "rating_b": rb,
            "n_a": sa.n_overall if sa else 0, "n_b": sb.n_overall if sb else 0,
            "n_a_surface": (sa.n_surface.get(surface, 0) if sa else 0),
            "n_b_surface": (sb.n_surface.get(surface, 0) if sb else 0),
            "known_a": sa is not None, "known_b": sb is not None,
            "surface": surface,
        }

    def leaderboard(self, surface: str | None = None, top: int = 25, min_matches: int = 20) -> pd.DataFrame:
        rows = [{"player": self.names.get(k, k), "rating": round(self._eff_rating(k, surface) if surface else st.overall, 1),
                 "n": st.n_overall, "last": st.last_date}
                for k, st in self.players.items() if st.n_overall >= min_matches]
        return (pd.DataFrame(rows).sort_values("rating", ascending=False)
                .head(top).reset_index(drop=True))

    # ── io ────────────────────────────────────────────────────────────────
    def artifact_path(self) -> Path:
        return ARTIFACT_DIR / f"tennis_elo_{self.tour}.pkl"

    def save(self, path: Path | None = None) -> None:
        path = path or self.artifact_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, tour: str = "atp") -> "TennisElo":
        with open(ARTIFACT_DIR / f"tennis_elo_{tour}.pkl", "rb") as f:
            return pickle.load(f)


# ── chargement des données (xlsx tennis-data + csv Sackmann) ────────────────────
_TD_RENAME = {"Winner": "winner_name", "Loser": "loser_name", "Surface": "surface",
              "Comment": "score"}


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """Ramène un fichier (tennis-data XLSX ou Sackmann CSV) aux colonnes canoniques."""
    cols = set(df.columns)
    if {"winner_name", "loser_name", "tourney_date"}.issubset(cols):   # Sackmann
        keep = ["tourney_date", "match_num", "surface", "winner_name", "loser_name", "score"]
        return df[[c for c in keep if c in cols]].copy()
    if {"Winner", "Loser", "Date"}.issubset(cols):                     # tennis-data.co.uk
        out = df.rename(columns=_TD_RENAME).copy()
        out["tourney_date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y%m%d")
        out["match_num"] = range(len(out))  # ordre du fichier = ordre chronologique du tournoi
        keep = ["tourney_date", "match_num", "surface", "winner_name", "loser_name", "score"]
        return out[[c for c in keep if c in out.columns]]
    return None


def load_matches(tour: str = "atp", data_dir: Path | None = None) -> pd.DataFrame:
    """Concatène tous les fichiers (xlsx/csv) sous data/raw/tennis/<tour>/."""
    base = (data_dir or DATA_DIR) / tour
    files = sorted(glob(str(base / "*.xlsx")) + glob(str(base / "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier sous {base}. Lance pipeline/fetch_tennis_data.py.")
    frames = []
    for f in files:
        try:
            df = pd.read_excel(f) if f.endswith(".xlsx") else pd.read_csv(f, low_memory=False)
            n = _normalize(df)
            if n is not None and len(n):
                frames.append(n)
        except Exception as e:
            print(f"  ⚠ {Path(f).name} ignoré : {e}")
    if not frames:
        raise ValueError(f"Fichiers trouvés sous {base} mais aucun exploitable.")
    return pd.concat(frames, ignore_index=True)


def build(tour: str = "atp", data_dir: Path | None = None) -> TennisElo:
    return TennisElo(tour=tour).fit(load_matches(tour, data_dir))


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Elo tennis surface-ajusté")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--predict", nargs=2, metavar=("A", "B"))
    ap.add_argument("--surface", default="Grass")
    args = ap.parse_args()

    if args.predict:
        m = TennisElo.load(args.tour)
        r = m.predict(*args.predict, surface=args.surface)
        for who, p, rt, n, ns, k in [
            (args.predict[0], r["p_a"], r["rating_a"], r["n_a"], r["n_a_surface"], r["known_a"]),
            (args.predict[1], r["p_b"], r["rating_b"], r["n_b"], r["n_b_surface"], r["known_b"])]:
            print(f"  P({who}) = {p:.1%}  (rating {rt:.0f}, {n} matchs / {ns} sur {r['surface']})"
                  f"{'' if k else '  ⚠ inconnu'}")
        return

    m = build(args.tour)
    m.save()
    print(f"✅ Elo {args.tour.upper()} : {len(m.players)} joueurs, {m.n_matches} matchs "
          f"(dernier {m.date_max}) → {m.artifact_path().name}")
    print("\nTop gazon :")
    print(m.leaderboard(surface="Grass", top=15).to_string(index=False))


if __name__ == "__main__":
    # Ré-import sous le vrai nom de module : sinon les classes (TennisElo,
    # _PlayerState) seraient picklées comme `__main__.*` et illisibles à la
    # relecture depuis un autre process (AttributeError sur _PlayerState).
    from models.tennis_elo import _cli as _real_cli
    _real_cli()
