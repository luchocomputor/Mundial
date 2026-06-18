"""Dashboard CDM 2026 · streamlit run dashboard.py"""

import json
import re
import sys
import time
import requests as _requests
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
TODAY = date.today().isoformat()
BANKROLL_INIT = 200.0
BETS_LOG      = ROOT / "data" / "bets_log.csv"
ODDS_PATH     = ROOT / "data" / "raw" / "betclic_odds_flat.json"
SCORES_CACHE  = ROOT / "data" / "scores_cache.json"

st.set_page_config(page_title="CDM 2026", page_icon="⚽", layout="wide")

st.markdown("""<style>
:root {
  --bg:     #0A0B0D;   /* graphite, neutre — zéro teinte bleue */
  --s1:     #0E0F12;   /* sidebar */
  --s2:     #131517;   /* carte */
  --s3:     #1A1D21;   /* surface élevée */
  --bd:     rgba(255,255,255,.07);
  --bd2:    rgba(255,255,255,.15);
  --t:      #ECEDEE;
  --m1:     rgba(236,237,238,.60);
  --m2:     rgba(236,237,238,.38);
  --m3:     rgba(236,237,238,.20);
  --accent: #34D399;   /* emerald — marque + signal value/positif */
  --accent2:#6E8BB7;   /* steel discret — secondaire (extérieur) */
  --indigo: #34D399;   /* alias historique → emerald */
  --green:  #34D399;
  --red:    #F87171;
  --amber:  #FBBF24;
  --orange: #FB923C;
  --r: 12px;
}

/* ── Hide Streamlit chrome ── */
header[data-testid="stHeader"]  { display:none !important; }
#MainMenu, footer, .stDeployButton { display:none !important; }

.stApp { background: var(--bg) !important; }
.block-container { padding-top: 1.6rem !important; max-width: 980px !important; }
/* Targeted font override — don't touch icon-rendering spans/divs */
html, body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif !important;
}
.block-container p, .block-container label, .block-container h1, .block-container h2,
.stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"] p,
.stButton > button { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--s1) !important;
  border-right: 1px solid var(--bd) !important;
}
[data-testid="stSidebarContent"] { padding-top: 1.6rem !important; }
.stButton > button {
  background: rgba(255,255,255,.06) !important;
  border: 1px solid var(--bd2) !important;
  color: var(--t) !important;
  border-radius: 9px !important;
  font-weight: 600 !important;
  font-size: .84rem !important;
  transition: background .15s, border-color .15s !important;
}
.stButton > button:hover {
  background: rgba(255,255,255,.1) !important;
  border-color: rgba(52,211,153,.4) !important;
}

/* ── KPI grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 10px;
  margin-bottom: 30px;
}
.kpi {
  background: linear-gradient(180deg, rgba(255,255,255,.022), rgba(255,255,255,0) 42%), var(--s2);
  border: 1px solid var(--bd);
  border-radius: var(--r);
  padding: 18px 20px 16px;
  position: relative;
  overflow: hidden;
}
.kpi::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
}
.kpi-g::after { background: linear-gradient(90deg, var(--green)  70%, transparent); }
.kpi-r::after { background: linear-gradient(90deg, var(--red)    70%, transparent); }
.kpi-b::after { background: linear-gradient(90deg, var(--accent2) 70%, transparent); }
.kpi-n::after { background: linear-gradient(90deg, var(--m3)     70%, transparent); }
.kpi-lbl { font-size:.58rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:rgba(236,237,238,.5); margin-bottom:8px; }
.kpi-val { font-size:1.72rem; font-weight:800; color:var(--t); line-height:1.05; letter-spacing:-.01em; }
.kpi-sub { font-size:.73rem; font-weight:500; margin-top:6px; }
.g { color:var(--green); } .r { color:var(--red); } .n { color:var(--m2); }

/* ── CLV panel ── */
.clv-wait {
  background: linear-gradient(180deg, rgba(255,255,255,.018), rgba(255,255,255,0) 30%), var(--s2);
  border: 1px solid var(--bd); border-radius: var(--r);
  padding: 15px 20px; font-size: .82rem; color: var(--m2); margin-bottom: 30px;
}
.clv-wait b { color: var(--t); font-weight: 700; }
.clv-wait-i { margin-right: 8px; opacity:.8; }
.clv-chips { display:flex; flex-wrap:wrap; gap:8px; margin: -20px 0 30px; }
.clv-chip {
  background: rgba(255,255,255,.03); border:1px solid var(--bd);
  border-radius: 8px; padding: 5px 11px; font-size:.72rem; color: var(--m2);
}
.clv-chip b { color: var(--t); font-weight:700; margin-right:5px; }
.clv-chip-n { color: var(--m3); margin-left:5px; }

/* ── Section dividers ── */
.sec {
  display: flex; align-items: center; gap: 14px;
  margin: 28px 0 12px;
}
.sec::before, .sec::after { content:''; flex:1; height:1px; background:var(--bd); }
.sec::before { max-width:12px; }
.sec span {
  font-size:.58rem; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; white-space:nowrap;
}
/* Past: muted */
.sec-past  span { color:var(--m3); }
/* Today: bright */
.sec-today span { color:var(--t); letter-spacing:.12em; }
.sec-today::before, .sec-today::after { background: rgba(52,211,153,.25); }
/* Future: mid */
.sec-future span { color:var(--m2); }

/* ── Match card base ── */
.mc {
  background: linear-gradient(180deg, rgba(255,255,255,.018), rgba(255,255,255,0) 30%), var(--s2);
  border: 1px solid var(--bd);
  border-radius: var(--r);
  margin-bottom: 8px;
  overflow: hidden;
  transition: border-color .15s, box-shadow .2s;
}
.mc:hover { border-color: var(--bd2); box-shadow: 0 6px 32px rgba(0,0,0,.4); }

/* ── Status color coding ── */
/* Finished → left accent barely visible, card slightly muted */
.mc.mc-ft {
  border-left: 3px solid rgba(255,255,255,.09);
}
.mc.mc-ft .mc-strip {
  background: rgba(255,255,255,.01);
}

/* Live → orange glow */
.mc.mc-live {
  border-left: 3px solid var(--orange);
  border-color: rgba(251,146,60,.28);
  box-shadow: -3px 0 12px rgba(251,146,60,.08);
}
.mc.mc-live .mc-strip {
  background: rgba(251,146,60,.06);
  border-bottom-color: rgba(251,146,60,.18);
}

/* Upcoming → indigo accent */
.mc.mc-ns {
  border-left: 3px solid rgba(52,211,153,.45);
  border-color: rgba(52,211,153,.18);
}
.mc.mc-ns .mc-strip {
  background: rgba(52,211,153,.04);
  border-bottom-color: rgba(52,211,153,.14);
}

/* Card strip */
.mc-strip {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 16px;
  background: rgba(255,255,255,.016);
  border-bottom: 1px solid var(--bd);
}
.mc-comp { font-size:.57rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--m3); }
.mc-meta { display:flex; align-items:center; gap:6px; }
.mc-dt   { font-size:.61rem; color:var(--m2); font-weight:500; }

/* Status badges */
.sb {
  display:inline-block; border-radius:20px;
  font-size:.57rem; font-weight:700; letter-spacing:.07em;
  text-transform:uppercase; padding:2px 8px;
}
.sb-ft   { background:rgba(255,255,255,.06); color:var(--m2); }
.sb-live { background:rgba(251,146,60,.16); color:#FCA96E; border:1px solid rgba(251,146,60,.25); }
.sb-ns   { background:rgba(52,211,153,.14); color:#6EE7B7; }
.ldot {
  display:inline-block; width:5px; height:5px; border-radius:50%;
  background:#FCA96E; margin-right:4px; vertical-align:middle;
  animation:pulse 1.2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.2;transform:scale(.5)} }

/* Card body — horizontal row */
.mc-body {
  display: flex; align-items: center;
  padding: 20px 18px 14px; gap: 8px;
}
.mc-team { flex:1; display:flex; align-items:center; gap:10px; min-width:0; overflow:hidden; }
.mc-team.away { flex-direction:row-reverse; }
.mc-flag { font-size:1.75rem; flex-shrink:0; line-height:1; }
.mc-name { font-size:.96rem; font-weight:700; color:var(--t); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* Score center — betting-site style tiles */
.mc-ctr  { min-width:160px; text-align:center; flex-shrink:0; }
.mc-sc   { display:flex; align-items:center; justify-content:center; gap:6px; }
.mc-sc-n {
  min-width:54px; height:60px;
  display:flex; align-items:center; justify-content:center;
  background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.11);
  border-radius:10px;
  font-size:2.5rem; font-weight:900; color:#fff;
  font-variant-numeric:tabular-nums; letter-spacing:-.025em; line-height:1;
  transition: opacity .15s;
}
.mc-sc-n.loser  { opacity:.35; background:rgba(255,255,255,.03); border-color:rgba(255,255,255,.06); }
.mc-sc-sep { font-size:.88rem; color:rgba(236,237,238,.18); font-weight:700; padding:0 1px; }
.mc-sc-lbl { font-size:.57rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-top:6px; }
.mc-ft   .mc-sc-lbl { color:var(--m3); }
.mc-live .mc-sc-lbl { color:#FCA96E; }
/* Upcoming: VS + date */
.mc-vs    { font-size:1.05rem; font-weight:800; color:rgba(236,237,238,.18); letter-spacing:.1em; }
.mc-vs-dt { font-size:.63rem; color:var(--m3); margin-top:5px; letter-spacing:.04em; }

/* Proba bar */
.mc-pb { display:flex; height:3px; margin:8px 18px 0; border-radius:2px; overflow:hidden; gap:1px; }
.mc-pb-h { background:var(--accent); }
.mc-pb-d { background:rgba(255,255,255,.1); }
.mc-pb-a { background:var(--accent2); }
.mc-pct  {
  display:flex; justify-content:space-between;
  padding:4px 18px 8px;
  font-size:.65rem; font-weight:600; color:var(--m2);
}
.pdot { display:inline-block; width:6px; height:6px; border-radius:50%; vertical-align:middle; }
.pdot-h { background:var(--accent);  margin-right:5px; }
.pdot-a { background:var(--accent2); margin-left:5px; }

/* Bet pills */
.mc-bets { display:flex; flex-wrap:wrap; gap:5px; padding:2px 16px 13px; }
.bt {
  display:inline-flex; align-items:center; gap:4px; padding:4px 10px;
  border-radius:7px; font-size:.73rem; font-weight:600;
  border:1px solid var(--bd); background:rgba(255,255,255,.03); color:var(--m1);
}
.bt-w   { background:rgba(52,211,153,.07);  border-color:rgba(52,211,153,.22);  color:var(--green); }
.bt-l   { background:rgba(248,113,113,.07); border-color:rgba(248,113,113,.22); color:var(--red); }
.bt-p   { color:var(--m1); }
.bt-adv { background:rgba(52,211,153,.07); border-color:rgba(52,211,153,.22); color:#6EE7B7; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(255,255,255,.018) !important;
  border: 1px solid var(--bd) !important;
  border-radius: 10px !important;
  margin-top: -4px !important;
  margin-bottom: 10px !important;
}
[data-testid="stExpanderDetails"] { padding:4px 14px 14px !important; }

/* ── Bet slips ── */
.slip {
  display:flex; align-items:center; gap:8px; padding:8px 11px; margin-bottom:4px;
  border-radius:8px; background:rgba(255,255,255,.024);
  border:1px solid var(--bd); border-left:3px solid rgba(255,255,255,.08);
}
.slip-w   { border-left-color:var(--green)  !important; }
.slip-l   { border-left-color:var(--red)    !important; }
.slip-p   { border-left-color:rgba(255,255,255,.1) !important; }
.slip-adv { border-left-color:var(--indigo) !important; }
.slip-icon { font-size:.9rem; width:18px; flex-shrink:0; }
.slip-mkt  { flex:1; font-size:.82rem; font-weight:600; color:var(--t); min-width:0; }
.slip-meta { font-size:.75rem; color:var(--m1); white-space:nowrap; }
.slip-cote { font-size:.82rem; font-weight:700; color:var(--t); white-space:nowrap; min-width:38px; text-align:right; }
.slip-stake{ font-size:.75rem; color:var(--m2); white-space:nowrap; min-width:28px; text-align:right; }
.slip-pnl-w{ font-size:.8rem; font-weight:700; color:var(--green); white-space:nowrap; min-width:52px; text-align:right; }
.slip-pnl-l{ font-size:.8rem; font-weight:700; color:var(--red);   white-space:nowrap; min-width:52px; text-align:right; }
.slip-pnl-p{ font-size:.8rem; color:var(--m2); white-space:nowrap; min-width:52px; text-align:right; }

/* ── Edge badges ── */
.eb { display:inline-block; padding:1px 7px; border-radius:20px; font-size:.67rem; font-weight:700; white-space:nowrap; }
.eb-r  { background:rgba(248,113,113,.1);  color:var(--red);    border:1px solid rgba(248,113,113,.22); }
.eb-o  { background:rgba(251,146,60,.1);   color:var(--orange); border:1px solid rgba(251,146,60,.22); }
.eb-y  { background:rgba(251,191,36,.1);   color:var(--amber);  border:1px solid rgba(251,191,36,.22); }
.eb-w  { background:rgba(255,255,255,.06); color:var(--m1);     border:1px solid rgba(255,255,255,.11); }
.eb-g  { background:rgba(52,211,153,.08);  color:var(--green);  border:1px solid rgba(52,211,153,.2); }
.eb-gr { background:rgba(255,255,255,.04); color:var(--m2);     border:1px solid var(--bd); }

/* ── Odds freshness banner ── */
.ob {
  display:flex; align-items:center; gap:10px;
  padding:9px 15px; border-radius:10px; margin-bottom:18px;
  font-size:.78rem; font-weight:600; border:1px solid;
}
.ob-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.ob-txt { color:var(--t); }
.ob-hint { margin-left:auto; font-size:.71rem; font-weight:600; white-space:nowrap; }
.ob-fresh { background:rgba(52,211,153,.06);  border-color:rgba(52,211,153,.22); }
.ob-fresh .ob-dot  { background:var(--green); box-shadow:0 0 8px rgba(52,211,153,.8); }
.ob-fresh .ob-hint { color:var(--green); }
.ob-ok    { background:rgba(251,191,36,.06);  border-color:rgba(251,191,36,.22); }
.ob-ok    .ob-dot  { background:var(--amber); }
.ob-ok    .ob-hint { color:var(--amber); }
.ob-stale { background:rgba(248,113,113,.07); border-color:rgba(248,113,113,.30); }
.ob-stale .ob-dot  { background:var(--red); animation:pulse 1.4s ease-in-out infinite; }
.ob-stale .ob-hint { color:var(--red); }

/* ── Misc ── */
.sub-lbl { font-size:.57rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--m3); margin:12px 0 6px; }
.xg-tag  { font-size:.67rem; color:var(--m2); background:rgba(255,255,255,.03); border:1px solid var(--bd); border-radius:6px; padding:2px 8px; display:inline-block; margin-bottom:10px; }
hr.thin  { border:none; border-top:1px solid var(--bd); margin:10px 0; }

/* ── Mobile (téléphone au stade) ── */
@media (max-width: 640px) {
  .block-container { padding-left:.7rem !important; padding-right:.7rem !important; }
  .kpi-grid { grid-template-columns:repeat(2,1fr); }
  .kpi-val  { font-size:1.5rem; }
  .mc-ctr   { min-width:118px; }
  .mc-sc-n  { min-width:44px; height:50px; font-size:2rem; }
  .mc-name  { font-size:.85rem; }
  .mc-body  { padding:16px 14px 12px; }
}
</style>""", unsafe_allow_html=True)


# ── Équipes : normalisation + drapeaux (source unique : pipeline/teams.py) ──────
from pipeline.teams import flag, normalize as _norm
# ── Heure de Montréal (regroupement + ordre + coup d'envoi) ─────────────────────
from pipeline.tz import mtl_date, mtl_time


# ── Helpers ────────────────────────────────────────────────────────────────────
def edge_badge(edge: float) -> str:
    pct = f"+{edge*100:.1f}%"
    if edge >= .13: return f'<span class="eb eb-r">{pct}</span>'
    if edge >= .10: return f'<span class="eb eb-o">{pct}</span>'
    if edge >= .07: return f'<span class="eb eb-y">{pct}</span>'
    if edge >= .05: return f'<span class="eb eb-w">{pct}</span>'
    return f'<span class="eb eb-gr">{edge*100:+.1f}%</span>'

def fmt_mkt(market, side, home="", away=""):
    return {
        ("1x2","home"): f"1X2 {home}", ("1x2","draw"): "Nul", ("1x2","away"): f"1X2 {away}",
        ("btts","yes"): "BTTS Oui", ("btts","no"): "BTTS Non",
        ("over_2.5","over"): "O2.5", ("over_2.5","under"): "U2.5",
        ("over_1.5","over"): "O1.5", ("over_3.5","over"): "O3.5",
        ("team_total","korea_over_1.5"): "Korea O1.5",
        ("buteur_supersub","jimenez_ou_remplacant"): "Jiménez/rempl.",
        ("buteur","mbappe"): "Mbappé buteur",
    }.get((market.lower(), side.lower()), f"{market} {side}")

def kelly(p, cote, br, frac=.25, cap=.05):
    b = cote - 1
    if b <= 0 or p <= 0: return 0.
    return round(min(cap, max(0., (b*p-(1-p))/b*frac))*br, 2)

def load_log() -> pd.DataFrame:
    cols = ["detected_at","date","home_team","away_team","market","side",
            "cote","p_model","edge","stake_eur","bankroll","result","profit_eur","notes"]
    if not BETS_LOG.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(BETS_LOG)
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df

def save_log(df):
    BETS_LOG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BETS_LOG, index=False)

def cur_br(log_df, base):
    if log_df.empty: return base
    return base + pd.to_numeric(log_df.profit_eur, errors="coerce").fillna(0).sum()

def is_paper_mask(df: pd.DataFrame) -> pd.Series:
    """True = pari paper du MODÈLE (auto-loggé) ; False = pari réel placé par Louis.
    Discriminé par la colonne paper_trade (modèle=True, manuel/réel=vide/False)."""
    if df.empty or "paper_trade" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["paper_trade"].astype(str).str.strip().isin(["True", "true", "1"])

def odds_freshness():
    """Renvoie (severity, label_court, label_long) depuis le mtime du fichier de
    cotes — None si absent. fresh <1h · ok <12h · stale au-delà."""
    if not ODDS_PATH.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(ODDS_PATH.stat().st_mtime)
    mins, hours = age.total_seconds() / 60, age.total_seconds() / 3600
    if hours < 1:
        return "fresh", f"il y a {int(mins)} min", "à jour"
    if hours < 12:
        return "ok", f"il y a {int(hours)}h", "encore valides"
    txt = f"il y a {int(hours)}h" if hours < 24 else f"il y a {int(hours // 24)}j"
    return "stale", txt, "périmées — rafraîchis"

def refresh_all():
    """Rafraîchit les cotes (The Odds API : Betclic + Pinnacle) puis résout les
    scores. Fallback bzzoiro si The Odds API échoue (quota épuisé, réseau)."""
    msgs = []
    try:
        from pipeline.fetch_odds_api import fetch_odds_api
        r = fetch_odds_api()
        if r.get("matched"):
            msgs.append(f"💱 {r['matched']} matchs (Betclic+Pinnacle)")
        else:
            raise RuntimeError(r.get("error", "0 match"))
    except Exception:
        try:
            from pipeline.dashboard_odds import refresh_flat_odds
            rb = refresh_flat_odds()
            msgs.append(f"💱 {rb.get('updated', 0)} cotes (bzzoiro fallback)")
        except Exception:
            msgs.append("⚠️ Cotes non rafraîchies")
    try:
        from pipeline.resolve_scores import run as resolve_run
        res = resolve_run(dry_run=False)
        if res["resolved"] > 0:
            msgs.append(f"✅ {res['resolved']} bet(s) résolu(s)")
    except Exception:
        msgs.append("⚠️ Sync scores échoué")
    try:
        # Fige les lignes sharp à l'approche du kickoff + remplit le CLV (KPI d'edge).
        from monitoring.paper_trading import fetch_and_update_clv, snapshot_closing_lines
        snapshot_closing_lines()
        n_clv = fetch_and_update_clv()
        if n_clv:
            msgs.append(f"📈 CLV: {n_clv} pari(s)")
    except Exception:
        pass
    return " · ".join(msgs)


# ── Live scores ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_scores() -> dict:
    # 1. Seed from persistent cache (keeps past finished scores alive)
    out: dict = {}
    try:
        if SCORES_CACHE.exists():
            for key_str, info in json.loads(SCORES_CACHE.read_text()).items():
                parts = key_str.split("|")
                if len(parts) == 3:
                    out[(parts[0], parts[1], parts[2])] = info
    except Exception:
        pass

    # 2. Overlay with live API data
    new_finished: dict = {}
    try:
        from pipeline.config import bzzoiro_headers, load_config
        cfg = load_config()
        for status in ["finished", "live"]:
            try:
                resp = _requests.get(
                    cfg.bzzoiro_base_url + "/events",
                    headers=bzzoiro_headers(cfg),
                    params={"league_id": 27, "status": status, "limit": 100},
                    timeout=5,
                )
                if resp.status_code != 200: continue
                for e in resp.json().get("results", []):
                    if "2026" not in e.get("event_date", ""): continue
                    d = mtl_date(e["event_date"])
                    hn, an = _norm(e["home_team"]), _norm(e["away_team"])
                    info = {
                        "sh": e.get("home_score"), "sa": e.get("away_score"),
                        "status": e.get("status", ""), "minute": e.get("minute"),
                        "home_team": e["home_team"], "away_team": e["away_team"],
                        "time": mtl_time(e["event_date"]),
                    }
                    out[(hn, an, d)] = info
                    if (e.get("status") or "").lower() in ("finished", "ft", "aet", "pen"):
                        new_finished[f"{hn}|{an}|{d}"] = info
            except Exception:
                continue
        # Persist new finished scores
        if new_finished:
            try:
                existing = json.loads(SCORES_CACHE.read_text()) if SCORES_CACHE.exists() else {}
                existing.update(new_finished)
                SCORES_CACHE.write_text(json.dumps(existing))
            except Exception:
                pass
    except Exception:
        pass

    return out


# ── Scan + prédictions ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Analyse en cours…")
def scan():
    from pipeline.model_loader import load_production_model
    from pipeline.value_detector import scan_value_bets
    from pipeline.config import load_config
    with open(ODDS_PATH) as f: odds = json.load(f)
    sharp_path = ROOT/"data/raw/pinnacle_odds_flat.json"
    sharp = json.loads(sharp_path.read_text()) if sharp_path.exists() else None
    wc = pd.read_parquet(ROOT/"data/raw/wc_all.parquet")
    wc26 = wc[wc["date"].dt.year==2026].copy()
    ok = lambda n: bool(re.match(r"^[A-Za-zÀ-ÿ\s\-\'\\.&]+$", n)) and len(n)>2
    wc26 = wc26[wc26.home_team.apply(ok) & wc26.away_team.apply(ok)].sort_values("date")
    wc26["fid"] = wc26.fixture_id.apply(lambda x: str(int(x)) if pd.notna(x) else None)
    matches = [{"home_team":r.home_team,"away_team":r.away_team,"date":mtl_date(r.date),
                "venue":"","is_friendly":False,"is_neutral":True,"fixture_id":r.fid}
               for _,r in wc26.iterrows()]
    kickoff = {r.fid: mtl_time(r.date) for _, r in wc26.iterrows() if r.fid}
    model = load_production_model()
    bets, _ = scan_value_bets(matches, model, odds, load_config(), bankroll=BANKROLL_INIT,
                              calibrator=None, sharp_odds=sharp)
    for b in bets:
        b["kickoff"] = kickoff.get(b.get("fixture_id"), "")
    return bets

@st.cache_data(ttl=300, show_spinner=False)
def predictions():
    from pipeline.model_loader import load_production_model
    from models.base import MatchContext
    wc = pd.read_parquet(ROOT/"data/raw/wc_all.parquet")
    wc26 = wc[wc["date"].dt.year==2026].copy()
    ok = lambda n: bool(re.match(r"^[A-Za-zÀ-ÿ\s\-\'\\.&]+$", n)) and len(n)>2
    wc26 = wc26[wc26.home_team.apply(ok) & wc26.away_team.apply(ok)].sort_values("date")
    model = load_production_model()
    out = {}
    for _, r in wc26.iterrows():
        p = model.predict_outcomes(r.home_team, r.away_team, MatchContext(is_neutral=True))
        out[(r.home_team, r.away_team)] = dict(
            home_win=p.home_win, draw=p.draw, away_win=p.away_win,
            btts=p.btts, over_2_5=p.over_2_5, mu=p.expected_home, nu=p.expected_away,
            date=mtl_date(r.date) if pd.notna(r.date) else "",
            time=mtl_time(r.date) if pd.notna(r.date) else "",
        )
    return out


@st.cache_data(ttl=300, show_spinner=False)
def clv_stats() -> dict:
    """Fige les clôtures sharp/Betclic, remplit le CLV des paris clôturés, et
    renvoie le verdict d'edge (CLV vs Pinnacle) + le détail par marché."""
    from monitoring.paper_trading import run_clv_cycle, BETS_LOG
    rep = run_clv_cycle()
    out = {
        "n_bets": rep.get("n_bets", 0), "mean": rep.get("clv_mean"),
        "beat": rep.get("beat_close_pct"), "sig": rep.get("significant"),
        "ci": rep.get("ci"), "betclic": rep.get("clv_betclic_mean"),
        "n_clv": 0, "by_market": {},
    }
    try:
        df = pd.read_csv(BETS_LOG)
        clv = pd.to_numeric(df.get("clv"), errors="coerce")
        out["n_clv"] = int(clv.notna().sum())
        d2 = df.assign(_c=clv).dropna(subset=["_c"])
        out["by_market"] = {m: (float(g._c.mean()), len(g)) for m, g in d2.groupby("market")}
    except Exception:
        pass
    return out


# ── Match card ─────────────────────────────────────────────────────────────────
def render_match_card(home, away, match_date, score_info, advised_bets, log_df, bankroll, preds, min_edge, kickoff=""):
    fh, fa = flag(home), flag(away)
    placed = log_df[
        log_df.home_team.apply(_norm).eq(_norm(home)) &
        log_df.away_team.apply(_norm).eq(_norm(away))
    ] if not log_df.empty else pd.DataFrame()

    # Determine status
    is_live = is_finished = False
    sh = sa = None
    if score_info:
        sh, sa = score_info.get("sh"), score_info.get("sa")
        st_raw = (score_info.get("status") or "").lower()
        is_finished = st_raw in ("finished","ft","aet","pen")
        is_live = not is_finished and (st_raw in ("live","inprogress","1h","2h","ht") or sh is not None)

    try:
        dt_str = datetime.strptime(match_date, "%Y-%m-%d").strftime("%-d %b").upper()
    except Exception:
        dt_str = match_date[5:]

    kick = (kickoff or "").strip()

    # Status class + badge
    if is_finished:
        status_cls = "mc-ft"
        badge      = '<span class="sb sb-ft">FT</span>'
        dt_part    = f'<span class="mc-dt">{dt_str}</span>'
    elif is_live:
        mt  = (score_info.get("minute","") or "") if score_info else ""
        mt_s = f"&thinsp;{mt}'" if mt else ""
        status_cls = "mc-live"
        badge      = f'<span class="sb sb-live"><span class="ldot"></span>LIVE{mt_s}</span>'
        dt_part    = f'<span class="mc-dt">{dt_str}</span>'
    else:
        status_cls = "mc-ns"
        badge      = f'<span class="sb sb-ns">{dt_str}{(" · " + kick) if kick else ""}</span>'
        dt_part    = ''

    # Score or vs
    if (is_finished or is_live) and sh is not None and sa is not None:
        mt  = (score_info.get("minute", "") or "") if score_info else ""
        lbl = (f"LIVE&nbsp;{mt}'" if mt else "LIVE") if is_live else "FT"
        if is_finished:
            h_cls = " loser" if sh <  sa else ""
            a_cls = " loser" if sa <  sh else ""
        else:
            h_cls = a_cls = ""
        ctr = (f'<div class="mc-sc">'
               f'<div class="mc-sc-n{h_cls}">{sh}</div>'
               f'<span class="mc-sc-sep">–</span>'
               f'<div class="mc-sc-n{a_cls}">{sa}</div>'
               f'</div>'
               f'<div class="mc-sc-lbl">{lbl}</div>')
    else:
        sub = f"{kick} <span style='opacity:.5'>MTL</span>" if kick else dt_str
        ctr = f'<div class="mc-vs">VS</div><div class="mc-vs-dt">{sub}</div>'

    # Probas
    p   = preds.get((home, away), {})
    ph  = p.get("home_win", .33)
    pd_ = p.get("draw", .20)
    pa  = p.get("away_win", .33)
    mu  = p.get("mu", 1.3)
    nu  = p.get("nu", 1.0)

    # Bet pills
    pills = []
    if not placed.empty:
        for _, row in placed.iterrows():
            mkt = fmt_mkt(str(row.market), str(row.side), home, away)
            res = str(row.result or "")
            c   = float(row.cote or 1)
            pnl = float(row.profit_eur or 0)
            s   = float(row.stake_eur or 0)
            if res == "W":
                cls, ico, tail = "bt-w", "✅", f"+{pnl:.2f}€"
            elif res == "L":
                cls, ico, tail = "bt-l", "❌", f"{pnl:.2f}€"
            else:
                cls, ico, tail = "bt-p", "⏳", f"{s:.0f}€"
            pills.append(f'<span class="bt {cls}">{ico}&nbsp;{mkt}&nbsp;@{c:.2f}&nbsp;{tail}</span>')
    else:
        adv_pre = [b for b in advised_bets if b["edge"] >= min_edge]
        if adv_pre:
            n = len(adv_pre)
            pills.append(f'<span class="bt bt-adv">🎯&nbsp;{n}&nbsp;value bet{"s" if n>1 else ""}</span>')

    bets_row = f'<div class="mc-bets">{"".join(pills)}</div>' if pills else ""

    st.markdown(f"""
<div class="mc {status_cls}">
  <div class="mc-strip">
    <span class="mc-comp">CDM 2026</span>
    <div class="mc-meta">{dt_part}{badge}</div>
  </div>
  <div class="mc-body">
    <div class="mc-team">
      <span class="mc-flag">{fh}</span>
      <span class="mc-name">{home}</span>
    </div>
    <div class="mc-ctr">{ctr}</div>
    <div class="mc-team away">
      <span class="mc-name">{away}</span>
      <span class="mc-flag">{fa}</span>
    </div>
  </div>
  <div class="mc-pb">
    <div class="mc-pb-h" style="width:{ph*100:.1f}%"></div>
    <div class="mc-pb-d" style="width:{pd_*100:.1f}%"></div>
    <div class="mc-pb-a" style="width:{pa*100:.1f}%"></div>
  </div>
  <div class="mc-pct">
    <span><span class="pdot pdot-h"></span>{home[:3].upper()} {ph*100:.0f}%</span>
    <span style="color:var(--m3)">Nul {pd_*100:.0f}%</span>
    <span>{away[:3].upper()} {pa*100:.0f}%<span class="pdot pdot-a"></span></span>
  </div>
  {bets_row}
</div>""", unsafe_allow_html=True)

    # Expander: analysis + value bets
    adv_shown = sorted([b for b in advised_bets if b["edge"] >= min_edge], key=lambda x: -x["edge"])
    if placed.empty and not adv_shown:
        return

    with st.expander(f"Analyse · {home} vs {away}", expanded=(match_date==TODAY and is_live)):
        st.markdown(f'<span class="xg-tag">xG &nbsp;{mu:.2f} – {nu:.2f}</span>', unsafe_allow_html=True)

        if not placed.empty:
            st.markdown('<div class="sub-lbl">Bets placés</div>', unsafe_allow_html=True)
            slips = []
            for _, row in placed.iterrows():
                mkt = fmt_mkt(str(row.market), str(row.side), home, away)
                res = str(row.result or "")
                s, c, pnl = float(row.stake_eur or 0), float(row.cote or 1), float(row.profit_eur or 0)
                e  = float(row.edge or 0)
                eb = edge_badge(e) if e > 0.01 else ""
                if res == "W":
                    cls, icon = "slip-w", "✅"
                    pnl_s = f'<span class="slip-pnl-w">+{pnl:.2f}€</span>'
                elif res == "L":
                    cls, icon = "slip-l", "❌"
                    pnl_s = f'<span class="slip-pnl-l">{pnl:.2f}€</span>'
                else:
                    cls, icon = "slip-p", "⏳"
                    pnl_s = '<span class="slip-pnl-p">en cours</span>'
                slips.append(f"""<div class="slip {cls}">
  <span class="slip-icon">{icon}</span>
  <span class="slip-mkt">{mkt}&nbsp;{eb}</span>
  <span class="slip-cote">@{c:.2f}</span>
  <span class="slip-stake">{s:.0f}€</span>
  {pnl_s}
</div>""")
            st.markdown("".join(slips), unsafe_allow_html=True)

        if adv_shown:
            st.markdown('<div class="sub-lbl">Value bets</div>', unsafe_allow_html=True)
            slips = []
            for b in sorted(adv_shown, key=lambda x: -x["edge"])[:6]:
                mkt = fmt_mkt(b["market"], b["side"], home, away)
                k   = kelly(b["p_model"], b["cote"], bankroll)
                eb  = edge_badge(b["edge"])
                pin = ('<span style="font-size:.6rem;color:var(--m2);border:1px solid var(--bd);'
                       'border-radius:4px;padding:0 4px;margin-left:5px">PINN</span>'
                       if b.get("anchor") == "pinnacle" else "")
                slips.append(f"""<div class="slip slip-adv">
  <span class="slip-icon">🎯</span>
  <span class="slip-mkt">{mkt}&nbsp;{eb}{pin}</span>
  <span class="slip-meta">{b['p_model']*100:.0f}%</span>
  <span class="slip-cote">@{b['cote']:.2f}</span>
  <span class="slip-pnl-w">{k:.2f}€&nbsp;K</span>
</div>""")
            st.markdown("".join(slips), unsafe_allow_html=True)

        st.markdown('<hr class="thin">', unsafe_allow_html=True)
        log_key = f"log_{home}_{away}_{match_date}"
        if st.button("+ Enregistrer un bet", key=f"btn_{home}_{away}_{match_date}"):
            st.session_state[log_key] = not st.session_state.get(log_key, False)
        if st.session_state.get(log_key):
            with st.form(key=f"f_{home}_{away}_{match_date}"):
                opts_data = adv_shown[:6]
                opts = [f"{fmt_mkt(b['market'], b['side'], home, away)} · K {kelly(b['p_model'], b['cote'], bankroll):.1f}€"
                        for b in opts_data] or ["Manuel (hors modèle)"]
                k_default = kelly(opts_data[0]["p_model"], opts_data[0]["cote"], bankroll) if opts_data else 2.0
                k_default = max(0.5, round(k_default * 2) / 2)
                fc1, fc2, fc3, fc4 = st.columns([3,1,1,1])
                sel  = fc1.selectbox("Marché", opts)
                idx  = opts.index(sel) if sel in opts else 0
                mise = fc2.number_input("Mise €", min_value=0.5, value=k_default, step=0.5)
                cote_in = fc3.number_input("Cote", min_value=1.01,
                                           value=float(opts_data[idx]["cote"]) if opts_data else 1.5,
                                           step=0.05)
                if fc4.form_submit_button("✓"):
                    b = opts_data[idx] if opts_data else {}
                    df = load_log()
                    save_log(pd.concat([df, pd.DataFrame([{
                        "detected_at": datetime.now().isoformat(), "date": match_date,
                        "home_team": home, "away_team": away,
                        "market": b.get("market","manuel"), "side": b.get("side","manuel"),
                        "cote": cote_in, "p_model": b.get("p_model",0), "edge": b.get("edge",0),
                        "stake_eur": mise, "bankroll": bankroll,
                        "result":"", "profit_eur":"", "notes":"",
                    }])], ignore_index=True))
                    st.session_state[log_key] = False
                    st.rerun()


def _sec(label: str, kind: str = ""):
    cls = f"sec sec-{kind}" if kind else "sec"
    st.markdown(f'<div class="{cls}"><span>{label}</span></div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:22px">
  <span style="font-size:1.5rem;line-height:1">⚽</span>
  <div>
    <div style="font-size:.96rem;font-weight:800;color:#ECEDEE;letter-spacing:.005em">CDM 2026</div>
    <div style="font-size:.64rem;color:rgba(236,237,238,.22);margin-top:1px">Value betting · Elo model</div>
  </div>
</div>""", unsafe_allow_html=True)

    page = st.radio("Nav", ["Tableau de bord", "Mes bets", "Planning"],
                    label_visibility="collapsed")
    st.divider()

    st.markdown('<div style="font-size:.59rem;font-weight:700;color:rgba(236,237,238,.2);text-transform:uppercase;letter-spacing:.13em;margin-bottom:8px">Paramètres</div>', unsafe_allow_html=True)
    bankroll = st.number_input("Bankroll (€)", value=BANKROLL_INIT, step=10.0)
    min_edge = st.slider("Edge min (%)", 0, 20, 5) / 100

    if st.button("↺  Rafraîchir cotes + scores", use_container_width=True):
        with st.spinner("Cotes & scores…"):
            st.session_state["resolve_msg"] = refresh_all()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    _log  = load_log()
    _log  = _log[~is_paper_mask(_log)] if not _log.empty else _log  # vraie bankroll = mes paris
    br_s  = cur_br(_log, bankroll)
    dc_s  = "var(--green)" if br_s >= bankroll else "var(--red)"
    _fr   = odds_freshness()
    _fr_c = {"fresh": "var(--green)", "ok": "var(--amber)", "stale": "var(--red)"}.get(
        _fr[0] if _fr else "", "rgba(236,237,238,.3)")
    _fr_t = f"Cotes {_fr[1]}" if _fr else "Cotes absentes"
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:baseline">
  <span style="font-size:.76rem;color:rgba(236,237,238,.3)">Bankroll</span>
  <span style="font-size:.94rem;font-weight:800;color:{dc_s}">{br_s:.0f}€</span>
</div>
<div style="font-size:.63rem;margin-top:6px"><span style="color:rgba(236,237,238,.3)">{TODAY} · </span><span style="color:{_fr_c}">{_fr_t}</span></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Tableau de bord":
    all_bets    = scan()
    preds       = predictions()
    full_log    = load_log()
    live_scores = fetch_live_scores()

    # Dissociation : MES paris (placés, vrai argent) vs paris du MODÈLE (paper).
    paper_m = is_paper_mask(full_log)
    log_df  = full_log[~paper_m].copy()   # mes paris réels → KPI bankroll/ROI
    paper_df = full_log[paper_m].copy()    # paris paper du modèle → suivi CLV séparé

    br    = cur_br(log_df, bankroll)
    delta = br - bankroll
    resolved = log_df[log_df.result.isin(["W","L"])] if not log_df.empty else pd.DataFrame()
    nw    = int((resolved.result=="W").sum()) if not resolved.empty else 0
    nl    = int((resolved.result=="L").sum()) if not resolved.empty else 0
    staked= pd.to_numeric(resolved.stake_eur, errors="coerce").sum() if not resolved.empty else 0
    roi   = (delta / bankroll * 100) if bankroll else 0
    pend  = int((log_df.result.apply(lambda x: str(x).strip()=="")).sum()) if not log_df.empty else 0

    if "resolve_msg" in st.session_state:
        st.success(st.session_state.pop("resolve_msg"))

    fr = odds_freshness()
    if fr:
        sev, short, long = fr
        st.markdown(f"""
<div class="ob ob-{sev}">
  <span class="ob-dot"></span>
  <span class="ob-txt">Cotes · {short}</span>
  <span class="ob-hint">{long}</span>
</div>""", unsafe_allow_html=True)

    kc = "kpi-g" if delta >= 0 else "kpi-r"
    rc = "kpi-g" if roi   >= 0 else "kpi-r"
    st.markdown('<div class="sec"><span>Mes paris · vrai argent</span></div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi {kc}">
    <div class="kpi-lbl">Bankroll</div>
    <div class="kpi-val">{br:.0f}€</div>
    <div class="kpi-sub {'g' if delta>=0 else 'r'}">{delta:+.2f}€</div>
  </div>
  <div class="kpi {rc}">
    <div class="kpi-lbl">ROI</div>
    <div class="kpi-val">{roi:+.1f}%</div>
    <div class="kpi-sub {'g' if roi>=0 else 'r'}">{staked:.0f}€ misés</div>
  </div>
  <div class="kpi kpi-b">
    <div class="kpi-lbl">W / L</div>
    <div class="kpi-val">{nw}W {nl}L</div>
    <div class="kpi-sub n">{nw+nl} résolus</div>
  </div>
  <div class="kpi kpi-n">
    <div class="kpi-lbl">En attente</div>
    <div class="kpi-val">{pend}</div>
    <div class="kpi-sub n">bets</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Modèle · paper : ce que le bot AURAIT misé (non placé) — CLV = le juge ──
    pres = paper_df[paper_df.result.isin(["W", "L"])] if not paper_df.empty else pd.DataFrame()
    p_nw = int((pres.result == "W").sum()) if not pres.empty else 0
    p_nl = int((pres.result == "L").sum()) if not pres.empty else 0
    p_stake = pd.to_numeric(pres.get("stake_eur"), errors="coerce").sum() if not pres.empty else 0
    p_profit = pd.to_numeric(pres.get("profit_eur"), errors="coerce").sum() if not pres.empty else 0
    p_roi = (p_profit / p_stake * 100) if p_stake else 0
    p_pend = int(len(paper_df) - len(pres))
    st.markdown(f'<div class="sec"><span>Modèle · paper — {len(paper_df)} suggestions, non placées</span></div>', unsafe_allow_html=True)
    rcol = "g" if p_roi >= 0 else "r"
    st.markdown(f"""<div class="clv-chips" style="margin:-4px 0 14px">
  <span class="clv-chip"><b>Résultats paper</b> {p_nw}W {p_nl}L · <span class="{rcol}">ROI {p_roi:+.0f}%</span></span>
  <span class="clv-chip"><b>{p_pend}</b><span class="clv-chip-n">en attente</span></span>
</div>""", unsafe_allow_html=True)
    cv = clv_stats()
    if cv["mean"] is None:
        st.markdown(f"""
<div class="clv-wait">
  <span class="clv-wait-i">📈</span>CLV en attente — se remplit à mesure que les matchs sont joués.
  <b>{cv['n_clv']}/{cv['n_bets']}</b> paris clôturés. Le CLV vs Pinnacle (sharp) dira si le modèle a un edge réel avant même les résultats.
</div>""", unsafe_allow_html=True)
    else:
        mean = cv["mean"]; beat = cv["beat"] or 0; betc = cv["betclic"]
        ci = cv["ci"] or [0, 0]
        ec = "kpi-g" if mean >= 0 else "kpi-r"
        verdict = "EDGE ✓" if cv["sig"] else "à confirmer"
        vc = "g" if cv["sig"] else "n"
        st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi {ec}">
    <div class="kpi-lbl">CLV vs Pinnacle</div>
    <div class="kpi-val">{mean*100:+.2f}%</div>
    <div class="kpi-sub {'g' if mean>=0 else 'r'}">{beat*100:.0f}% battent la clôture</div>
  </div>
  <div class="kpi kpi-b">
    <div class="kpi-lbl">Verdict</div>
    <div class="kpi-val {vc}" style="font-size:1.3rem">{verdict}</div>
    <div class="kpi-sub n">IC95 [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]%</div>
  </div>
  <div class="kpi kpi-n">
    <div class="kpi-lbl">CLV vs Betclic</div>
    <div class="kpi-val">{(betc*100 if betc is not None else 0):+.2f}%</div>
    <div class="kpi-sub n">ligne du book de mise</div>
  </div>
  <div class="kpi kpi-n">
    <div class="kpi-lbl">Échantillon</div>
    <div class="kpi-val">{cv['n_clv']}<span style="font-size:1rem;color:var(--m3)"> / {cv['n_bets']}</span></div>
    <div class="kpi-sub n">paris clôturés</div>
  </div>
</div>""", unsafe_allow_html=True)
        if cv["by_market"]:
            chips = "".join(
                f'<span class="clv-chip"><b>{m}</b><span class="{"g" if v>=0 else "r"}">{v*100:+.1f}%</span>'
                f'<span class="clv-chip-n">{n}</span></span>'
                for m, (v, n) in sorted(cv["by_market"].items(), key=lambda x: -x[1][1]))
            st.markdown(f'<div class="clv-chips">{chips}</div>', unsafe_allow_html=True)

    # ── Matchs : TOUS les fixtures CDM 2026, datés en heure Montréal ───────────
    # Source autoritaire = predictions() : liste complète des matchs (parquet) avec
    # proba Elo + date/heure MTL — indépendante des cotes/bets. Une seule entrée
    # par affiche (clé = paire normalisée) → aucun doublon, aucun match manquant,
    # tout regroupé/ordonné en heure de Montréal (un 22h MTL reste le dernier match
    # de SA soirée même s'il est à 03h Paris / le lendemain en UTC).
    match_reg: dict[tuple, dict] = {}
    def _register(home, away, d, kickoff=""):
        if not d:
            return
        key = (_norm(home), _norm(away))
        e = match_reg.get(key)
        if e is None:
            match_reg[key] = {"home": home, "away": away, "date": d, "kickoff": kickoff or ""}
        elif kickoff and not e["kickoff"]:
            e["kickoff"] = kickoff

    # 1) preds = liste complète, date/heure MTL fiables → fixe nom + jour affichés
    for (h, a), pp in preds.items():
        _register(h, a, pp.get("date", ""), pp.get("time", ""))
    # 2) live = matchs joués/en cours (knockouts dès qu'ils ont de vraies équipes)
    for (hn, an, d), info in live_scores.items():
        _register(info["home_team"], info["away_team"], d, info.get("time", ""))
    # 3) repli : bet loggé sur une affiche absente des sources ci-dessus
    if not log_df.empty:
        for _, row in log_df.iterrows():
            _register(str(row.home_team), str(row.away_team), str(row.date)[:10], "")

    # Index value bets par clé normalisée
    bets_by_match: dict[tuple, list] = defaultdict(list)
    for b in all_bets:
        bets_by_match[(_norm(b["home_team"]), _norm(b["away_team"]))].append(b)

    # Regroupement par jour
    days_map: dict[str, list] = defaultdict(list)
    for e in match_reg.values():
        days_map[e["date"]].append(e)
    all_days = sorted(days_map.keys())  # chronologique

    if not all_days:
        st.markdown("""
<div style="text-align:center;padding:72px 0;color:rgba(236,237,238,.18)">
  <div style="font-size:2.4rem;margin-bottom:12px">⚽</div>
  <div style="font-size:.9rem;font-weight:600">Aucun match à afficher</div>
  <div style="font-size:.78rem;margin-top:6px">Rafraîchis pour charger les données</div>
</div>""", unsafe_allow_html=True)

    for d in all_days:
        # Tri chronologique intra-jour par coup d'envoi (heure Montréal)
        entries = sorted(days_map[d], key=lambda e: (e["kickoff"] or "99:99", e["home"]))
        if not entries: continue

        # Section header — color-coded by era
        try:
            dt_obj = datetime.strptime(d, "%Y-%m-%d")
            if d == TODAY:
                try: dl = dt_obj.strftime("%-d %b").upper()
                except Exception: dl = d
                _sec(f"Aujourd'hui · {dl}", "today")
            elif d < TODAY:
                try: dl = dt_obj.strftime("%A %-d %b").capitalize()
                except Exception: dl = d
                _sec(dl, "past")
            else:
                try: dl = dt_obj.strftime("%A %-d %b").capitalize()
                except Exception: dl = d
                _sec(dl, "future")
        except Exception:
            _sec(d)

        for e in entries:
            home, away = e["home"], e["away"]
            sk = (_norm(home), _norm(away), d)
            render_match_card(
                home, away, d, live_scores.get(sk),
                bets_by_match.get((_norm(home), _norm(away)), []),
                log_df, bankroll, preds, min_edge,
                kickoff=e["kickoff"],
            )


# ══════════════════════════════════════════════════════════════════════════════
# MES BETS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Mes bets":
    log_df = load_log()  # complet : nécessaire pour save_log (ne pas écraser le paper)
    # Vue MES paris réels (indices préservés → les resolve modifient log_df complet).
    real_df = log_df[~is_paper_mask(log_df)] if not log_df.empty else log_df

    if "resolve_msg" in st.session_state:
        st.success(st.session_state.pop("resolve_msg"))

    pending = pd.DataFrame()
    if not real_df.empty:
        pending = real_df[real_df.result.apply(lambda x: str(x).strip()=="")].copy()

    if not pending.empty:
        _sec("Résultats à renseigner", "future")
        for (match_date, home, away), grp in pending.groupby(["date","home_team","away_team"]):
            with st.container(border=True):
                h_col, score_col = st.columns([2, 3])
                h_col.markdown(
                    f"**{flag(home)} {home}** vs **{away} {flag(away)}**  \n"
                    f"<span style='color:rgba(236,237,238,.28);font-size:.76rem'>{match_date}</span>",
                    unsafe_allow_html=True)
                auto_mkts = grp[grp.market.isin(["1X2","1x2","btts","over_2.5","over_1.5","over_3.5"])]
                if not auto_mkts.empty:
                    with score_col:
                        sc1, sc2, sc3, sc4 = st.columns([1,1,1,1])
                        sh = sc1.number_input("Dom.", min_value=0, max_value=20, value=0,
                                              key=f"sh_{home}_{away}_{match_date}")
                        sc2.markdown("<br><center style='color:rgba(236,237,238,.2)'>–</center>",
                                     unsafe_allow_html=True)
                        sa = sc3.number_input("Ext.", min_value=0, max_value=20, value=0,
                                              key=f"sa_{home}_{away}_{match_date}")
                        if sc4.button("Auto →", key=f"auto_{home}_{away}_{match_date}"):
                            from pipeline.resolve_scores import resolve_bet, resolve_team_total
                            changed = False
                            for i, row in grp.iterrows():
                                m, s = str(row.market), str(row.side)
                                res = resolve_team_total(sh, sa, s, home, away) if m=="team_total" else resolve_bet(sh, sa, m, s)
                                if res:
                                    stk = float(row.stake_eur or 0); c = float(row.cote or 1)
                                    log_df.at[i,"result"]     = res
                                    log_df.at[i,"profit_eur"] = stk*(c-1) if res=="W" else (-stk if res=="L" else 0.)
                                    changed = True
                            if changed:
                                save_log(log_df); st.rerun()
                st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:8px 0'>",
                            unsafe_allow_html=True)
                for i, row in grp.iterrows():
                    mkt = fmt_mkt(str(row.market), str(row.side), home, away)
                    sv, cv = float(row.stake_eur or 0), float(row.cote or 1)
                    rc1, rc2, rc3, rc4, rc5 = st.columns([3,1.2,1,1,0.8])
                    rc1.markdown(f"**{mkt}** @ {cv:.2f}")
                    mise_e  = rc2.number_input("Mise", value=sv, step=.5, key=f"me_{i}", label_visibility="collapsed")
                    res_sel = rc3.selectbox("Res", ["—","W","L","V"], key=f"rs_{i}", label_visibility="collapsed")
                    if res_sel != "—":
                        pnl = mise_e*(cv-1) if res_sel=="W" else (-mise_e if res_sel=="L" else 0.)
                        rc4.metric("P&L", f"{pnl:+.2f}€")
                        if rc5.button("✓", key=f"ok_{i}"):
                            log_df.at[i,"result"]     = res_sel
                            log_df.at[i,"profit_eur"] = pnl
                            log_df.at[i,"stake_eur"]  = mise_e
                            save_log(log_df); st.rerun()

    resolved = real_df[real_df.result.isin(["W","L","V"])] if not real_df.empty else pd.DataFrame()
    br = cur_br(real_df, bankroll); delta = br - bankroll

    if not resolved.empty:
        profits = pd.to_numeric(resolved.profit_eur, errors="coerce").fillna(0)
        stakes  = pd.to_numeric(resolved.stake_eur,  errors="coerce").fillna(0)
        roi  = profits.sum()/stakes.sum()*100 if stakes.sum()>0 else 0
        nw   = int((resolved.result=="W").sum()); nl = int((resolved.result=="L").sum())
        kc   = "kpi-g" if delta>=0 else "kpi-r"; rc_ = "kpi-g" if roi>=0 else "kpi-r"
        _sec("Performance", "today")
        st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi {kc}">
    <div class="kpi-lbl">Bankroll</div>
    <div class="kpi-val">{br:.0f}€</div>
    <div class="kpi-sub {'g' if delta>=0 else 'r'}">{delta:+.2f}€</div>
  </div>
  <div class="kpi {rc_}">
    <div class="kpi-lbl">ROI</div>
    <div class="kpi-val">{roi:+.1f}%</div>
    <div class="kpi-sub {'g' if roi>=0 else 'r'}">{profits.sum():+.2f}€</div>
  </div>
  <div class="kpi kpi-b">
    <div class="kpi-lbl">W / L</div>
    <div class="kpi-val">{nw}W {nl}L</div>
    <div class="kpi-sub n">{nw+nl} résolus</div>
  </div>
  <div class="kpi kpi-n">
    <div class="kpi-lbl">Misé</div>
    <div class="kpi-val">{stakes.sum():.0f}€</div>
    <div class="kpi-sub n">&nbsp;</div>
  </div>
</div>""", unsafe_allow_html=True)

        rs  = resolved.sort_values("detected_at")
        cum = bankroll + pd.to_numeric(rs.profit_eur, errors="coerce").fillna(0).cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(cum))), y=cum.values, mode="lines+markers",
            line=dict(color="#34D399", width=2),
            marker=dict(size=9, color=["#34D399" if r=="W" else "#F87171" for r in rs.result],
                        line=dict(width=0)),
            hovertemplate="Bet #%{x}<br>Bankroll : %{y:.2f}€<extra></extra>",
        ))
        fig.add_hline(y=bankroll, line_dash="dot", line_color="rgba(255,255,255,.1)")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=210, margin=dict(l=0,r=0,t=8,b=0), showlegend=False,
            xaxis=dict(color="rgba(236,237,238,.2)", gridcolor="rgba(255,255,255,.04)", title=""),
            yaxis=dict(color="rgba(236,237,238,.2)", gridcolor="rgba(255,255,255,.04)", title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    if not real_df.empty:
        _sec("Historique", "past")
        rows = []
        for _, r in real_df.sort_values("detected_at", ascending=False).iterrows():
            mkt = fmt_mkt(str(r.market), str(r.side), str(r.home_team), str(r.away_team))
            res = str(r.result or ""); pnl = float(r.profit_eur or 0)
            rows.append({
                "Date": str(r.date)[:10], "Match": f"{r.home_team} vs {r.away_team}",
                "Marché": mkt, "Cote": float(r.cote or 0), "Mise": f"{float(r.stake_eur or 0):.2f}€",
                "Edge": f"{float(r.edge or 0)*100:+.1f}%",
                "Résultat": {"W":"✅ W","L":"❌ L","V":"↩ V"}.get(res,"⏳ —"),
                "P&L": f"{pnl:+.2f}€" if res in ("W","L") else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={"Cote": st.column_config.NumberColumn(format="%.2f")})

    if not real_df.empty:
        with st.expander("Modifier ou supprimer un bet"):
            def _lbl(i):
                r = log_df.loc[i]
                m = fmt_mkt(str(r.market), str(r.side), str(r.home_team), str(r.away_team))
                res = str(r.result or "").strip() or "—"
                return f"{str(r.date)[:10]} · {r.home_team} v {r.away_team} · {m} @{float(r.cote or 1):.2f} · {res}"
            sel_i = st.selectbox("Bet", list(real_df.index), format_func=_lbl, key="ed_sel")
            row = log_df.loc[sel_i]
            c1, c2, c3 = st.columns(3)
            e_mise = c1.number_input("Mise €", min_value=0.0, value=float(row.stake_eur or 0), step=0.5, key="ed_mise")
            e_cote = c2.number_input("Cote", min_value=1.0, value=float(row.cote or 1), step=0.05, key="ed_cote")
            _ro = ["—", "W", "L", "V"]
            cur = str(row.result or "").strip()
            e_res = c3.selectbox("Résultat", _ro, index=_ro.index(cur) if cur in _ro else 0, key="ed_res")
            b1, b2 = st.columns(2)
            if b1.button("Mettre à jour", use_container_width=True, key="ed_upd"):
                log_df.at[sel_i, "stake_eur"] = e_mise
                log_df.at[sel_i, "cote"] = e_cote
                if e_res == "—":
                    log_df.at[sel_i, "result"] = ""
                    log_df.at[sel_i, "profit_eur"] = ""
                else:
                    log_df.at[sel_i, "result"] = e_res
                    log_df.at[sel_i, "profit_eur"] = (
                        e_mise * (e_cote - 1) if e_res == "W" else (-e_mise if e_res == "L" else 0.0))
                save_log(log_df); st.rerun()
            if b2.button("Supprimer ce bet", use_container_width=True, key="ed_del"):
                save_log(log_df.drop(index=sel_i).reset_index(drop=True)); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PLANNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Planning":
    all_bets = scan(); log_df = load_log()
    real_log = log_df[~is_paper_mask(log_df)] if not log_df.empty else log_df  # placé = par moi
    placed_ids = set()
    if not real_log.empty:
        for _, r in real_log.iterrows():
            placed_ids.add((str(r.date), str(r.home_team), str(r.away_team), str(r.market), str(r.side)))

    f1, f2 = st.columns(2)
    date_from  = f1.date_input("À partir du", value=date.today())
    mkt_filter = f2.multiselect("Marchés", ["1X2","btts","over_1.5","over_2.5","over_3.5","team_total"])

    filtered = [b for b in all_bets if b["edge"]>=min_edge and b["date"]>=str(date_from)
                and (not mkt_filter or b["market"] in mkt_filter)]
    st.caption(f"{len(filtered)} bets · {len({b['date'] for b in filtered})} jours")

    by_date: dict = defaultdict(lambda: defaultdict(list))
    for b in filtered:
        by_date[b["date"]][(b["home_team"], b["away_team"])].append(b)

    for day in sorted(by_date.keys()):
        day_n = sum(len(v) for v in by_date[day].values())
        try: lbl = datetime.strptime(day,"%Y-%m-%d").strftime("%A %-d %b").capitalize()
        except Exception: lbl = day
        kind = "today" if day==TODAY else ("past" if day<TODAY else "future")
        if day == TODAY: lbl = f"Aujourd'hui · {lbl}"
        _sec(f"{lbl} · {day_n} bets", kind)
        rows = []
        for (home, away), bets in sorted(
            by_date[day].items(),
            key=lambda kv: (kv[1][0].get("kickoff", "99:99"), kv[0]),
        ):
            for b in sorted(bets, key=lambda x: -x["edge"]):
                mkt = fmt_mkt(b["market"], b["side"], home, away)
                k   = kelly(b["p_model"], b["cote"], bankroll)
                rows.append({
                    "": "✅" if (b["date"],home,away,b["market"],b["side"]) in placed_ids else "",
                    "Heure": b.get("kickoff", ""),
                    "Match": f"{flag(home)} {home} vs {away} {flag(away)}",
                    "Marché": mkt, "Cote": b["cote"],
                    "Modèle": f"{b['p_model']*100:.1f}%",
                    "No-vig": f"{b['p_novig']*100:.1f}%",
                    "Edge": f"+{b['edge']*100:.1f}%",
                    "Kelly": f"{k:.2f}€",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={"Cote": st.column_config.NumberColumn(format="%.2f")})
