# Betting Bot — CDM 2026

## Contexte projet
Système de détection de value bets sur la Coupe du Monde 2026 (début 11 juin 2026).
Pipeline : data → modèle Dixon-Coles → value detector → alertes Telegram.
Horizon immédiat : matchs amicaux de préparation CDM (maintenant) puis phase de groupes.

## Stack technique
- Python 3.11+
- pandas, numpy, scipy, requests, python-telegram-bot
- PyMC (Bayesian inference, phase 2)
- GPU disponible : 2x GTX 1080 Ti (CUDA) — réservé pour inférence live et NLP features
- API-Football (RapidAPI) — clé dans config.yaml
- Betfair Exchange API (phase automatisation)

## Structure du projet
```
betting-bot/
├── CLAUDE.md
├── config.yaml              # API keys, thresholds, bankroll
├── data/
│   ├── raw/                 # Données brutes API
│   └── processed/           # Features engineered
├── models/
│   ├── dixon_coles.py       # Modèle core
│   └── calibration.py       # Isotonic regression
├── pipeline/
│   ├── fetch_data.py        # Pull API-Football
│   ├── features.py          # Feature engineering
│   └── value_detector.py   # Edge calculation + Kelly
├── alerts/
│   └── telegram_bot.py     # Bot alertes
├── backtest/
│   └── backtest_wc2022.py  # Validation sur CDM 2022
└── notebooks/
    └── exploration.ipynb
```

## Config attendue (config.yaml)
```yaml
api_football:
  key: "TON_API_KEY_ICI"
  base_url: "https://api-football-v1.p.rapidapi.com/v3"

telegram:
  token: "TON_BOT_TOKEN_ICI"
  chat_id: "TON_CHAT_ID_ICI"

model:
  min_edge_threshold: 0.05      # 5% edge minimum pour alerter
  kelly_fraction: 0.25           # Quart-Kelly (prudent)
  max_kelly_bet: 0.05            # Max 5% bankroll par bet

bankroll:
  initial: 200                   # En euros, à ajuster
```

## Phase 1 — Data pipeline (PRIORITÉ 1)

### fetch_data.py
Implémenter les fonctions suivantes :

```python
fetch_world_cup_matches(year: int) -> pd.DataFrame
# Pull tous les matchs d'une CDM via API-Football
# Endpoint : /fixtures?league=1&season={year}
# CDM 2018 : league_id=1, CDM 2022 : league_id=1
# Amicaux sélections nationales : league_id=10

fetch_team_stats(team_id: int, season: int) -> dict
# Stats agrégées : buts marqués/encaissés, xG si dispo, corners

fetch_h2h(team1_id: int, team2_id: int) -> pd.DataFrame
# 10 derniers matchs entre deux équipes

fetch_friendly_matches() -> pd.DataFrame
# Amicaux CDM en cours (league_id=10, saison 2026)
```

Sauvegarder en parquet dans data/raw/.

### IDs équipes CDM 2026 à récupérer
Les 48 équipes qualifiées incluent :
- UEFA (16) : France, Espagne, Angleterre, Allemagne, Portugal, Pays-Bas, Belgique, Italie, Croatie, Autriche, Suisse, Danemark, Pologne, Serbie, Türkiye, Bosnie-Herzégovine, Suède, Tchéquie
- CONMEBOL (6) : Argentine, Brésil, Colombie, Uruguay, Équateur, Paraguay
- AFC (9) : Japon, Corée du Sud, Australie, Iran, Arabie Saoudite, Qatar, Jordanie, Ouzbékistan, Irak
- CAF (10) : Maroc, Sénégal, Égypte, Algérie, Ghana, Côte d'Ivoire, Tunisie, Afrique du Sud, Cabo Verde, RD Congo
- CONCACAF (6) : USA, Mexique, Canada, + 3 autres
- OFC (1) : Nouvelle-Zélande

Fetcher la liste complète via /teams?league=1&season=2026.

## Phase 2 — Modèle Dixon-Coles (PRIORITÉ 2)

### dixon_coles.py
Implémenter le modèle complet :

**Principe :**
- Chaque équipe a un paramètre d'attaque α et de défense β
- Score home ~ Poisson(α_home × β_away × γ) où γ = home advantage
- Score away ~ Poisson(α_away × β_home)
- Correction Dixon-Coles sur scores serrés (0-0, 1-0, 0-1, 1-1) via paramètre ρ

**Interface attendue :**
```python
class DixonColesModel:
    def fit(self, matches: pd.DataFrame) -> None
    # matches : DataFrame avec colonnes home_team, away_team, home_goals, away_goals, date
    # Optimisation MLE via scipy.optimize.minimize
    # Pondération temporelle : matchs récents poids plus élevé (decay=0.0065)

    def predict_score_matrix(self, home: str, away: str, max_goals: int = 8) -> np.ndarray
    # Retourne matrice 9x9 des probabilités P(home=i, away=j)

    def predict_outcomes(self, home: str, away: str) -> dict
    # Retourne {"home_win": p, "draw": p, "away_win": p,
    #           "over_2.5": p, "btts": p, "expected_home": mu, "expected_away": mu}

    def predict_corners(self, home: str, away: str) -> dict
    # Modèle Poisson séparé sur les corners (à entraîner séparément)
```

**Pondération temporelle :**
```python
weight = exp(-decay * days_since_match)
# decay = 0.0065 → match il y a 1 an pèse ~0.09
```

**Features supplémentaires à intégrer :**
- FIFA ranking (proxy de force relative)
- Altitude du stade (critique pour matchs au Mexique — Mexico City à 2240m)
- Contexte match : amical vs officiel (réduire poids amicaux × 0.5)

## Phase 3 — Value Detector (PRIORITÉ 3)

### value_detector.py
```python
def compute_edge(p_model: float, cote: float) -> float:
    p_implicite = 1 / cote
    return p_model - p_implicite

def kelly_size(edge: float, cote: float, fraction: float = 0.25) -> float:
    # Kelly criterion : f = (bp - q) / b où b = cote-1, p = p_model, q = 1-p_model
    b = cote - 1
    p = edge + (1 / cote)  # p_model
    q = 1 - p
    full_kelly = (b * p - q) / b
    return max(0, full_kelly * fraction)

def scan_value_bets(matches: list, model: DixonColesModel, bookmaker_odds: dict) -> list:
    # Pour chaque match et chaque marché disponible
    # Retourne liste de value bets avec edge > threshold
    # Marchés à scanner : 1X2, over/under 2.5, BTTS, handicap asiatique
```

**Marchés prioritaires :**
1. Résultat 1X2
2. Over/Under 2.5 buts
3. BTTS (both teams to score)
4. Corners over/under (modèle séparé)
5. Handicap asiatique

## Phase 4 — Alertes Telegram (PRIORITÉ 4)

### telegram_bot.py
Format d'alerte :
```
🎯 VALUE BET DÉTECTÉE

⚽ France vs Australie
📅 15 juin 2026 — 20h00
🏟️ Los Angeles

Marché : Over 2.5 buts
Cote Betclic : 1.85
Probabilité modèle : 61.2%
Probabilité implicite : 54.1%
Edge : +7.1% ✅

Kelly recommandé : 2.3% bankroll
Mise suggérée : 4.60€ (bankroll 200€)

Confiance modèle : ★★★☆☆
```

Envoyer aussi un résumé quotidien des matchs du jour à 9h00.

## Phase 5 — Backtest CDM 2022

### backtest_wc2022.py
- Entraîner le modèle sur données 2018-2021 (sélections nationales)
- Prédire tous les matchs CDM 2022
- Simuler les bets avec Kelly quarter sur toutes les value bets détectées
- Métriques à rapporter :
  - Brier Score vs baseline (cotes bookmaker)
  - Log-loss
  - ROI simulé
  - Calibration curve (reliability diagram)
  - Nombre de value bets détectées / converties

Seuil de validation : si ROI simulé > 0% sur CDM 2022 → modèle valide pour CDM 2026.

## Angles spéciaux à implémenter

### Amicaux CDM (maintenant)
- Réduire confiance modèle sur amicaux (rotations massives)
- Feature : "2 équipes annoncées" par le coach → flag dans les données
- Focus sur marchés corners et BTTS (moins sensibles aux compos)
- Buteurs remplaçants : alerter quand cote buteur > 6.0 sur joueur confirmé entrant

### Altitude Mexico
```python
STADIUM_ALTITUDE = {
    "Estadio Azteca": 2240,      # Mexico City
    "Estadio Akron": 1650,       # Guadalajara
    "Estadio BBVA": 500,         # Monterrey
    # Stades USA et Canada : altitude standard
}

def altitude_adjustment(altitude: int) -> float:
    # Réduire over/under de ~0.3 buts par 1000m d'altitude
    return -0.3 * (altitude / 1000)
```

## Ordre d'exécution recommandé

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Fetch data
python pipeline/fetch_data.py --years 2018 2022 --include-friendlies

# 3. Train modèle
python models/dixon_coles.py --train --seasons 2018 2019 2020 2021 2022

# 4. Backtest validation
python backtest/backtest_wc2022.py

# 5. Si backtest OK → lancer le scanner
python pipeline/value_detector.py --live --alert-telegram

# 6. Bot Telegram en continu
python alerts/telegram_bot.py
```

## Contraintes et règles

- Ne jamais miser plus de 5% du bankroll sur un seul pari (Kelly cap)
- Toujours utiliser quart-Kelly minimum (kelly_fraction=0.25)
- Ne pas automatiser le placement — alertes Telegram uniquement, placement manuel
- Logger tous les bets détectés et résultats dans data/bets_log.csv pour suivi ROI réel
- Backtest obligatoire avant tout usage en production

## Ce qui n'est PAS dans le scope
- Placement automatique sur Betclic (pas d'API officielle, risque compte)
- Combinés (EV négatif sauf edge confirmée sur chaque leg)
- Score exact (marge bookmaker 15-20%)
- Esports (manipulation trop fréquente)
