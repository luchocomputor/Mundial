# Plan d'amélioration — Détecteur de value bets CDM 2026

> Document de référence technique. Objectif : transformer un prototype propre mais non validé en un système de paris quantitatif **rigoureux, reproductible et réellement profitable**, sur la base de méthodes prouvées dans la littérature et la pratique des parieurs « sharp ».
>
> Principe directeur unique : **on ne cherche pas à prédire les matchs mieux que tout le monde. On cherche à battre la ligne de clôture (closing line) du marché, de façon mesurable et répétée.** Tout le reste en découle.

---

## 0. Philosophie et KPI souverain

### 0.1 La seule vérité terrain : le Closing Line Value (CLV)

La cote de clôture (juste avant le coup d'envoi), dé-vigorisée, est **le meilleur estimateur public de la probabilité réelle** d'un événement sportif. C'est un fait empirique robuste (efficience quasi-forte des marchés de paris liquides, cf. Pinnacle, travaux sur l'efficience des marchés sportifs).

Conséquence opérationnelle :

- Si nos paris obtiennent **systématiquement une meilleure cote que la clôture** (CLV > 0, statistiquement significatif), nous avons un edge réel, **indépendamment du résultat des matchs sur l'échantillon**.
- Si notre CLV est nul ou négatif, **aucun ROI positif observé n'est crédible** : c'est de la variance, pas de la compétence.

→ Le **CLV est le KPI souverain**. Le ROI est un KPI secondaire, bruité, à ne juger que sur >1000 paris.

### 0.2 Hiérarchie des métriques

1. **CLV moyen** et % de paris « beat the close » (gold standard).
2. **Ranked Probability Score (RPS)** pour le 1X2 (métrique correcte pour les issues ordinales football — Constantinou & Fenton 2012). Pas le Brier seul.
3. **Log-loss** et **calibration** (ECE / reliability diagrams) par marché.
4. **ROI** et **drawdown** simulés (walk-forward, frais inclus), uniquement comme contrôle de cohérence.

---

## 1. Diagnostic synthétique de l'existant

Rappel des défauts identifiés à l'audit (du plus grave au mineur) :

| # | Problème | Gravité | Effet |
|---|----------|---------|-------|
| 1 | Backtest sur **cotes synthétiques dérivées du modèle** → edge toujours négatif, 0 pari, validation circulaire | Bloquant | Aucune preuve d'edge |
| 2 | Backtest lit des fichiers/`status` qui n'existent pas (`wc_2022.parquet`, `status=="FT"`) alors que le fetch produit `wc_all.parquet`/`"finished"` | Bloquant | Backtest ne tourne pas |
| 3 | `decay=0.0065` ancré à 2026 sur données 2018-2022 → poids ≈ 1e-6, entraînement dégénéré | Bloquant | Modèle vide de signal |
| 4 | Avantage domicile `gamma` appliqué aux **matchs neutres** (la quasi-totalité d'une CDM), `is_neutral` ignoré | Majeur | Biais 1X2 systématique |
| 5 | Données d'entraînement = **seulement CDM + amicaux** ; équipes inconnues → `alpha=beta=0` | Majeur | Faux edges sur outsiders |
| 6 | Pas de contrainte d'identifiabilité (`sum(attack)=0`) | Majeur | Paramètres instables |
| 7 | Blend avec bzzoiro **déjà calibré sur le marché** → on dilue notre edge dans le marché | Majeur | Edge rogné/artificiel |
| 8 | Edge calculé sur cote **brute (vig inclus)**, pas de dé-vigorisation | Majeur | Edge mal défini |
| 9 | Filtre regex `^[W|L|R|Q|H|G|...]` exclut Qatar, Ghana, Haiti… | Moyen | Matchs réels ignorés |
| 10 | FIFA ranking / corners / confiance : features calculées mais **mortes** | Mineur | Dette |
| 11 | Token API committé en clair | Mineur (sécu) | Fuite credential |

Ces points structurent les phases ci-dessous.

---

## 2. Principes directeurs (state of the art)

1. **No look-ahead, jamais.** Toute feature d'un match au temps *t* n'utilise que de l'information disponible strictement avant *t*. Validation exclusivement en **walk-forward** (rolling-origin), jamais de k-fold aléatoire sur des séries temporelles.
2. **Pooling partiel plutôt que points-estimates.** Pour 48 sélections dont beaucoup sont peu observées (Cabo Verde, Jordanie, Ouzbékistan), un modèle **hiérarchique bayésien** régularise les équipes peu vues vers la moyenne de leur confédération. Fini les `alpha=0`.
3. **Quantifier l'incertitude et la propager au sizing.** Une proba avec grande variance d'estimation mérite une mise plus petite. Kelly sur l'espérance brute est sur-optimiste (cf. Kelly + incertitude paramétrique).
4. **Dé-vigoriser proprement** (méthode de Shin, ou power method) avant tout calcul d'edge.
5. **Mesurer le bon objectif** : RPS + calibration + CLV, pas le Brier seul ni le ROI brut.
6. **Ensembler par stacking appris**, pas par poids 60/40 arbitraires.
7. **Paper trading obligatoire** avant tout euro réel : tracer le CLV en conditions réelles ≥ 4-6 semaines.
8. **Reproductibilité** : seed fixé, environnement figé, données versionnées, runs loggés.

---

## 3. Roadmap par phases

### Phase 0 — Fondations & rigueur expérimentale *(prérequis, ~2-3 j)*

**Objectif :** rendre le projet exécutable, sûr et reproductible avant toute modélisation.

- [ ] **Environnement reproductible** : `pyproject.toml` + lockfile (uv/poetry), versions épinglées, `make setup`.
- [ ] **Secrets hors du code** : sortir le token bzzoiro dans `.env` (déjà présent dans le dossier), lire via `os.environ`. `config.yaml` ne contient que des seuils. Purger le token de l'historique git (`git filter-repo`) et le régénérer.
- [ ] **Schéma de données unifié** (contrat unique) : un seul `status` canonique (`finished`/`upcoming`), un seul format de fichiers, documenté dans `data/SCHEMA.md`. Corriger l'incohérence fetch ↔ backtest (#2).
- [ ] **Couche de validation de données** (pandera ou pydantic) : types, plages, unicité `fixture_id`, pas de fuite de score sur matchs `upcoming`.
- [ ] **Tests** : `pytest` sur `compute_edge`, `kelly_size`, dé-vigorisation, parsing API, filtre placeholders. Corriger le filtre regex (#9) → liste blanche d'équipes valides issue des données, pas un test sur la 1ʳᵉ lettre.
- [ ] **CI** : lint (ruff) + tests + `py_compile` sur push.

**Definition of Done :** `make test` vert ; `make pipeline` tourne de bout en bout sur données réelles sans FileNotFoundError ; aucun secret dans le repo.

---

### Phase 1 — Données : la fondation qui manque *(le levier #1, ~3-5 j)*

> Un modèle de force d'équipe ne vaut que par la richesse de son historique. Aujourd'hui : CDM + amicaux seulement. C'est la cause racine des faux edges.

**Élargir massivement les sources** (toutes via l'API bzzoiro, qui couvre 30+ ligues, 64k+ matchs, 15 ans) :

- [ ] **Toutes les compétitions de sélections** : qualifications (UEFA, CONMEBOL, CAF, AFC, CONCACAF), **Nations League**, **Euro, Copa América, CAN, Coupe d'Asie, Gold Cup**, amicaux, CDM. C'est ce qui donne des liens entre confédérations (essentiel pour comparer des équipes qui ne se rencontrent presque jamais).
- [ ] **Profondeur ≥ 8-10 ans** pondérée par decay (cf. ci-dessous), pas 4 ans.
- [ ] **Features par match, sans fuite** :
  - **Terrain neutre** (`is_neutral`) — à exploiter dans le modèle (#4).
  - **Repos / congestion** : jours depuis le dernier match, nombre de matchs sur 14 j.
  - **Voyage & altitude** : distance parcourue, altitude du stade (remplacer le `-0.3/1000m` arbitraire par un coefficient **estimé sur données**, et appliqué de façon asymétrique selon l'adaptation des équipes).
  - **Force de l'effectif** : valeur marchande agrégée (Transfermarkt) comme proxy puissant et stable, en complément du rating dynamique.
  - **Contexte** : enjeu (officiel vs amical), phase de tournoi.
  - **xG** quand disponible (bzzoiro fournit xG + shot maps) → modèle « goals » ET modèle « xG ».
- [ ] **Cotes historiques de clôture** : indispensables pour le backtest CLV et la dé-vigorisation. Récupérer via l'Odds API bzzoiro (14+ bookmakers + Polymarket). **Sans cotes de clôture réelles, aucune validation n'est possible** (corrige #1).

**Definition of Done :** dataset unifié ≥ 15-20k matchs de sélections avec, pour chaque match jouable, au moins une cote de clôture 1X2/OU/BTTS d'un bookmaker « sharp » (Pinnacle de préférence).

---

### Phase 2 — Protocole de validation & métriques *(à construire AVANT les modèles, ~2-3 j)*

> Erreur classique : coder le modèle puis bricoler un backtest. On fait l'inverse. **Le banc de test est l'actif le plus précieux du projet.**

- [ ] **Walk-forward (rolling-origin)** : pour chaque date de prédiction, n'entraîner que sur le passé strict. Réentraînement périodique (hebdo/mensuel). Aucune fuite temporelle.
- [ ] **Métriques implémentées proprement** :
  - **RPS** (Ranked Probability Score) pour 1X2 — métrique de référence football (Constantinou & Fenton 2012 ; Wheatcroft 2019).
  - **Log-loss** et **Brier** par marché binaire (OU, BTTS).
  - **Calibration** : reliability diagram + **ECE** (Expected Calibration Error) par marché.
  - **Baseline obligatoire = le marché lui-même** (cotes de clôture dé-vigorisées). *Le modèle n'a de valeur que s'il s'approche du RPS du marché, et la value n'existe que là où il s'en écarte sur les cotes d'ouverture.*
- [ ] **CLV tracking** : pour chaque pari simulé sur cote d'ouverture/intermédiaire, comparer à la cote de clôture dé-vigorisée → distribution du CLV, test de significativité (t-test / bootstrap).
- [ ] **Backtest réaliste** : frais/commission, mises arrondies, liquidité, **pas de bet sur la cote de clôture elle-même** (on parie en amont, on mesure vs clôture).

**Definition of Done :** un module `evaluation/` qui, donné (prédictions, cotes ouverture, cotes clôture, résultats), sort RPS/log-loss/ECE/CLV/ROI/drawdown avec intervalles de confiance bootstrap, et compare au marché.

---

### Phase 3 — Dé-vigorisation des cotes *(rapide mais critique, ~1 j)*

> L'« edge » actuel est calculé sur `1/cote` brute, qui inclut la marge du bookmaker (vig). C'est mal défini (#8).

- [ ] Implémenter plusieurs méthodes de retrait du vig et **choisir empiriquement la meilleure** (celle dont la proba no-vig prédit le mieux les résultats) :
  - **Normalisation multiplicative** (baseline naïve).
  - **Méthode de Shin** (Shin 1992/1993) — modélise la part d'« insiders », généralement la plus précise sur 1X2.
  - **Power method / additive** — alternatives robustes.
- [ ] Définir l'edge sur **probabilités no-vig** : `edge = p_model − p_novig_close`, et ne parier que si l'edge dépasse un seuil **calibré** (pas 5% arbitraire) tenant compte de l'incertitude du modèle.

**Definition of Done :** module `odds/devig.py` testé ; benchmark des 3 méthodes ; la proba no-vig de clôture bat le modèle en RPS (sanity check d'efficience de marché).

---

### Phase 4 — Modèles de force d'équipe (cœur SOTA) *(~2-3 sem)*

On construit une **échelle de complexité croissante**, chaque niveau devant battre le précédent **en walk-forward**, sinon on ne le garde pas.

#### 4.1 Baselines fortes (à ne jamais sous-estimer)

- [ ] **Elo / pi-ratings** : rating dynamique mis à jour match par match. Les **pi-ratings** (Constantinou & Fenton 2013) et l'Elo football (Hvattum & Arntzen 2010) sont des baselines redoutables, robustes aux données sparse, et idéaux pour les sélections. Servent de **feature** et de **garde-fou** (un modèle complexe qui ne bat pas Elo est suspect).

#### 4.2 Dixon-Coles, fait correctement

Réparer l'existant plutôt que le jeter :

- [ ] **Contrainte d'identifiabilité** `mean(attack)=0` (#6).
- [ ] **Decay correct** : référence ancrée à la date de prédiction (walk-forward), half-life calibrée (typiquement ~ 1-2 ans pour les sélections, à optimiser par RPS) — surtout pas des poids à 1e-6 (#3).
- [ ] **Terrain neutre** : `home_advantage = gamma * (1 − is_neutral)` ; estimer aussi un avantage résiduel « pays hôte » (#4).
- [ ] **Vectorisation** de la log-vraisemblance (NumPy) + gradient analytique → fit ×100 plus rapide que l'`iterrows` actuel.

#### 4.3 Bivariate Poisson & extensions

- [ ] **Bivariate Poisson** (Karlis & Ntzoufras 2003) : capture la corrélation entre buts domicile/extérieur, mieux que deux Poisson indépendants + correction rho.
- [ ] Gérer la **sur/sous-dispersion** : modèles **Conway-Maxwell-Poisson** ou **Negative Binomial** si les données le justifient (test de dispersion).

#### 4.4 Modèle hiérarchique bayésien (la pièce maîtresse pour 48 sélections)

> C'est **la** réponse SOTA au problème des équipes peu observées et à la quantification d'incertitude.

- [ ] **Hiérarchie bayésienne** (PyMC, déjà dans `requirements`) à la Baio & Blangiardo (2010) / Rue & Salvesen (2000) :
  - Attaque/défense par équipe **avec priors hiérarchiques par confédération** → Cabo Verde régularisé vers la moyenne CAF, etc. (résout #5 sans inventer de chiffres).
  - **Force variant dans le temps** : marche aléatoire / état-espace sur les ratings (dynamic model).
  - Sortie = **distribution prédictive complète** → on récupère la variance de chaque proba, réutilisée au sizing (Phase 6).
- [ ] Inférence : NUTS pour la recherche, **approximation variationnelle / point-estimate MAP** pour la prod live (latence).

#### 4.5 Piste xG (signal complémentaire)

- [ ] Modèle parallèle entraîné sur **xG** plutôt que buts réels (réduit le bruit des finitions). bzzoiro fournit xG par tir + shot maps. À ensembler en Phase 5.

**Definition of Done :** au moins un modèle bat Elo **et** s'approche du RPS du marché en walk-forward, avec calibration ECE < seuil défini.

---

### Phase 5 — Calibration & ensembling appris *(~1 sem)*

> Le blend 60/40 actuel est arbitraire et **dilue notre signal dans bzzoiro, lui-même calibré sur le marché** (#7). On remplace par un ensemblage appris et une calibration sérieuse.

- [ ] **Stacking / meta-modèle** : un modèle de second niveau (régression logistique multinomiale régularisée, ou gradient boosting léger) apprend les **poids optimaux** entre les sous-modèles (DC, bivariate, bayésien, xG, Elo, et éventuellement bzzoiro **en feature, pas en oracle**), entraîné en walk-forward pour éviter la fuite.
- [ ] **Calibration post-hoc** : isotonic (déjà esquissée), **Platt/beta calibration**, comparées par ECE. Calibrer **par marché** et idéalement **par tranche de favori/outsider**.
- [ ] **Décision claire sur bzzoiro** : si bzzoiro est calibré sur le marché, alors près de la clôture il **ne peut pas** générer d'edge vs marché. On le garde comme feature stabilisatrice et pour les marchés où nos modèles sont faibles, mais **la source d'edge doit venir de notre vision propre face aux cotes d'ouverture**.

**Definition of Done :** l'ensemble calibré bat chaque sous-modèle individuel en RPS/log-loss walk-forward ; ECE proche de 0 ; poids appris documentés.

---

### Phase 6 — Edge, sizing & gestion du risque *(~1 sem)*

> Le sizing actuel : Kelly quart sur l'edge brut, cap 5%. Correct comme base, mais sur-optimiste car il ignore l'incertitude d'estimation et la corrélation entre paris.

- [ ] **Edge no-vig** : `edge = p_model_calibré − p_novig_close` (Phase 3).
- [ ] **Kelly fractionné conscient de l'incertitude** :
  - Conserver le **quart-Kelly** (fraction 0.25) et le cap (≤ 2-5 % bankroll) — prudence justifiée empiriquement.
  - **Shrinkage de l'edge** par l'incertitude du modèle (variance prédictive bayésienne de Phase 4.4) : `f ∝ edge / variance`. Un edge incertain → mise réduite. (Cf. Kelly sous incertitude paramétrique.)
- [ ] **Kelly simultané / corrélé** : sur une même journée, plusieurs paris sur des matchs/marchés corrélés (ex. plusieurs « over ») ne sont pas indépendants. Optimiser le vecteur de mises sous contrainte de risque global (approche portefeuille), ou au minimum **plafonner l'exposition totale par jour**.
- [ ] **Contrôles de risque** : drawdown max, stop-loss mensuel, exposition max par match/marché/journée, journalisation de chaque pari avec son **edge, sa proba, sa cote, et la cote de clôture obtenue ensuite**.
- [ ] **Seuil d'edge dérivé du backtest**, pas fixé à 5 % a priori : on garde la zone d'edge où le CLV est positif et significatif.

**Definition of Done :** simulateur de bankroll réaliste ; CLV positif sur la zone d'edge retenue ; respect strict des caps ; log enrichi avec suivi CLV post-clôture.

---

### Phase 7 — Backtest rigoureux & décision Go/No-Go *(~1 sem)*

Le backtest devient le **juge de paix**, sur cotes réelles, sans fuite.

- [ ] **Protocole** : walk-forward sur 8-10 ans, paris placés sur cotes d'ouverture/intermédiaires, mesure vs clôture dé-vigorisée + résultats réels, frais inclus.
- [ ] **Rapports** : RPS vs marché, log-loss, courbes de calibration, distribution du CLV (+ IC bootstrap), courbe de bankroll, drawdown, nombre de paris, hit-rate, ROI.
- [ ] **Critères Go/No-Go chiffrés (à figer avant de regarder les résultats)** :

| Critère | Seuil minimal pour passer en paper trading |
|---|---|
| CLV moyen | > 0, significatif à 95 % (bootstrap) |
| % paris « beat the close » | > 55 % |
| RPS du modèle | ≤ RPS marché + ε (proche du marché) |
| Calibration (ECE) | < 0.03 par marché principal |
| ROI walk-forward (frais inclus) | > 0 sur ≥ 500 paris |
| Drawdown max | acceptable vs bankroll (< 35 %) |

> Règle d'or : **CLV ≥ 0 significatif est NÉCESSAIRE**. Un ROI positif sans CLV positif = variance, **No-Go**.

**Definition of Done :** rapport de backtest reproductible ; décision Go/No-Go tracée.

---

### Phase 8 — Production, MLOps & paper trading *(~1-2 sem puis continu)*

- [ ] **Orchestration** : pipeline planifié (cron/Prefect) — fetch cotes + lineups + prédictions → scan → alertes. Lineups confirmés 1 h avant (bzzoiro) = signal fort sur amicaux (rotations).
- [ ] **Paper trading ≥ 4-6 semaines** : enregistrer les paris « comme si », puis **mesurer le CLV réel en live**. C'est le dernier filtre avant l'argent réel.
- [ ] **Monitoring & alerting** : dérive de calibration, taux de couverture des cotes, erreurs API, **CLV roulant**. Alerte si le CLV live devient négatif → arrêt automatique.
- [ ] **Suivi ROI réel** : `bets_log.csv` enrichi (edge, cote prise, cote clôture, CLV, résultat, P&L), tableau de bord.
- [ ] **Garde-fous produit** (conformes au CLAUDE.md) : pas de placement automatique, alertes Telegram uniquement, caps Kelly respectés, backtest validé obligatoire.

**Definition of Done :** système tourne en autonomie ; CLV live positif sur la période de paper trading avant tout euro réel.

---

## 4. Refactors concrets dans ce dépôt

Cartographie des changements par fichier :

```
betting-bot/
├── data/
│   ├── SCHEMA.md                  # NOUVEAU : contrat de données canonique
│   └── raw/ processed/            # cotes clôture + dataset élargi
├── odds/
│   └── devig.py                   # NOUVEAU : Shin / power / multiplicatif
├── models/
│   ├── ratings.py                 # NOUVEAU : Elo / pi-ratings (baseline)
│   ├── dixon_coles.py             # FIX : identifiabilité, neutre, decay, vectorisation
│   ├── bivariate_poisson.py       # NOUVEAU : Karlis-Ntzoufras
│   ├── hierarchical_bayes.py      # NOUVEAU : PyMC, pooling confédération, dynamique
│   ├── xg_model.py                # NOUVEAU : modèle basé xG
│   ├── ensemble.py                # NOUVEAU : stacking appris (remplace blend 60/40)
│   └── calibration.py             # ÉTENDU : isotonic + beta + ECE par marché
├── evaluation/
│   ├── metrics.py                 # NOUVEAU : RPS, log-loss, Brier, ECE
│   ├── clv.py                     # NOUVEAU : closing line value
│   └── walkforward.py             # NOUVEAU : backtest rolling-origin
├── pipeline/
│   ├── fetch_data.py              # ÉTENDU : qualifs, Nations League, Euro/Copa/CAN, cotes historiques
│   ├── features.py                # FIX : neutre, repos, voyage, altitude estimée, valeur effectif
│   └── value_detector.py          # FIX : edge no-vig, Kelly+incertitude, Kelly corrélé, caps
├── backtest/
│   └── backtest_wc2022.py         # RÉÉCRIT : cotes réelles, walk-forward, critères Go/No-Go
├── scan_now.py                    # FIX : filtre placeholders propre (#9)
└── tests/                         # NOUVEAU : pytest sur edge, devig, kelly, parsing
```

---

## 5. Definition of Done globale (chiffrée)

Le système est « prêt pour argent réel » **uniquement si tout est vrai** :

1. Pipeline reproductible, secrets hors repo, tests verts en CI.
2. Dataset ≥ 15-20k matchs de sélections + cotes de clôture réelles.
3. Backtest walk-forward sans fuite, sur cotes réelles.
4. **CLV moyen > 0 significatif (95 %)** et **> 55 % de paris battent la clôture**.
5. RPS proche du marché, ECE < 0.03 sur marchés principaux.
6. ROI walk-forward > 0 frais inclus sur ≥ 500 paris, drawdown maîtrisé.
7. **Paper trading live ≥ 4-6 semaines avec CLV positif confirmé.**

Tant que (4) et (7) ne sont pas atteints : **mode observation uniquement**.

---

## 6. Anti-patterns à proscrire

- ❌ Backtester contre des cotes dérivées du modèle (le défaut actuel #1).
- ❌ Juger l'edge sur le ROI d'un petit échantillon sans regarder le CLV.
- ❌ K-fold aléatoire / toute fuite temporelle.
- ❌ Comparer une proba modèle à une cote **avec vig**.
- ❌ Full Kelly, ou Kelly sur edge incertain non shrinké.
- ❌ Poids d'ensemble arbitraires figés à la main.
- ❌ Traiter une équipe inconnue comme « moyenne » (alpha=0) au lieu de la régulariser.
- ❌ Faire confiance à un modèle qui ne bat même pas Elo.
- ❌ Placer de l'argent réel sans paper trading préalable.

---

## 7. Séquencement & estimation

| Phase | Contenu | Durée | Dépend de |
|---|---|---|---|
| 0 | Fondations, secrets, tests, schéma | 2-3 j | — |
| 1 | Données élargies + cotes clôture | 3-5 j | 0 |
| 2 | Métriques (RPS/CLV) + walk-forward | 2-3 j | 1 |
| 3 | Dé-vigorisation (Shin) | 1 j | 2 |
| 4 | Modèles (Elo → DC → bivariate → bayésien → xG) | 2-3 sem | 1,2 |
| 5 | Calibration + stacking appris | 1 sem | 4 |
| 6 | Edge no-vig + sizing + risque | 1 sem | 3,5 |
| 7 | Backtest rigoureux + Go/No-Go | 1 sem | 6 |
| 8 | Prod + paper trading | continu | 7 |

> Note CDM 2026 : la compétition est imminente. Réaliste = viser **calibration honnête + observation/paper trading** pendant cette CDM, et **argent réel seulement après validation CLV**. Mieux vaut un système prudent et prouvé qu'un système rapide et illusoire.

---

## 8. Références (méthodes éprouvées)

- **Dixon & Coles (1997)** — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*, JRSS-C. (modèle de base + correction scores serrés)
- **Maher (1982)** — *Modelling association football scores*, Statistica Neerlandica. (Poisson, attaque/défense)
- **Karlis & Ntzoufras (2003)** — *Analysis of sports data using bivariate Poisson models*, JRSS-D. (corrélation des buts)
- **Rue & Salvesen (2000)** — *Prediction and retrospective analysis of soccer matches*, JRSS-D. (modèle dynamique bayésien)
- **Baio & Blangiardo (2010)** — *Bayesian hierarchical model for football results*. (pooling partiel)
- **Constantinou, Fenton & Neil (2013)** — *pi-football / pi-ratings*, Knowledge-Based Systems.
- **Constantinou & Fenton (2012)** — *Solving the problem of inadequate scoring rules for forecasting football*, J. Quant. Anal. Sports. (justifie le RPS)
- **Hvattum & Arntzen (2010)** — *Using ELO ratings for match result prediction in association football*, Int. J. Forecasting.
- **Wheatcroft (2019/2020)** — métriques de scoring et profitabilité football.
- **Shin (1992/1993)** — retrait du vig / modèle d'insiders pour cotes.
- **Kelly (1956)** — *A New Interpretation of Information Rate*. (critère de mise) + littérature Kelly fractionné / sous incertitude.
- **Efficience des marchés de paris / Closing Line Value** — pratique « sharp » établie (Pinnacle et al.) : la clôture comme meilleur estimateur, le CLV comme preuve d'edge.

---

*Ce plan est volontairement exigeant : il privilégie la preuve d'edge (CLV) sur l'illusion de performance (ROI court terme), et la régularisation/incertitude sur les point-estimates. C'est la différence entre un projet qui « affiche des value bets » et un système qui en a réellement.*








