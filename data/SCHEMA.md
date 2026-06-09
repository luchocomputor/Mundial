# Schéma de données — Mundial

## Table `matches` (`data/raw/all_matches.parquet`)

| Colonne | Type | Règle |
|---------|------|-------|
| `fixture_id` | int | unique, non null |
| `date` | datetime UTC | non null |
| `status` | enum | `finished` \| `upcoming` \| `cancelled` |
| `home_team`, `away_team` | str | non null, pas placeholder |
| `home_goals`, `away_goals` | int? | null si `upcoming` |
| `league_id`, `league_name` | int/str | traçabilité source |
| `competition_type` | enum | `wc` \| `qualifier` \| `nations_league` \| `continental` \| `friendly` \| `other` |
| `is_friendly`, `is_neutral` | bool | |
| `venue_id`, `venue`, `city` | | |
| `home_confederation`, `away_confederation` | str | depuis `data/confederations.yaml` |
| `days_since_last_match_home/away` | float | calculé sans fuite |
| `matches_last_14d_home/away` | int | |
| `xg_home`, `xg_away` | float? | si dispo |

### Mapping status API bzzoiro

- `finished`, `FT`, `complete` → `finished`
- `notstarted`, `scheduled`, `NS` → `upcoming`
- `cancelled`, `postponed`, `abandoned` → `cancelled`

## Table `odds_history` (`data/raw/odds_history.parquet`)

| Colonne | Type | Règle |
|---------|------|-------|
| `fixture_id` | int | FK matches |
| `bookmaker` | str | ex. `pinnacle` |
| `snapshot_type` | enum | `open` \| `mid` \| `close` |
| `captured_at` | datetime | |
| `market` | str | `1X2`, `over_2.5`, `btts` |
| `side` | str | `home`/`draw`/`away`/`over`/`under`/`yes`/`no` |
| `odds_decimal` | float | > 1.0 |
