"""Construction du dataset master unifié."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from pipeline.features import add_confederations, add_rest_features
from pipeline.schemas import validate_matches_df

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def load_confederations() -> dict[str, str]:
    path = ROOT / "data" / "confederations.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("teams", {})


def build_master_dataset(
    matches_path: Path | None = None,
    odds_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mpath = matches_path or DATA_RAW / "all_matches.parquet"
    opath = odds_path or DATA_RAW / "odds_history.parquet"

    if not mpath.exists():
        raise FileNotFoundError(f"Matchs manquants : {mpath}. Lance fetch_data d'abord.")

    matches = pd.read_parquet(mpath)
    matches = validate_matches_df(matches)
    matches = add_confederations(matches, load_confederations())
    matches = add_rest_features(matches)

    odds = pd.DataFrame()
    if opath.exists():
        odds = pd.read_parquet(opath)

    matches.to_parquet(DATA_RAW / "all_matches.parquet", index=False)

    report = {
        "n_matches": len(matches),
        "n_finished": int((matches["status"] == "finished").sum()),
        "n_upcoming": int((matches["status"] == "upcoming").sum()),
        "n_teams": len(
            set(matches["home_team"].unique()) | set(matches["away_team"].unique())
        ),
        "n_odds_rows": len(odds),
        "odds_coverage_pct": round(
            len(odds["fixture_id"].unique()) / max(len(matches), 1) * 100, 1
        )
        if not odds.empty
        else 0,
        "date_range": [
            str(matches["date"].min()) if not matches.empty else None,
            str(matches["date"].max()) if not matches.empty else None,
        ],
    }
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    (DATA_PROCESSED / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(f"Dataset : {report}")

    return matches, odds


if __name__ == "__main__":
    build_master_dataset()
