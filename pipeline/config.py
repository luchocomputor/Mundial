"""Configuration centralisée : seuils YAML + secrets depuis .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class ModelConfig:
    min_edge_threshold: float = 0.05
    kelly_fraction: float = 0.25
    max_kelly_bet: float = 0.05
    decay: float = 0.0065
    friendly_weight: float = 0.5
    devig_method: str = "shin"
    max_market_divergence: float = 0.15
    market_anchor_weight: float = 0.3
    sparse_match_threshold: int = 5
    sparse_anchor_weight: float = 0.5
    production_mode: str = "paper_only"


@dataclass
class RiskConfig:
    max_daily_exposure: float = 0.15
    max_drawdown: float = 0.35


@dataclass
class AppConfig:
    bzzoiro_token: str
    bzzoiro_base_url: str = "https://sports.bzzoiro.com/api/v2"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    bankroll_initial: float = 200.0
    leagues: dict = field(default_factory=dict)
    seasons: dict = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def as_dict(self) -> dict:
        """Compatibilité avec l'ancien format dict."""
        return {
            "bzzoiro": {"token": self.bzzoiro_token, "base_url": self.bzzoiro_base_url},
            "telegram": {"token": self.telegram_token, "chat_id": self.telegram_chat_id},
            "model": {
                "min_edge_threshold": self.model.min_edge_threshold,
                "kelly_fraction": self.model.kelly_fraction,
                "max_kelly_bet": self.model.max_kelly_bet,
                "decay": self.model.decay,
                "friendly_weight": self.model.friendly_weight,
                "devig_method": self.model.devig_method,
                "max_market_divergence": self.model.max_market_divergence,
                "market_anchor_weight": self.model.market_anchor_weight,
                "sparse_match_threshold": self.model.sparse_match_threshold,
                "sparse_anchor_weight": self.model.sparse_anchor_weight,
                "production_mode": self.model.production_mode,
            },
            "bankroll": {"initial": self.bankroll_initial},
            "leagues": self.leagues,
            "seasons": self.seasons,
            "risk": {
                "max_daily_exposure": self.risk.max_daily_exposure,
                "max_drawdown": self.risk.max_drawdown,
            },
        }


def load_config(path: Path | None = None) -> AppConfig:
    load_dotenv(ROOT / ".env")
    cfg_path = path or CONFIG_PATH
    raw: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}

    token = os.environ.get("BZZOIRO_TOKEN") or raw.get("bzzoiro", {}).get("token", "")
    if not token:
        raise ValueError(
            "BZZOIRO_TOKEN manquant. Définir dans .env ou bzzoiro.token dans config.yaml"
        )

    model_raw = raw.get("model", {})
    risk_raw = raw.get("risk", {})

    return AppConfig(
        bzzoiro_token=token,
        bzzoiro_base_url=raw.get("bzzoiro", {}).get(
            "base_url", "https://sports.bzzoiro.com/api/v2"
        ),
        telegram_token=os.environ.get("TELEGRAM_TOKEN")
        or raw.get("telegram", {}).get("token", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID")
        or raw.get("telegram", {}).get("chat_id", ""),
        model=ModelConfig(
            min_edge_threshold=model_raw.get("min_edge_threshold", 0.05),
            kelly_fraction=model_raw.get("kelly_fraction", 0.25),
            max_kelly_bet=model_raw.get("max_kelly_bet", 0.05),
            decay=model_raw.get("decay", 0.0065),
            friendly_weight=model_raw.get("friendly_weight", 0.5),
            devig_method=model_raw.get("devig_method", "shin"),
            max_market_divergence=model_raw.get("max_market_divergence", 0.15),
            market_anchor_weight=model_raw.get("market_anchor_weight", 0.3),
            sparse_match_threshold=model_raw.get("sparse_match_threshold", 5),
            sparse_anchor_weight=model_raw.get("sparse_anchor_weight", 0.5),
            production_mode=model_raw.get("production_mode", "paper_only"),
        ),
        bankroll_initial=raw.get("bankroll", {}).get("initial", 200),
        leagues=raw.get("leagues", {}),
        seasons=raw.get("seasons", {}),
        risk=RiskConfig(
            max_daily_exposure=risk_raw.get("max_daily_exposure", 0.15),
            max_drawdown=risk_raw.get("max_drawdown", 0.35),
        ),
    )


def bzzoiro_headers(cfg: AppConfig | None = None) -> dict:
    if cfg is None:
        cfg = load_config()
    return {"Authorization": f"Token {cfg.bzzoiro_token}"}
