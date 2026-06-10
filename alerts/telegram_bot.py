"""
Bot Telegram pour alertes value bets + récap quotidien 9h.
En mode paper_only : récap sans incitation à parier, pas d'alertes value bets.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from telegram import Bot
from telegram.error import TelegramError

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import load_config
from pipeline.fetch_data import fetch_odds_batch, fetch_upcoming_fixtures
from pipeline.fetch_predictions import fetch_international_predictions
from pipeline.model_loader import load_production_model
from pipeline.production import alerts_allowed, load_latest_go_nogo
from pipeline.value_detector import compute_roi_from_log, format_alert, log_bets, scan_value_bets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def send_message(bot: Bot, chat_id: str, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TelegramError as e:
        logger.error(f"Erreur Telegram: {e}")


async def send_value_bets(bets: list[dict], cfg) -> None:
    if not bets or not cfg.telegram_token:
        return
    if not alerts_allowed(cfg):
        logger.info("Alertes value bets désactivées (paper_only ou NO_GO)")
        return
    bot = Bot(token=cfg.telegram_token)
    for bet in bets:
        await send_message(bot, cfg.telegram_chat_id, format_alert(bet))
        await asyncio.sleep(0.5)


async def send_daily_summary(cfg, model) -> None:
    today = datetime.now().date()
    if not cfg.telegram_token:
        return
    bot = Bot(token=cfg.telegram_token)

    go = load_latest_go_nogo()
    mode = cfg.model.production_mode
    header = f"Récap CDM 2026 — {today.strftime('%d %B %Y')}\n"
    if mode == "paper_only":
        header += "📋 Mode observation (paper only) — pas de mise recommandée\n"
    header += f"Validation : {go.value}\n\n"

    try:
        df = fetch_upcoming_fixtures()
    except Exception as e:
        logger.error(f"Erreur fetch: {e}")
        df = None

    bzzo_map = {}
    try:
        preds_df = fetch_international_predictions()
        bzzo_map = {row["event_id"]: row.to_dict() for _, row in preds_df.iterrows()}
    except Exception:
        pass

    lines = [header]

    if df is not None and not df.empty:
        today_matches = df[df["date"].dt.date == today]
        if today_matches.empty:
            lines.append("Aucun match aujourd'hui.")
        else:
            lines.append(f"{len(today_matches)} match(s) aujourd'hui :")
            for _, row in today_matches.iterrows():
                home, away = row["home_team"], row["away_team"]
                kickoff = row["date"].strftime("%H:%M")
                try:
                    from models.base import MatchContext
                    ctx = MatchContext(is_neutral=bool(row.get("is_neutral", False)))
                    bz = bzzo_map.get(row["fixture_id"])
                    pred = model.predict_outcomes(home, away, context=ctx)
                    if bz and hasattr(model, "predict_outcomes"):
                        try:
                            pred = model.predict_outcomes(home, away, context=ctx, bzzoiro_features=bz)
                        except TypeError:
                            pass
                    d = pred.to_dict() if hasattr(pred, "to_dict") else pred
                    lines.append(
                        f"{home} vs {away} — {kickoff}\n"
                        f"  Win: {d['home_win']*100:.0f}% / Draw: {d['draw']*100:.0f}% / {d['away_win']*100:.0f}%\n"
                        f"  Over 2.5: {d['over_2.5']*100:.0f}% | BTTS: {d['btts']*100:.0f}%"
                    )
                except Exception:
                    lines.append(f"{home} vs {away} — {kickoff}")
    else:
        lines.append("Données non disponibles.")

    roi_stats = compute_roi_from_log()
    if roi_stats["n_bets"] > 0:
        lines.append(
            f"\nROI paper: {roi_stats['roi']:+.1f}% ({roi_stats['n_bets']} bets) | "
            f"CLV moy: {roi_stats.get('clv_mean', 0):+.1f}%"
        )
        if roi_stats.get("clv_mean", 0) < 0:
            lines.append("⚠ CLV 7j glissant négatif — prudence")

    await send_message(bot, cfg.telegram_chat_id, "\n".join(lines))


async def run_scanner_loop(cfg, model, interval_minutes: int = 30):
    logger.info(f"Scanner démarré (intervalle {interval_minutes} min, mode={cfg.model.production_mode})")

    while True:
        now = datetime.now()
        if now.hour == 9 and now.minute < interval_minutes:
            await send_daily_summary(cfg, model)

        try:
            df = fetch_upcoming_fixtures()
            matches = df.to_dict("records") if not df.empty else []
        except Exception as e:
            logger.error(f"Erreur fetch: {e}")
            matches = []

        if matches and alerts_allowed(cfg):
            event_ids = [m["fixture_id"] for m in matches[:20]]
            bookmaker_odds = fetch_odds_batch(event_ids)
            has_odds = sum(1 for o in bookmaker_odds.values() if o)

            if has_odds > 0:
                try:
                    preds_df = fetch_international_predictions()
                    bzzo_map = {row["event_id"]: row.to_dict() for _, row in preds_df.iterrows()}
                except Exception:
                    bzzo_map = {}

                bets, stats = scan_value_bets(
                    matches[:20], model, bookmaker_odds, cfg, cfg.bankroll_initial, bzzo_map
                )
                logger.info(f"Garde-fous: {stats.accepted} acceptés, {stats.rejected_divergence} rejetés")
                if bets:
                    log_bets(bets)
                    await send_value_bets(bets, cfg)
        elif matches:
            logger.debug("Scan sans alertes (paper_only ou NO_GO)")

        await asyncio.sleep(interval_minutes * 60)


def main():
    cfg = load_config()
    model = load_production_model(prefer_elo=True)
    asyncio.run(run_scanner_loop(cfg, model))


if __name__ == "__main__":
    main()
