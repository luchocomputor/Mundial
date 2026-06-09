"""
Bot Telegram pour les alertes value bets CDM 2026.
Envoie une alerte immédiate à chaque value bet détectée.
Envoie un récapitulatif quotidien des matchs à 9h00.
"""

import asyncio
import logging
from datetime import datetime, time
from pathlib import Path

import yaml
from telegram import Bot
from telegram.error import TelegramError

from pipeline.fetch_data import fetch_upcoming_fixtures, fetch_odds_batch
from pipeline.fetch_predictions import fetch_international_predictions
from pipeline.value_detector import compute_roi_from_log, format_alert, scan_value_bets, log_bets
from models.dixon_coles import DixonColesModel


ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


async def send_message(bot: Bot, chat_id: str, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except TelegramError as e:
        logger.error(f"Erreur Telegram: {e}")


async def send_value_bets(bets: list[dict], cfg: dict) -> None:
    if not bets:
        return
    bot = Bot(token=cfg["telegram"]["token"])
    chat_id = cfg["telegram"]["chat_id"]
    for bet in bets:
        text = format_alert(bet)
        await send_message(bot, chat_id, text)
        await asyncio.sleep(0.5)


async def send_daily_summary(cfg: dict, model: DixonColesModel) -> None:
    """Résumé matinal : matchs du jour + ROI courant."""
    today = datetime.now().date()
    bot = Bot(token=cfg["telegram"]["token"])
    chat_id = cfg["telegram"]["chat_id"]

    try:
        df = fetch_upcoming_fixtures(cfg=cfg)
    except Exception as e:
        logger.error(f"Erreur fetch fixtures: {e}")
        df = None

    try:
        preds_df = fetch_international_predictions(cfg=cfg)
        bzzo_map = {
            row["event_id"]: row.to_dict()
            for _, row in preds_df.iterrows()
        }
    except Exception:
        bzzo_map = {}

    lines = [f"☀️ <b>Récap CDM 2026 — {today.strftime('%d %B %Y')}</b>\n"]

    if df is not None and not df.empty:
        today_matches = df[df["date"].dt.date == today]
        if today_matches.empty:
            lines.append("Aucun match aujourd'hui.")
        else:
            lines.append(f"<b>{len(today_matches)} match(s) aujourd'hui :</b>")
            for _, row in today_matches.iterrows():
                home = row["home_team"]
                away = row["away_team"]
                kickoff = row["date"].strftime("%H:%M")
                fid = row["fixture_id"]
                try:
                    from pipeline.fetch_predictions import blend_predictions
                    dc = model.predict_outcomes(home, away)
                    bz = bzzo_map.get(fid)
                    preds = blend_predictions(dc, bz)
                    lines.append(
                        f"⚽ {home} vs {away} — {kickoff}\n"
                        f"   Win: {preds['home_win']*100:.0f}% / Draw: {preds['draw']*100:.0f}% / {preds['away_win']*100:.0f}%\n"
                        f"   Over 2.5: {preds['over_2.5']*100:.0f}% | BTTS: {preds['btts']*100:.0f}%"
                    )
                except Exception:
                    lines.append(f"⚽ {home} vs {away} — {kickoff} (prédiction indispo)")
    else:
        lines.append("Données matchs non disponibles.")

    roi_stats = compute_roi_from_log()
    if roi_stats["n_bets"] > 0:
        lines.append(
            f"\n📊 <b>ROI suivi</b> : {roi_stats['roi']:+.1f}% sur {roi_stats['n_bets']} bets\n"
            f"   Profit : {roi_stats['profit']:+.2f}€ (misé : {roi_stats['total_staked']:.2f}€)"
        )

    text = "\n".join(lines)
    await send_message(bot, chat_id, text)
    logger.info("Récapitulatif quotidien envoyé.")


async def run_scanner_loop(cfg: dict, model: DixonColesModel, interval_minutes: int = 30):
    """Boucle principale : scanne les value bets toutes les N minutes."""
    bankroll = cfg["bankroll"]["initial"]
    logger.info(f"Scanner démarré. Vérification toutes les {interval_minutes} min.")

    while True:
        now = datetime.now()

        if now.hour == 9 and now.minute < interval_minutes:
            logger.info("Envoi du récapitulatif quotidien...")
            await send_daily_summary(cfg, model)

        try:
            df = fetch_upcoming_fixtures(cfg=cfg)
            if not df.empty:
                matches = df.to_dict("records")
                logger.info(f"{len(matches)} matchs à venir (WC+amicaux)")
            else:
                matches = []
        except Exception as e:
            logger.error(f"Erreur fetch: {e}")
            matches = []

        if matches:
            event_ids = [m["fixture_id"] for m in matches[:20]]
            bookmaker_odds = fetch_odds_batch(event_ids, cfg)
            has_odds = sum(1 for o in bookmaker_odds.values() if o)
            logger.info(f"Cotes disponibles pour {has_odds}/{len(event_ids)} matchs")

            if has_odds > 0:
                try:
                    preds_df = fetch_international_predictions(cfg=cfg)
                    bzzo_map = {row["event_id"]: row.to_dict() for _, row in preds_df.iterrows()}
                except Exception:
                    bzzo_map = {}

                bk = cfg["bankroll"]["initial"]
                bets = scan_value_bets(matches[:20], model, bookmaker_odds, cfg, bk, bzzo_map)
                if bets:
                    logger.info(f"{len(bets)} value bets détectés!")
                    log_bets(bets)
                    await send_value_bets(bets, cfg)
                else:
                    logger.info("Aucun value bet pour l'instant.")

        await asyncio.sleep(interval_minutes * 60)


def main():
    cfg = load_config()

    try:
        model = DixonColesModel.load()
        logger.info("Modèle chargé.")
    except FileNotFoundError:
        logger.error("Modèle non trouvé. Lance d'abord : python models/dixon_coles.py --train")
        return

    asyncio.run(run_scanner_loop(cfg, model))


if __name__ == "__main__":
    main()
