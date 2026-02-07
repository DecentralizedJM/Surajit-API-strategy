"""
Telegram Notifier
=================

Sends bot notifications to Telegram via Bot API.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Send notifications to Telegram. No-op if token or chat_id not configured."""

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()
        self.enabled = bool(self.bot_token and self.chat_id)

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram. Returns True if sent, False otherwise."""
        if not self.enabled:
            return False
        try:
            url = TELEGRAM_API.format(token=self.bot_token)
            r = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4096],  # Telegram limit
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if not r.ok:
                logger.warning(f"Telegram send failed: {r.status_code} {r.text[:200]}")
                return False
            return True
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")
            return False

    def notify_open(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: int,
        dry_run: bool = False,
    ) -> bool:
        prefix = "[DRY] " if dry_run else ""
        text = (
            f"{prefix}🟢 <b>OPEN {side}</b> {symbol}\n"
            f"Qty: {quantity} | Entry: ${entry_price:.4f}\n"
            f"SL: ${stop_loss:.4f} | TP: ${take_profit:.4f} | Lev: {leverage}x"
        )
        return self.send(text)

    def notify_close(
        self,
        symbol: str,
        side: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        dry_run: bool = False,
    ) -> bool:
        if side == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        notional = entry_price * quantity
        pnl_pct = (pnl / notional) * 100 if notional else 0
        sign = "+" if pnl >= 0 else ""
        prefix = "[DRY] " if dry_run else ""
        text = (
            f"{prefix}🔴 <b>CLOSE {side}</b> {symbol}\n"
            f"Reason: {reason} | Entry: ${entry_price:.4f} | Exit: ${exit_price:.4f}\n"
            f"PnL: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)"
        )
        return self.send(text)

    def notify_cycle(
        self,
        balance: float,
        positions_count: int,
        signals: int,
        opened: int,
        closed: int,
        tsl_updates: int,
        errors: int,
        dry_run: bool = False,
    ) -> bool:
        prefix = "[DRY] " if dry_run else ""
        text = (
            f"{prefix}📊 <b>Cycle Summary</b>\n"
            f"Balance: ${balance:.2f} | Positions: {positions_count}\n"
            f"Signals: {signals} | Opened: {opened} | Closed: {closed} | TSL: {tsl_updates}\n"
            f"Errors: {errors}"
        )
        return self.send(text)
