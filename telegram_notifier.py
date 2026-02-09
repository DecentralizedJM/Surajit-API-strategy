"""
Telegram Notifier
=================

Sends bot notifications to Telegram via Bot API.
Supports multiple chat IDs (comma-separated in TELEGRAM_CHAT_ID).
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

import logging
from typing import List, Union

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Send notifications to Telegram. No-op if token or chat_ids not configured."""

    def __init__(self, bot_token: str = "", chat_ids: Union[str, List[str]] = ""):
        self.bot_token = (bot_token or "").strip()
        if isinstance(chat_ids, str):
            raw = (chat_ids or "").strip()
            self.chat_ids = [c.strip() for c in raw.split(",") if c.strip()]
        else:
            self.chat_ids = [str(c).strip() for c in (chat_ids or []) if str(c).strip()]
        self.enabled = bool(self.bot_token and self.chat_ids)

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message to all configured chat IDs. Returns True if at least one sent."""
        if not self.enabled:
            return False
        ok = False
        payload = {
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        url = TELEGRAM_API.format(token=self.bot_token)
        for chat_id in self.chat_ids:
            try:
                r = requests.post(url, json={**payload, "chat_id": chat_id}, timeout=10)
                if r.ok:
                    ok = True
                else:
                    logger.warning(f"Telegram send failed for {chat_id}: {r.status_code}")
            except Exception as e:
                logger.warning(f"Telegram send error for {chat_id}: {e}")
        return ok

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

    def notify_warning(self, symbol: str, message: str) -> bool:
        """Send a warning notification (e.g. trade impossible even after scaling)."""
        text = f"⚠️ <b>WARNING</b> {symbol}\n{message}"
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
        timeframe: str = "",
        margin_percent: int = 0,
    ) -> bool:
        prefix = "[DRY] " if dry_run else ""
        header = f"{prefix}📊 <b>Cycle Summary</b>"
        if timeframe or margin_percent:
            header += f" (TF: {timeframe} | Margin: {margin_percent}%)"
        text = (
            f"{header}\n"
            f"Balance: ${balance:.2f} | Positions: {positions_count}\n"
            f"Signals: {signals} | Opened: {opened} | Closed: {closed} | TSL: {tsl_updates}\n"
            f"Errors: {errors}"
        )
        return self.send(text)
