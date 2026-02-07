"""
Configuration Module
====================

Central configuration for the Supertrend TSL Strategy + Mudrex integration.
Optimized for Railway: Only MUDREX_API_SECRET is required from environment.
All strategy parameters are hardcoded as per requirements.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StrategyConfig:
    """Hardcoded Strategy Parameters."""

    # Supertrend
    atr_period: int = 10
    factor: float = 3.0

    # Risk (same ATR period for Supertrend and risk)
    risk_atr_len: int = 10  # Same as atr_period
    tsl_atr_len: int = 10   # Same as atr_period
    risk_atr_mult: float = 2.0
    tsl_mult: float = 2.0
    tp_rr: float = 2.0  # Risk:Reward 1:2

    # Position sizing: 2% margin, leverage 5x-10x
    margin_pct: float = 0.02
    leverage_min: int = 5
    leverage_max: int = 10
    leverage: int = 5

    # Exits
    max_bars_in_trade: int = 24

    # Volatility filter (optional)
    volatility_filter_enabled: bool = False
    volatility_median_window: int = 20


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    
    # Set to empty list to automatically fetch ALL tradable assets from Mudrex
    symbols: List[str] = field(default_factory=list)
    
    # Default leverage (clamped to leverage_min/max at execution)
    leverage: str = "5"
    leverage_min: int = 5
    leverage_max: int = 10

    # Margin per entry as percent of balance (1-100). Default 2%
    # Set via env MARGIN_PERCENT (e.g. 2 for 2%, 5 for 5%)
    margin_percent: int = 2

    # Margin type
    margin_type: str = "ISOLATED"
    
    # Timeframe - Bybit interval: '1', '5', '15', '60', 'D'. Set via env TIMEFRAME. Default 15m.
    timeframe: str = "15"
    
    # Number of candles to fetch/maintain
    lookback_periods: int = 200
    
    # Minimum balance required to trade (USDT)
    min_balance: float = 10.0
    
    # Maximum concurrent positions (no cap = 999)
    max_positions: int = 999

    # Dry run mode
    dry_run: bool = False


@dataclass
class TelegramConfig:
    """Telegram notifications configuration. Supports multiple chat IDs (comma-separated)."""

    bot_token: str = ""
    chat_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        raw = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        chat_ids = [c.strip() for c in raw.split(",") if c.strip()]
        return cls(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            chat_ids=chat_ids,
        )


@dataclass
class MudrexConfig:
    """Mudrex API configuration."""
    
    # API Secret (Loaded from environment)
    api_secret: str = ""
    
    # Default settings
    base_url: Optional[str] = None
    timeout: int = 30
    rate_limit: bool = True
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> "MudrexConfig":
        """Load API secret from environment."""
        return cls(
            api_secret=os.getenv("MUDREX_API_SECRET", ""),
        )
    
    def validate(self) -> bool:
        if not self.api_secret:
            raise ValueError("MUDREX_API_SECRET environment variable is missing")
        return True


@dataclass
class Config:
    """Main configuration container."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mudrex: MudrexConfig = field(default_factory=MudrexConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig.from_env)

    @classmethod
    def from_env(cls) -> "Config":
        """Load minimal config from environment."""
        dry_run = os.getenv("TRADING_DRY_RUN", "false").lower() == "true"
        raw = os.getenv("MARGIN_PERCENT", "2").strip()
        try:
            margin_percent = max(1, min(100, int(raw)))
        except ValueError:
            margin_percent = 2

        config = cls(
            mudrex=MudrexConfig.from_env(),
            telegram=TelegramConfig.from_env(),
        )
        config.trading.dry_run = dry_run
        config.trading.margin_percent = margin_percent
        config.trading.leverage_min = config.strategy.leverage_min
        config.trading.leverage_max = config.strategy.leverage_max
        tf = os.getenv("TIMEFRAME", "").strip()
        if tf:
            config.trading.timeframe = tf
        config.strategy.margin_pct = margin_percent / 100.0
        return config
    
    def validate(self) -> bool:
        self.mudrex.validate()
        return True
    
    def to_dict(self) -> dict:
        return {
            "strategy": {
                "atr_period": self.strategy.atr_period,
                "factor": self.strategy.factor,
                "tp_rr": self.strategy.tp_rr,
                "margin_pct": self.strategy.margin_pct,
                "leverage_min": self.strategy.leverage_min,
                "leverage_max": self.strategy.leverage_max,
            },
            "trading": {
                "leverage": self.trading.leverage,
                "margin_percent": self.trading.margin_percent,
                "dry_run": self.trading.dry_run,
                "all_symbols": True if not self.trading.symbols else False
            }
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
