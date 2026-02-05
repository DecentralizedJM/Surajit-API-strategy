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
    
    # Supertrend parameters
    atr_period: int = 10
    factor: float = 3.0
    
    # Risk ATR parameters
    risk_atr_len: int = 14
    risk_atr_mult: float = 2.0
    
    # Trailing Stop Loss parameters
    tsl_atr_len: int = 14
    tsl_mult: float = 2.0
    
    # Take Profit (Risk:Reward ratio 1:2)
    tp_rr: float = 2.0
    
    # Position sizing (2% of futures wallet balance as margin)
    position_size_pct: float = 0.02


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    
    # Set to empty list to automatically fetch ALL tradable assets from Mudrex
    symbols: List[str] = field(default_factory=list)
    
    # Default leverage
    leverage: str = "5"
    
    # Margin type
    margin_type: str = "ISOLATED"
    
    # Timeframe
    timeframe: str = "5m"
    
    # Number of candles to fetch
    lookback_periods: int = 100
    
    # Minimum balance required to trade (USDT)
    min_balance: float = 10.0
    
    # Maximum concurrent positions (Increased for 'All Symbols' mode)
    max_positions: int = 50
    
    # Dry run mode
    dry_run: bool = False


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
class CCXTConfig:
    """CCXT configuration for price data."""
    exchange: str = "binance"
    sandbox: bool = False


@dataclass
class Config:
    """Main configuration container."""
    
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mudrex: MudrexConfig = field(default_factory=MudrexConfig)
    ccxt: CCXTConfig = field(default_factory=CCXTConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load minimal config from environment."""
        # Check dry run override from env if needed for safety during setup
        dry_run = os.getenv("TRADING_DRY_RUN", "false").lower() == "true"
        
        config = cls(
            mudrex=MudrexConfig.from_env(),
        )
        config.trading.dry_run = dry_run
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
                "position_size_pct": self.strategy.position_size_pct,
            },
            "trading": {
                "leverage": self.trading.leverage,
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
