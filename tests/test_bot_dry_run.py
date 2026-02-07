"""Integration test: run bot one cycle in dry-run with mock data and mocked Mudrex."""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from config import Config, TradingConfig, MudrexConfig, StrategyConfig as MainStrategyConfig
from data_manager import DataManager
from supertrend_mudrex_bot import SupertrendMudrexBot, BotExecutionResult
from strategy_core import StrategyConfig


def test_bot_run_once_dry_run():
    """Run one cycle with dry_run=True, 2 symbols, pre-filled data; no real API/WS calls."""
    logging.basicConfig(level=logging.WARNING)
    mudrex_config = MudrexConfig(api_secret="test_secret_dry_run")
    trading_config = TradingConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        dry_run=True,
        timeframe="15",
        margin_percent=2,
        max_positions=10,
    )
    config = Config(
        mudrex=mudrex_config,
        trading=trading_config,
        strategy=MainStrategyConfig(),
    )
    dm = DataManager(interval="15", lookback=200)
    n = 50
    for sym in ["BTCUSDT", "ETHUSDT"]:
        close = 40000 + np.cumsum(np.random.randn(n) * 10) if sym == "BTCUSDT" else 2000 + np.cumsum(np.random.randn(n))
        high = close + np.abs(np.random.randn(n)) * 5
        low = close - np.abs(np.random.randn(n)) * 5
        df = pd.DataFrame({
            "open": close - 1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000,
        })
        df.index = pd.date_range("2025-01-01", periods=n, freq="15min")
        dm.data[sym] = df
        dm.active_symbols.add(sym)
    bot = SupertrendMudrexBot(config=config, data_manager=dm)
    bot.adapter.get_balance = MagicMock(return_value=1000.0)
    bot.adapter.get_open_positions = MagicMock(return_value={})
    bot.adapter._positions = {}
    bot.adapter.has_position = MagicMock(return_value=False)
    bot.adapter.get_asset_info = MagicMock(return_value={"min_quantity": 0.001, "quantity_step": 0.001})
    bot.adapter.execute_proposed_position = MagicMock(
        return_value=MagicMock(success=True, action="NONE", symbol="X", message="", error=None, position_state=None)
    )
    with patch.object(dm, "subscribe", return_value=None):
        result = bot.run_once()
    assert isinstance(result, BotExecutionResult)
    assert result.symbols_processed == 2
    assert result.timestamp is not None
    assert result.errors is not None
