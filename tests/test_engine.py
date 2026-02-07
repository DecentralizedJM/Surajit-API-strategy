"""Unit tests for strategy_core engine (process_candle)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from strategy_core.engine import process_candle
from strategy_core.config import StrategyConfig
from strategy_core.state import TradeState


def _make_ohlcv(n: int, trend: str = "flat") -> list:
    base = 100.0
    if trend == "up":
        close = base + np.cumsum(np.random.rand(n) * 0.5)
    elif trend == "down":
        close = base - np.cumsum(np.random.rand(n) * 0.5)
    else:
        close = base + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    return [{"open": float(close[i] - 0.1), "high": float(high[i]), "low": float(low[i]), "close": float(close[i]), "volume": 1000.0} for i in range(n)]


def test_process_candle_insufficient_data():
    config = StrategyConfig()
    ohlcv = _make_ohlcv(10)  # need at least atr_period+1
    out = process_candle(ohlcv, 10000.0, {"min_quantity": 0.001, "quantity_step": 0.001}, TradeState.flat(), config)
    assert out["signal"] == "HOLD"
    assert out["reason"] == "insufficient_data"
    assert out["proposed_position"] is None


def test_process_candle_hold_flat():
    config = StrategyConfig(atr_period=5)
    ohlcv = _make_ohlcv(50, "flat")
    out = process_candle(ohlcv, 10000.0, {"min_quantity": 0.001, "quantity_step": 0.001}, TradeState.flat(), config)
    assert out["signal"] in ("HOLD", "LONG", "SHORT")
    if out["signal"] == "HOLD":
        assert out["proposed_position"] is None


def test_process_candle_proposed_position_shape():
    config = StrategyConfig(atr_period=5)
    np.random.seed(42)
    ohlcv = _make_ohlcv(100)
    out = process_candle(ohlcv, 10000.0, {"min_quantity": 0.001, "quantity_step": 0.001}, TradeState.flat(), config)
    if out.get("proposed_position"):
        pp = out["proposed_position"]
        assert "side" in pp and "quantity" in pp and "entry_price" in pp
        assert "stop_loss" in pp and "take_profit" in pp
        assert pp["side"] in ("LONG", "SHORT")
        assert pp["quantity"] >= 0
        assert pp["entry_price"] > 0
