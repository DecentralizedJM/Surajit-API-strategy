"""Unit tests for strategy_core indicators (ATR, Supertrend)."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy_core.indicators import wilder_atr, supertrend, atr_above_median


def test_wilder_atr_length():
    n = 50
    high = np.ones(n) * 110
    low = np.ones(n) * 90
    close = np.ones(n) * 100
    atr = wilder_atr(high, low, close, period=14)
    assert len(atr) == n
    assert np.isnan(atr[0])
    assert np.isnan(atr[12])
    assert not np.isnan(atr[13])
    assert atr[13] == 20.0  # first ATR = mean of first 14 TRs (all 20)


def test_wilder_atr_positive():
    n = 30
    high = np.ones(n) * 105
    low = np.ones(n) * 95
    close = np.ones(n) * 100
    atr = wilder_atr(high, low, close, period=10)
    assert np.all(atr[9:] >= 0)
    assert np.all(np.isnan(atr[:9]) | (atr[:9] >= 0))


def test_supertrend_shape():
    n = 50
    high = 100 + np.cumsum(np.random.randn(n) * 0.5)
    low = high - 2
    close = (high + low) / 2
    atr = wilder_atr(high, low, close, period=10)
    st, direction = supertrend(high, low, close, atr, factor=3.0)
    assert len(st) == n
    assert len(direction) == n
    assert np.any(np.isnan(st) == False)
    assert np.any(np.isin(direction, [1.0, -1.0]))


def test_supertrend_direction_values():
    n = 30
    high = np.linspace(100, 110, n)
    low = high - 1
    close = high - 0.5
    atr = wilder_atr(high, low, close, period=5)
    st, direction = supertrend(high, low, close, atr, factor=2.0)
    # direction is 0 for initial bars, then 1 or -1
    valid = direction[~np.isnan(st)]
    assert np.all(np.isin(valid[valid != 0], [1.0, -1.0]))


def test_atr_above_median():
    atr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert bool(atr_above_median(atr, idx=9, window=5)) is True
    assert bool(atr_above_median(atr, idx=4, window=5)) is True
