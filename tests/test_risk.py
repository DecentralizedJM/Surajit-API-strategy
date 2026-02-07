"""Unit tests for strategy_core risk (position sizing, leverage)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy_core.risk import position_size, compute_leverage


def test_compute_leverage_clamp():
    assert compute_leverage(5, 1, 10) == 5
    assert compute_leverage(0, 5, 10) == 5
    assert compute_leverage(15, 5, 10) == 10
    assert compute_leverage(7, 5, 10) == 7


def test_position_size_basic():
    qty, lev = position_size(
        balance=10000.0,
        margin_pct=0.02,
        entry_price=50000.0,
        leverage=5,
        min_qty=0.001,
        quantity_step=0.001,
    )
    # margin = 200, notional = 1000, qty = 1000/50000 = 0.02
    assert qty >= 0.02 - 0.001
    assert qty >= 0.001
    assert lev == 5


def test_position_size_zero_balance():
    qty, lev = position_size(0, 0.02, 100.0, 5, 0.001, 0.001)
    assert qty == 0.0
    assert lev == 5


def test_position_size_below_min_qty():
    qty, lev = position_size(
        balance=100.0,
        margin_pct=0.01,
        entry_price=1_000_000.0,
        leverage=5,
        min_qty=0.1,
        quantity_step=0.01,
    )
    # very small qty, below min_qty
    assert qty == 0.0
    assert lev == 5


def test_position_size_quantity_step_rounding():
    qty, _ = position_size(
        balance=10000.0,
        margin_pct=0.02,
        entry_price=100.0,
        leverage=5,
        min_qty=0.1,
        quantity_step=0.1,
    )
    # margin 200, notional 1000, raw qty 10; should be multiple of 0.1
    assert qty >= 0.1
    assert abs(qty - round(qty * 10) / 10) < 1e-9
