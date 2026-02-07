"""
Backtest and walk-forward evaluation for Supertrend TSL strategy.
Uses Bybit REST klines; runs process_candle bar-by-bar and tracks simulated PnL.
Run: python scripts/backtest_supertrend.py [--symbol BTCUSDT] [--walk-forward]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pybit.unified_trading import HTTP
from strategy_core import process_candle, TradeState, StrategyConfig


def fetch_klines(symbol: str, interval: str = "15", limit: int = 500) -> list:
    """Fetch historical klines from Bybit (linear). Returns list of dicts with open, high, low, close, volume."""
    session = HTTP(testnet=False)
    r = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
    result = (r or {}).get("result", {})
    rows = result.get("list") or []
    out = []
    for row in reversed(rows):
        out.append({
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return out


def run_backtest(
    ohlcv: list,
    config: StrategyConfig,
    initial_equity: float = 10000.0,
    min_qty: float = 0.001,
    quantity_step: float = 0.001,
) -> dict:
    """
    Run strategy bar-by-bar. Track state and simulated PnL.
    Returns dict with trades, equity curve, and summary metrics.
    """
    contract_specs = {"min_quantity": min_qty, "quantity_step": quantity_step}
    state = TradeState.flat()
    equity = initial_equity
    trades = []
    equity_curve = []

    for i in range(config.atr_period + 1, len(ohlcv)):
        window = ohlcv[: i + 1]
        out = process_candle(
            ohlcv=window,
            account_equity=equity,
            contract_specs=contract_specs,
            prev_state=state,
            config=config,
        )

        if out["signal"] in ("LONG", "SHORT") and out.get("proposed_position"):
            pp = out["proposed_position"]
            state = TradeState(
                position_side=pp["side"],
                entry_price=pp["entry_price"],
                stop_loss=pp["stop_loss"],
                take_profit=pp["take_profit"],
                trailing_stop=None,
                bars_in_trade=0,
                extreme_price=pp["entry_price"],
            )
            trades.append({
                "bar": i,
                "side": pp["side"],
                "entry": pp["entry_price"],
                "sl": pp["stop_loss"],
                "tp": pp["take_profit"],
                "qty": pp["quantity"],
            })
        elif out["signal"] == "EXIT" and state.position_side != "FLAT":
            close_price = window[-1]["close"]
            if state.position_side == "LONG":
                pnl_pct = (close_price - state.entry_price) / state.entry_price
            else:
                pnl_pct = (state.entry_price - close_price) / state.entry_price
            last_trade = trades[-1]
            last_trade["exit_bar"] = i
            last_trade["exit_price"] = close_price
            last_trade["exit_reason"] = out["reason"]
            last_trade["pnl_pct"] = pnl_pct
            equity *= 1 + pnl_pct
            state = TradeState.flat()
        elif state.position_side != "FLAT":
            high, low = window[-1]["high"], window[-1]["low"]
            exit_price = None
            if state.position_side == "LONG":
                if low <= state.stop_loss:
                    exit_price = state.stop_loss
                elif state.take_profit and high >= state.take_profit:
                    exit_price = state.take_profit
            else:
                if high >= state.stop_loss:
                    exit_price = state.stop_loss
                elif state.take_profit and low <= state.take_profit:
                    exit_price = state.take_profit
            if exit_price is not None:
                if state.position_side == "LONG":
                    pnl_pct = (exit_price - state.entry_price) / state.entry_price
                else:
                    pnl_pct = (state.entry_price - exit_price) / state.entry_price
                last_trade = trades[-1]
                last_trade["exit_bar"] = i
                last_trade["exit_price"] = exit_price
                last_trade["pnl_pct"] = pnl_pct
                equity *= 1 + pnl_pct
                state = TradeState.flat()
            else:
                # Advance in-trade state for next bar (bars_in_trade, extreme)
                new_extreme = (
                    max(state.extreme_price, high) if state.position_side == "LONG"
                    else min(state.extreme_price, low)
                )
                if state.extreme_price == 0:
                    new_extreme = high if state.position_side == "LONG" else low
                state = TradeState(
                    position_side=state.position_side,
                    entry_price=state.entry_price,
                    stop_loss=state.stop_loss,
                    take_profit=state.take_profit,
                    trailing_stop=state.trailing_stop,
                    bars_in_trade=state.bars_in_trade + 1,
                    extreme_price=new_extreme,
                )

        equity_curve.append(equity)

    completed = [t for t in trades if "pnl_pct" in t]
    total_return = (equity - initial_equity) / initial_equity if initial_equity else 0
    return {
        "trades": trades,
        "completed_trades": len(completed),
        "total_return_pct": total_return * 100,
        "final_equity": equity,
        "equity_curve": equity_curve,
        "win_rate": (sum(1 for t in completed if t["pnl_pct"] > 0) / len(completed) * 100) if completed else 0,
    }


def walk_forward(symbol: str, interval: str = "15", train_bars: int = 350, test_bars: int = 150) -> None:
    """Fetch data, split train/test, grid search params on train, report best on test."""
    klines = fetch_klines(symbol, interval=interval, limit=train_bars + test_bars)
    if len(klines) < train_bars + test_bars:
        print(f"Not enough data: {len(klines)} bars")
        return
    train = klines[:train_bars]
    test = klines[train_bars : train_bars + test_bars]

    best_train_return = -1e9
    best_config = None
    grid = [
        {"atr_period": 10, "factor": 3.0, "tp_rr": 2.0},
        {"atr_period": 14, "factor": 3.0, "tp_rr": 2.0},
        {"atr_period": 10, "factor": 2.5, "tp_rr": 2.0},
        {"atr_period": 10, "factor": 3.0, "tp_rr": 1.5},
    ]
    for p in grid:
        cfg = StrategyConfig(
            atr_period=p["atr_period"],
            supertrend_factor=p["factor"],
            tp_rr=p["tp_rr"],
        )
        res = run_backtest(train, cfg)
        if res["total_return_pct"] > best_train_return:
            best_train_return = res["total_return_pct"]
            best_config = (cfg, p)

    if best_config is None:
        print("No valid config")
        return
    cfg, p = best_config
    print(f"Best on train: atr_period={p['atr_period']}, factor={p['factor']}, tp_rr={p['tp_rr']} -> return={best_train_return:.2f}%")
    test_res = run_backtest(test, cfg)
    print(f"Test: return={test_res['total_return_pct']:.2f}%, trades={test_res['completed_trades']}, win_rate={test_res['win_rate']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT", help="Symbol for backtest")
    ap.add_argument("--interval", default="15", help="Kline interval")
    ap.add_argument("--limit", type=int, default=500, help="Number of bars")
    ap.add_argument("--walk-forward", action="store_true", help="Run walk-forward param search")
    args = ap.parse_args()

    if args.walk_forward:
        walk_forward(args.symbol, args.interval)
        return

    klines = fetch_klines(args.symbol, interval=args.interval, limit=args.limit)
    print(f"Loaded {len(klines)} bars for {args.symbol}")
    if len(klines) < 30:
        print("Not enough data")
        return

    config = StrategyConfig()
    res = run_backtest(klines, config)
    print(f"Trades: {res['completed_trades']}, Total return: {res['total_return_pct']:.2f}%, Win rate: {res['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
