"""
Supertrend Mudrex Trading Bot
=============================

Main trading bot that integrates Supertrend TSL Strategy with Mudrex SDK.
Uses real-time OHLCV data from Bybit WebSocket (via DataManager).
Generates signals and executes trades via Mudrex SDK.
"""

import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd

from strategy_core import process_candle, TradeState, StrategyConfig, update_trailing
from mudrex_adapter import MudrexStrategyAdapter, ExecutionResult, PositionState
from mudrex_adapter import Signal
from config import Config, get_config
from data_manager import DataManager
from telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


@dataclass
class BotExecutionResult:
    """Result from a single bot execution cycle."""
    success: bool
    timestamp: datetime
    symbols_processed: int
    signals_generated: int
    trades_executed: int
    tsl_updates: int
    errors: List[str] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "symbols_processed": self.symbols_processed,
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "tsl_updates": self.tsl_updates,
            "errors": self.errors,
            "results": self.results,
        }


class SupertrendMudrexBot:
    """
    Trading bot combining Supertrend strategy with Mudrex execution.
    
    This bot:
    1. Uses DataManager to get real-time OHLCV data from Bybit
    2. Generates signals using Supertrend TSL strategy
    3. Executes trades via Mudrex SDK
    4. Manages trailing stop losses
    """
    
    def __init__(self, config: Optional[Config] = None, data_manager: Optional[DataManager] = None):
        self.config = config or get_config()
        self.data_manager = data_manager or DataManager(
            interval=self.config.trading.timeframe,
            lookback=self.config.trading.lookback_periods
        )

        self.strategy_config = StrategyConfig(
            atr_period=self.config.strategy.atr_period,
            supertrend_factor=self.config.strategy.factor,
            risk_atr_mult=self.config.strategy.risk_atr_mult,
            tsl_atr_mult=self.config.strategy.tsl_mult,
            tp_rr=self.config.strategy.tp_rr,
            margin_pct=self.config.strategy.margin_pct,
            leverage_min=self.config.strategy.leverage_min,
            leverage_max=self.config.strategy.leverage_max,
            leverage=int(self.config.trading.leverage) if self.config.trading.leverage else 5,
            max_bars_in_trade=self.config.strategy.max_bars_in_trade,
            volatility_filter_enabled=self.config.strategy.volatility_filter_enabled,
            volatility_median_window=self.config.strategy.volatility_median_window,
        )

        # Initialize Mudrex adapter
        self.adapter = MudrexStrategyAdapter(
            mudrex_config=self.config.mudrex,
            trading_config=self.config.trading,
            dry_run=self.config.trading.dry_run,
        )

        # Initialize Telegram notifier (no-op if not configured)
        self.notifier = TelegramNotifier(
            bot_token=self.config.telegram.bot_token,
            chat_ids=self.config.telegram.chat_ids,
        )
        
    def _df_to_ohlcv(self, df: pd.DataFrame) -> list[dict]:
        """Convert DataFrame to OHLCV list of dicts for strategy_core."""
        ohlcv = []
        for _, row in df.iterrows():
            ohlcv.append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if "volume" in row else 0.0,
            })
        return ohlcv

    def _position_to_trade_state(self, position: PositionState) -> TradeState:
        """Build TradeState from adapter PositionState."""
        trailing = position.stop_loss if position.stop_loss != position.initial_stop_loss else None
        return TradeState(
            position_side=position.side.value,
            entry_price=position.entry_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            trailing_stop=trailing,
            bars_in_trade=position.bars_in_trade,
            extreme_price=position.highest_price if position.side == Signal.LONG else position.lowest_price,
            initial_stop_loss=position.initial_stop_loss,
        )

    def process_symbol(
        self,
        symbol: str,
        balance: float,
    ) -> ExecutionResult:
        """
        Process a single symbol: get data, run strategy_core, execute.
        """
        logger.debug(f"Processing {symbol}...")

        df = self.data_manager.get_ohlcv(symbol)
        if df is None or len(df) < 20:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Insufficient candle data for {symbol} ({len(df) if df is not None else 0} candles)",
                error="Data unavailable",
            )

        ohlcv = self._df_to_ohlcv(df)
        asset_info = self.adapter.get_asset_info(symbol)
        contract_specs = (
            {
                "min_quantity": float(asset_info["min_quantity"]),
                "quantity_step": float(asset_info["quantity_step"]),
            }
            if asset_info
            else {"min_quantity": 0.001, "quantity_step": 0.001}
        )

        prev_state = TradeState.flat()
        if self.adapter.has_position(symbol):
            pos = self.adapter._positions[symbol]
            prev_state = self._position_to_trade_state(pos)

        output = process_candle(
            ohlcv=ohlcv,
            account_equity=balance,
            contract_specs=contract_specs,
            prev_state=prev_state,
            config=self.strategy_config,
        )

        if output["signal"] != "HOLD":
            logger.info(f"{symbol} Signal: {output['signal']} ({output['reason']})")
        else:
            logger.debug(f"{symbol} Signal: HOLD ({output['reason']})")

        if output["signal"] in ("LONG", "SHORT") and output.get("proposed_position"):
            result = self.adapter.execute_proposed_position(symbol, output["proposed_position"], balance=balance)
            if result.success and result.position_state and result.action in ("OPEN_LONG", "OPEN_SHORT"):
                pp = output["proposed_position"]
                self.notifier.notify_open(
                    symbol=symbol,
                    side=pp["side"],
                    quantity=pp["quantity"],
                    entry_price=pp["entry_price"],
                    stop_loss=pp["stop_loss"],
                    take_profit=pp["take_profit"],
                    leverage=pp.get("leverage", 5),
                    dry_run=self.config.trading.dry_run,
                )
            return result

        if output["signal"] == "EXIT":
            logger.info(f"Exit signal for {symbol}: {output['reason']}")
            pos = self.adapter._positions.get(symbol)
            exit_price = float(df["close"].iloc[-1]) if pos else 0.0
            result = self.adapter.close_position(symbol)
            if result.success and pos:
                self.notifier.notify_close(
                    symbol=symbol,
                    side=pos.side.value,
                    reason=output["reason"],
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    dry_run=self.config.trading.dry_run,
                )
            return result

        if output["signal"] == "HOLD" and self.adapter.has_position(symbol):
            pos = self.adapter._positions[symbol]
            pos.bars_in_trade += 1
            pos.highest_price = max(pos.highest_price, float(df["high"].iloc[-1]))
            pos.lowest_price = min(pos.lowest_price, float(df["low"].iloc[-1]))
            state_updated = self._position_to_trade_state(pos)
            import numpy as np
            from strategy_core.indicators import wilder_atr
            h = np.array([float(r["high"]) for r in ohlcv])
            l_arr = np.array([float(r["low"]) for r in ohlcv])
            c = np.array([float(r["close"]) for r in ohlcv])
            atr_arr = wilder_atr(h, l_arr, c, self.strategy_config.atr_period)
            atr_val = float(atr_arr[-1]) if len(atr_arr) > 0 and not np.isnan(atr_arr[-1]) else 1e-6
            new_tsl = update_trailing(
                state_updated,
                float(df["high"].iloc[-1]),
                float(df["low"].iloc[-1]),
                max(atr_val, 1e-6),
                self.strategy_config.tsl_atr_mult,
            )
            if new_tsl is not None:
                if (pos.side == Signal.LONG and new_tsl > pos.stop_loss) or (pos.side == Signal.SHORT and new_tsl < pos.stop_loss):
                    return self.adapter.update_trailing_stop(symbol, new_tsl)

        return ExecutionResult(
            success=True,
            action="NONE",
            symbol=symbol,
            message=f"No action for {symbol}",
        )
    
    def run_once(self) -> BotExecutionResult:
        """
        Run one iteration of the bot.
        """
        timestamp = datetime.utcnow()
        errors: List[str] = []
        results: List[Dict[str, Any]] = []
        signals_generated = 0
        trades_executed = 0
        tsl_updates = 0
        
        logger.info("=" * 50)
        logger.info(f"Bot execution started at {timestamp.isoformat()}")
        logger.info("=" * 50)
        
        # Check balance
        balance = self.adapter.get_balance()
        if balance < self.config.trading.min_balance:
            error = f"Insufficient balance: ${balance:.2f} < ${self.config.trading.min_balance:.2f}"
            logger.error(error)
            return BotExecutionResult(
                success=False,
                timestamp=timestamp,
                symbols_processed=0,
                signals_generated=0,
                trades_executed=0,
                tsl_updates=0,
                errors=[error],
            )
        
        # Sync positions with exchange
        self.adapter.get_open_positions()
        
        # Determine symbols to process
        symbols = self.config.trading.symbols
        if not symbols:
            logger.info("No symbols specified, fetching all tradable assets from Mudrex...")
            try:
                assets = self.adapter.client.assets.list_all()
                # Use all active USDT symbols
                symbols = [a.symbol for a in assets if a.symbol.endswith("USDT") and a.is_active]
                logger.info(f"Discovered {len(symbols)} active USDT pairs")
            except Exception as e:
                logger.error(f"Failed to fetch assets: {e}")
                errors.append(f"Asset discovery failed: {e}")
                symbols = []

        if not symbols:
            return BotExecutionResult(
                success=False,
                timestamp=timestamp,
                symbols_processed=0,
                signals_generated=0,
                trades_executed=0,
                tsl_updates=0,
                errors=["No symbols available"],
            )

        # Ensure DataManager is subscribed to these symbols
        self.data_manager.subscribe(symbols)
        
        # Wait for some data to arrive if this is the first run
        if not self.data_manager.wait_for_data(symbols, timeout=10):
            logger.warning("Still waiting for some symbol data to arrive...")

        # Process each symbol
        for symbol in symbols:
            try:
                result = self.process_symbol(symbol, balance)
                results.append(result.to_dict())
                
                if result.action in ("OPEN_LONG", "OPEN_SHORT"):
                    signals_generated += 1
                    if result.success:
                        trades_executed += 1
                elif result.action == "UPDATE_TSL" and result.success:
                    tsl_updates += 1
                elif result.action == "CLOSE" and result.success:
                    trades_executed += 1
                
                if not result.success and result.error and result.error != "Data unavailable":
                    errors.append(f"{symbol}: {result.error}")
            
            except Exception as e:
                error = f"Error processing {symbol}: {str(e)}"
                logger.debug(error)
            
            # Very small delay just for logging clarity
            time.sleep(0.01)
        
        success = len(errors) == 0
        skipped_no_data = sum(
            1 for r in results
            if r.get("error") == "Data unavailable" or "Insufficient candle" in (r.get("message") or "")
        )
        coverage = self.data_manager.get_data_coverage(symbols, min_bars=20)
        logger.info(
            "cycle_metrics symbols_processed=%s signals=%s trades_executed=%s tsl_updates=%s errors=%s skipped_no_data=%s data_sufficient=%s/%s",
            len(symbols), signals_generated, trades_executed, tsl_updates, len(errors),
            skipped_no_data, coverage["symbols_with_sufficient_data"], coverage["total"],
        )

        # Cycle summary: only when trades executed, TSL updates, or errors (not signals-only)
        if trades_executed > 0 or tsl_updates > 0 or errors:
            positions_count = len(self.adapter._positions)
            opened_count = sum(1 for r in results if r.get("action") in ("OPEN_LONG", "OPEN_SHORT") and r.get("success"))
            closed_count = sum(1 for r in results if r.get("action") == "CLOSE" and r.get("success"))
            self.notifier.notify_cycle(
                balance=balance,
                positions_count=positions_count,
                signals=signals_generated,
                opened=opened_count,
                closed=closed_count,
                tsl_updates=tsl_updates,
                errors=len(errors),
                dry_run=self.config.trading.dry_run,
                timeframe=self.config.trading.timeframe,
                margin_percent=self.config.trading.margin_percent,
            )

        logger.info("=" * 50)
        logger.info(f"Bot execution completed")
        logger.info(f"  Symbols processed: {len(symbols)}")
        logger.info(f"  Signals generated: {signals_generated}")
        logger.info(f"  Trades executed: {trades_executed}")
        logger.info(f"  TSL updates: {tsl_updates}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info("=" * 50)

        return BotExecutionResult(
            success=success,
            timestamp=timestamp,
            symbols_processed=len(symbols),
            signals_generated=signals_generated,
            trades_executed=trades_executed,
            tsl_updates=tsl_updates,
            errors=errors,
            results=results,
        )
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from storage."""
        self.adapter.load_state(state)
    
    def save_state(self) -> Dict[str, Any]:
        """Save state for storage."""
        return self.adapter.save_state()
    
    def close(self) -> None:
        """Clean up resources."""
        self.data_manager.stop()
        self.adapter.close()
        logger.info("Bot closed")


def main():
    """Main entry point for local testing."""
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Supertrend Mudrex Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no real trades)")
    parser.add_argument("--symbols", nargs="+", help="Override trading symbols")
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Override from args
    if args.dry_run:
        config.trading.dry_run = True
    if args.symbols:
        config.trading.symbols = args.symbols
    
    # Create and run bot
    bot = SupertrendMudrexBot(config)
    
    try:
        # Initial wait for WebSocket data
        bot.data_manager.start()
        print("Starting WebSocket and waiting for initial data...")
        time.sleep(10)
        
        result = bot.run_once()
        print("\nExecution Result:")
        print(f"  Success: {result.success}")
        print(f"  Symbols: {result.symbols_processed}")
        print(f"  Trades: {result.trades_executed}")
        print(f"  TSL Updates: {result.tsl_updates}")
        if result.errors:
            print(f"  Errors: {result.errors}")
    finally:
        bot.close()


if __name__ == "__main__":
    main()
