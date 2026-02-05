"""
Supertrend Mudrex Trading Bot
=============================

Main trading bot that integrates Supertrend TSL Strategy with Mudrex SDK.
Fetches OHLCV data, generates signals, and executes trades.
"""

import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

import ccxt
import pandas as pd

from strategy import SupertrendTSLStrategy, Signal, StrategyResult
from mudrex_adapter import MudrexStrategyAdapter, ExecutionResult, PositionState
from config import Config, get_config

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
    1. Fetches OHLCV data for configured symbols
    2. Generates signals using Supertrend TSL strategy
    3. Executes trades via Mudrex SDK
    4. Manages trailing stop losses
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Initialize strategy
        self.strategy = SupertrendTSLStrategy(
            atr_period=self.config.strategy.atr_period,
            factor=self.config.strategy.factor,
            risk_atr_len=self.config.strategy.risk_atr_len,
            risk_atr_mult=self.config.strategy.risk_atr_mult,
            tsl_atr_len=self.config.strategy.tsl_atr_len,
            tsl_mult=self.config.strategy.tsl_mult,
            tp_rr=self.config.strategy.tp_rr,
            position_size_pct=self.config.strategy.position_size_pct,
        )
        
        # Initialize Mudrex adapter
        self.adapter = MudrexStrategyAdapter(
            mudrex_config=self.config.mudrex,
            trading_config=self.config.trading,
            dry_run=self.config.trading.dry_run,
        )
        
        # Initialize CCXT exchange
        self.exchange = self._init_exchange()
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange for price data."""
        exchange_id = self.config.ccxt.exchange
        exchange_class = getattr(ccxt, exchange_id)
        
        exchange_config = {
            "enableRateLimit": True,
        }
        
        if self.config.ccxt.api_key and self.config.ccxt.api_secret:
            exchange_config["apiKey"] = self.config.ccxt.api_key
            exchange_config["secret"] = self.config.ccxt.api_secret
        
        exchange = exchange_class(exchange_config)
        
        if self.config.ccxt.sandbox:
            exchange.set_sandbox_mode(True)
        
        logger.info(f"Initialized {exchange_id} exchange for price data")
        return exchange
    
    def fetch_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
            
        Returns:
        --------
        Optional[pd.DataFrame]
            OHLCV dataframe or None if fetch failed
        """
        try:
            # Convert symbol format for CCXT (e.g., BTCUSDT -> BTC/USDT)
            ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}"  # Assumes 4-char quote (USDT)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=self.config.trading.timeframe,
                limit=self.config.trading.lookback_periods,
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df
        
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None
    
    def process_symbol(
        self,
        symbol: str,
        balance: float,
    ) -> ExecutionResult:
        """
        Process a single symbol: fetch data, generate signal, execute.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        balance : float
            Available balance
            
        Returns:
        --------
        ExecutionResult
            Result of processing
        """
        logger.info(f"Processing {symbol}...")
        
        # Fetch OHLCV data
        df = self.fetch_ohlcv(symbol)
        if df is None or len(df) < self.config.trading.lookback_periods // 2:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message="Failed to fetch sufficient OHLCV data",
                error="Data fetch failed",
            )
        
        # Check if we have an existing position
        if self.adapter.has_position(symbol):
            return self._manage_existing_position(symbol, df)
        
        # Generate signal
        signal_result = self.strategy.generate_signal(df)
        logger.info(f"{symbol} Signal: {signal_result.signal.value}")
        
        if signal_result.signal == Signal.NEUTRAL:
            return ExecutionResult(
                success=True,
                action="NONE",
                symbol=symbol,
                message=f"No signal for {symbol}",
            )
        
        # Execute signal
        return self.adapter.execute_signal(
            symbol=symbol,
            signal_result=signal_result,
            balance=balance,
        )
    
    def _manage_existing_position(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> ExecutionResult:
        """
        Manage an existing position (update TSL, check for close signal).
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        df : pd.DataFrame
            OHLCV data
            
        Returns:
        --------
        ExecutionResult
            Result of management
        """
        position = self.adapter._positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Update high/low for trailing stop
        if position.side == Signal.LONG:
            position.highest_price = max(position.highest_price, float(df['high'].iloc[-1]))
        else:
            position.lowest_price = min(position.lowest_price, float(df['low'].iloc[-1]))
        
        # Check for opposite signal
        signal_result = self.strategy.generate_signal(df)
        
        if (position.side == Signal.LONG and signal_result.signal == Signal.SHORT) or \
           (position.side == Signal.SHORT and signal_result.signal == Signal.LONG):
            # Close position on opposite signal
            logger.info(f"Opposite signal detected for {symbol}, closing position")
            return self.adapter.close_position(symbol)
        
        # Calculate new trailing stop
        new_tsl = self.strategy.calculate_trailing_stop(
            df=df,
            position_side=position.side,
            current_stop=position.stop_loss,
            highest_price=position.highest_price,
            lowest_price=position.lowest_price,
        )
        
        # Update TSL if moved favorably
        if position.side == Signal.LONG and new_tsl > position.stop_loss:
            return self.adapter.update_trailing_stop(symbol, new_tsl)
        elif position.side == Signal.SHORT and new_tsl < position.stop_loss:
            return self.adapter.update_trailing_stop(symbol, new_tsl)
        
        return ExecutionResult(
            success=True,
            action="NONE",
            symbol=symbol,
            message=f"Position maintained, current PnL tracking",
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
                # Filter for USDT pairs only to match CCXT expectations and common strategy
                symbols = [a.symbol for a in assets if a.symbol.endswith("USDT") and a.is_active]
                logger.info(f"Discovered {len(symbols)} active USDT pairs")
            except Exception as e:
                logger.error(f"Failed to fetch assets: {e}")
                errors.append(f"Asset discovery failed: {e}")
                symbols = []

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
                
                if not result.success and result.error:
                    # Don't clutter logs with 'Data fetch failed' for obscure pairs
                    if "Data fetch failed" not in result.message:
                        errors.append(f"{symbol}: {result.error}")
            
            except Exception as e:
                # Catch CCXT errors for specific pairs quietly
                if "symbol not found" in str(e).lower():
                    continue
                error = f"Error processing {symbol}: {str(e)}"
                logger.debug(error)
            
            # Small delay between symbols to avoid rate limits
            time.sleep(0.1)
        
        success = len(errors) == 0
        
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
    
    def run_continuous(self, interval_seconds: int = 300) -> None:
        """
        Run the bot continuously at specified interval.
        
        Parameters:
        -----------
        interval_seconds : int
            Seconds between executions (default: 300 = 5 minutes)
        """
        logger.info(f"Starting continuous bot with {interval_seconds}s interval")
        
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.exception(f"Bot error: {e}")
            
            logger.info(f"Sleeping for {interval_seconds} seconds...")
            time.sleep(interval_seconds)
        
        self.close()
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from storage."""
        self.adapter.load_state(state)
    
    def save_state(self) -> Dict[str, Any]:
        """Save state for storage."""
        return self.adapter.save_state()
    
    def close(self) -> None:
        """Clean up resources."""
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
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds (for continuous mode)")
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
        if args.continuous:
            bot.run_continuous(interval_seconds=args.interval)
        else:
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
