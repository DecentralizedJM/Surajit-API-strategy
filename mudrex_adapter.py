"""
Mudrex Strategy Adapter
=======================

Bridge between the Supertrend TSL Strategy and Mudrex SDK.
Handles order execution, position management, and trailing stop updates.
"""

import sys
import os
import time
import math
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from datetime import datetime

# Add mudrex-sdk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mudrex-sdk'))

from mudrex import MudrexClient
from mudrex.exceptions import MudrexAPIError, MudrexRateLimitError, MudrexValidationError

from strategy import Signal, StrategyResult
from config import Config, MudrexConfig, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    """State for an open position."""
    symbol: str
    position_id: str
    side: Signal
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    highest_price: float
    lowest_price: float
    initial_stop_loss: float = 0.0  # For 1R trailing activation
    bars_in_trade: int = 0
    entry_time: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "position_id": self.position_id,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "initial_stop_loss": self.initial_stop_loss,
            "bars_in_trade": self.bars_in_trade,
            "entry_time": self.entry_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionState":
        return cls(
            symbol=data["symbol"],
            position_id=data["position_id"],
            side=Signal(data["side"]),
            entry_price=data["entry_price"],
            quantity=data["quantity"],
            stop_loss=data["stop_loss"],
            take_profit=data["take_profit"],
            highest_price=data["highest_price"],
            lowest_price=data["lowest_price"],
            initial_stop_loss=data.get("initial_stop_loss", data["stop_loss"]),
            bars_in_trade=data.get("bars_in_trade", 0),
            entry_time=datetime.fromisoformat(data["entry_time"]),
        )


@dataclass
class ExecutionResult:
    """Result of a trade execution."""
    success: bool
    action: str  # "OPEN_LONG", "OPEN_SHORT", "CLOSE", "UPDATE_TSL", "NONE"
    symbol: str
    message: str
    order_id: Optional[str] = None
    position_state: Optional[PositionState] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "action": self.action,
            "symbol": self.symbol,
            "message": self.message,
        }
        if self.order_id:
            result["order_id"] = self.order_id
        if self.position_state:
            result["position_state"] = self.position_state.to_dict()
        if self.error:
            result["error"] = self.error
        return result


class MudrexStrategyAdapter:
    """
    Adapter between Supertrend strategy and Mudrex SDK.
    
    Handles:
    - Order execution (market orders with SL/TP)
    - Position monitoring
    - Trailing stop loss updates
    - Position sizing
    """
    
    def __init__(
        self,
        mudrex_config: MudrexConfig,
        trading_config: TradingConfig,
        dry_run: bool = False,
    ):
        self.mudrex_config = mudrex_config
        self.trading_config = trading_config
        self.dry_run = dry_run or trading_config.dry_run
        
        # Initialize Mudrex client
        self._client: Optional[MudrexClient] = None
        
        # Position state cache
        self._positions: Dict[str, PositionState] = {}
        # Asset specs from list_all() to avoid 500+ get(symbol) calls per cycle (rate limits)
        self._asset_specs_map: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_asset_specs(self) -> None:
        """Build symbol -> {min_quantity, quantity_step, max_leverage} from list_all() (single bulk call)."""
        if self._asset_specs_map is not None:
            return
        try:
            assets = self.client.assets.list_all()
            self._asset_specs_map = {}
            for a in assets:
                self._asset_specs_map[a.symbol] = {
                    "symbol": a.symbol,
                    "min_quantity": float(a.min_quantity) if a.min_quantity else 0.001,
                    "max_leverage": int(a.max_leverage) if a.max_leverage else 20,
                    "quantity_step": float(a.quantity_step) if a.quantity_step else 0.001,
                }
            logger.info(f"Loaded asset specs for {len(self._asset_specs_map)} symbols (bulk)")
        except MudrexAPIError as e:
            logger.error(f"Failed to load asset list: {e}")
            self._asset_specs_map = {}

    @property
    def client(self) -> MudrexClient:
        """Lazy-load Mudrex client."""
        if self._client is None:
            if self.dry_run:
                logger.info("Running in DRY RUN mode - no trades will be executed")
            
            self._client = MudrexClient(
                api_secret=self.mudrex_config.api_secret,
                base_url=self.mudrex_config.base_url,
                timeout=self.mudrex_config.timeout,
                rate_limit=self.mudrex_config.rate_limit,
                max_retries=self.mudrex_config.max_retries,
            )
        return self._client
    
    def get_balance(self) -> float:
        """Get available futures balance."""
        try:
            balance = self.client.wallet.get_futures_balance()
            available = float(balance.available)
            logger.info(f"Futures balance: ${available:.2f}")
            return available
        except MudrexAPIError as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def get_asset_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset information from bulk list; fallback to get(symbol) only if symbol not in list."""
        self._ensure_asset_specs()
        if self._asset_specs_map and symbol in self._asset_specs_map:
            return self._asset_specs_map[symbol]
        try:
            asset = self.client.assets.get(symbol)
            info = {
                "symbol": asset.symbol,
                "min_quantity": float(asset.min_quantity) if asset.min_quantity else 0.001,
                "max_leverage": int(asset.max_leverage) if asset.max_leverage else 20,
                "quantity_step": float(asset.quantity_step) if asset.quantity_step else 0.001,
            }
            if self._asset_specs_map is not None:
                self._asset_specs_map[symbol] = info
            return info
        except MudrexAPIError as e:
            logger.error(f"Failed to get asset info for {symbol}: {e}")
            return None
    
    def round_quantity(self, quantity: float, quantity_step: float) -> float:
        """Round quantity to valid step size."""
        if quantity_step <= 0:
            quantity_step = 0.001
        
        # Convert to Decimal for precision
        qty = Decimal(str(quantity))
        step = Decimal(str(quantity_step))
        
        # Round down to nearest step
        rounded = (qty / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
        return float(rounded)
    
    def set_leverage(self, symbol: str, leverage: str) -> bool:
        """Set leverage for a symbol."""
        try:
            self.client.leverage.set(
                symbol=symbol,
                leverage=leverage,
                margin_type=self.trading_config.margin_type,
            )
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except MudrexAPIError as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    def get_open_positions(self) -> Dict[str, PositionState]:
        """Get all open positions and sync with local cache."""
        try:
            positions = self.client.positions.list_open()
            
            # Update cache with exchange data; hydrate positions not in cache (e.g. from prior run)
            for pos in positions:
                if pos.symbol in self._positions:
                    # Update existing position state
                    state = self._positions[pos.symbol]
                    current_price = float(pos.mark_price) if pos.mark_price else state.entry_price
                    # Update high/low for trailing stop
                    if state.side == Signal.LONG:
                        state.highest_price = max(state.highest_price, current_price)
                    else:
                        state.lowest_price = min(state.lowest_price, current_price)
                else:
                    # Hydrate from exchange (prior run or manual); prevents opening over max_positions
                    entry = float(pos.entry_price) if pos.entry_price else 0.0
                    mark = float(pos.mark_price) if pos.mark_price else entry
                    sl = float(pos.stoploss_price) if pos.stoploss_price else 0.0
                    tp = float(pos.takeprofit_price) if pos.takeprofit_price else 0.0
                    qty = float(pos.quantity) if pos.quantity else 0.0
                    side = Signal(getattr(pos.side, "value", str(pos.side)))
                    self._positions[pos.symbol] = PositionState(
                        symbol=pos.symbol,
                        position_id=pos.position_id,
                        side=side,
                        entry_price=entry,
                        quantity=qty,
                        stop_loss=sl,
                        take_profit=tp,
                        highest_price=max(entry, mark) if side == Signal.LONG else entry,
                        lowest_price=min(entry, mark) if side == Signal.SHORT else entry,
                        initial_stop_loss=sl,
                    )
            
            return self._positions
        except MudrexAPIError as e:
            logger.error(f"Failed to get positions: {e}")
            return self._positions
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for a symbol."""
        return symbol in self._positions
    
    def execute_signal(
        self,
        symbol: str,
        signal_result: StrategyResult,
        balance: float,
    ) -> ExecutionResult:
        """
        Execute a trading signal.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., "BTCUSDT")
        signal_result : StrategyResult
            Signal from strategy
        balance : float
            Available balance
            
        Returns:
        --------
        ExecutionResult
            Result of execution
        """
        signal = signal_result.signal
        
        # Check if we already have a position
        if self.has_position(symbol):
            existing = self._positions[symbol]
            
            # If signal is opposite, close existing position
            if (existing.side == Signal.LONG and signal == Signal.SHORT) or \
               (existing.side == Signal.SHORT and signal == Signal.LONG):
                close_result = self.close_position(symbol)
                if not close_result.success:
                    return close_result
                # Continue to open new position
            else:
                # Same direction or neutral - no action needed
                return ExecutionResult(
                    success=True,
                    action="NONE",
                    symbol=symbol,
                    message=f"Position already exists for {symbol}",
                )
        
        # No position and neutral signal - nothing to do
        if signal == Signal.NEUTRAL:
            return ExecutionResult(
                success=True,
                action="NONE",
                symbol=symbol,
                message=f"No signal for {symbol}",
            )

        # Enforce max concurrent positions
        if len(self._positions) >= self.trading_config.max_positions:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Max positions reached ({self.trading_config.max_positions})",
                error="Max positions reached",
            )
        
        # Get asset info for quantity rounding (required; do not trade with default spec)
        asset_info = self.get_asset_info(symbol)
        if not asset_info:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Failed to get asset info for {symbol}",
                error="Asset info unavailable",
            )
        
        # Calculate position size: margin_percent of balance as margin
        # Margin = (Quantity * Price) / Leverage
        # Quantity = (Margin * Leverage) / Price
        # Margin = Balance * (margin_percent / 100)
        leverage = int(self.trading_config.leverage)
        leverage = max(
            self.trading_config.leverage_min,
            min(self.trading_config.leverage_max, leverage),
        )
        margin_pct = self.trading_config.margin_percent / 100.0
        margin = balance * margin_pct
        
        raw_quantity = (margin * leverage) / signal_result.entry_price
        
        # Round to valid quantity
        quantity = self.round_quantity(raw_quantity, asset_info["quantity_step"])
        
        # Check minimum quantity
        if quantity < asset_info["min_quantity"]:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Position size {quantity} below minimum {asset_info['min_quantity']}",
                error="Position too small",
            )
        
        # Open position with 1:2 RR
        return self.open_position(
            symbol=symbol,
            side=signal,
            quantity=quantity,
            entry_price=signal_result.entry_price,
            stop_loss=signal_result.stop_loss,
            take_profit=signal_result.take_profit,
        )
    
    def execute_proposed_position(
        self,
        symbol: str,
        proposed_position: dict,
        balance: Optional[float] = None,
    ) -> ExecutionResult:
        """Open position from strategy_core proposed_position (quantity, leverage precomputed)."""
        if len(self._positions) >= self.trading_config.max_positions:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Max positions reached ({self.trading_config.max_positions})",
                error="Max positions reached",
            )
        side = Signal(proposed_position["side"])
        quantity = proposed_position["quantity"]
        entry_price = proposed_position["entry_price"]
        notional = quantity * entry_price
        min_val = getattr(self.trading_config, "min_order_value", 7.0)
        asset_info = self.get_asset_info(symbol)
        if not asset_info:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Failed to get asset info for {symbol}",
                error="Asset info unavailable",
            )
        asset_max_lev = int(asset_info.get("max_leverage", 20))
        effective_max_lev = min(self.trading_config.leverage_max, asset_max_lev)
        lev = int(proposed_position.get("leverage", self.trading_config.leverage))
        lev = max(
            self.trading_config.leverage_min,
            min(effective_max_lev, lev),
        )

        if notional < min_val and balance is not None and balance > 0:
            margin_pct = self.trading_config.margin_percent / 100.0
            margin = balance * margin_pct
            required_lev = min_val / margin
            if required_lev <= effective_max_lev:
                lev = max(lev, math.ceil(required_lev))
                lev = min(lev, effective_max_lev)
                quantity = (margin * lev) / entry_price
                quantity = self.round_quantity(quantity, asset_info["quantity_step"])
                if quantity < asset_info["min_quantity"]:
                    return ExecutionResult(
                        success=False,
                        action="NONE",
                        symbol=symbol,
                        message=f"Scaled qty {quantity} below asset min {asset_info['min_quantity']}",
                        error="Order value below minimum",
                    )
                notional = quantity * entry_price
                if notional >= min_val:
                    logger.info(f"Scaled to {lev}x (asset max {asset_max_lev}x) for min order value (notional ${notional:.2f})")
            else:
                return ExecutionResult(
                    success=False,
                    action="NONE",
                    symbol=symbol,
                    message=f"Cannot reach min ${min_val:.0f} (need {required_lev:.0f}x > asset max {asset_max_lev}x)",
                    error="Order value below minimum",
                )

        if notional < min_val:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Order value ${notional:.2f} below minimum ${min_val:.0f}",
                error="Order value below minimum",
            )

        leverage = str(lev)
        stop_loss = proposed_position["stop_loss"]
        take_profit = proposed_position["take_profit"]
        return self.open_position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
        )

    def open_position(
        self,
        symbol: str,
        side: Signal,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Open a new position.
        
        Parameters:
        -----------
        symbol : str
            Trading symbol
        side : Signal
            LONG or SHORT
        quantity : float
            Position size
        entry_price : float
            Expected entry price
        stop_loss : float
            Stop loss price
        take_profit : float
            Take profit price
            
        Returns:
        --------
        ExecutionResult
            Result of execution
        """
        action = f"OPEN_{side.value}"
        lev = leverage or self.trading_config.leverage

        logger.info(f"Opening {side.value} position on {symbol}")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"  Entry: ${entry_price:.4f}")
        logger.info(f"  Stop Loss: ${stop_loss:.4f}")
        logger.info(f"  Take Profit: ${take_profit:.4f}")
        
        if not self.dry_run:
            delay = getattr(self.trading_config, "order_delay_seconds", 4.0)
            if delay > 0:
                time.sleep(delay)
        
        if self.dry_run:
            # Simulate order in dry run mode
            position_state = PositionState(
                symbol=symbol,
                position_id=f"DRY_RUN_{symbol}_{datetime.utcnow().timestamp()}",
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                highest_price=entry_price,
                lowest_price=entry_price,
                initial_stop_loss=stop_loss,
            )
            self._positions[symbol] = position_state

            return ExecutionResult(
                success=True,
                action=action,
                symbol=symbol,
                message=f"[DRY RUN] Opened {side.value} position",
                order_id="DRY_RUN",
                position_state=position_state,
            )
        
        try:
            # Set leverage first
            self.set_leverage(symbol, lev)

            # Place market order with SL/TP
            order = self.client.orders.create_market_order(
                symbol=symbol,
                side=side.value,
                quantity=str(quantity),
                leverage=lev,
                stoploss_price=str(round(stop_loss, 4)),
                takeprofit_price=str(round(take_profit, 4)),
            )
            
            logger.info(f"Order placed: {order.order_id}")
            
            # Create position state
            position_state = PositionState(
                symbol=symbol,
                position_id=order.order_id,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                highest_price=entry_price,
                lowest_price=entry_price,
                initial_stop_loss=stop_loss,
            )
            self._positions[symbol] = position_state
            
            return ExecutionResult(
                success=True,
                action=action,
                symbol=symbol,
                message=f"Opened {side.value} position",
                order_id=order.order_id,
                position_state=position_state,
            )
        
        except MudrexRateLimitError as e:
            logger.warning(f"Rate limited: {e}")
            return ExecutionResult(
                success=False,
                action=action,
                symbol=symbol,
                message="Rate limited",
                error=str(e),
            )
        
        except MudrexValidationError as e:
            logger.error(f"Validation error: {e}")
            return ExecutionResult(
                success=False,
                action=action,
                symbol=symbol,
                message="Validation error",
                error=str(e),
            )
        
        except MudrexAPIError as e:
            logger.error(f"API error: {e}")
            return ExecutionResult(
                success=False,
                action=action,
                symbol=symbol,
                message="API error",
                error=str(e),
            )
    
    def close_position(self, symbol: str) -> ExecutionResult:
        """Close an existing position."""
        if symbol not in self._positions:
            return ExecutionResult(
                success=False,
                action="CLOSE",
                symbol=symbol,
                message=f"No position found for {symbol}",
            )
        
        position = self._positions[symbol]
        
        logger.info(f"Closing position on {symbol}")
        
        if self.dry_run:
            del self._positions[symbol]
            return ExecutionResult(
                success=True,
                action="CLOSE",
                symbol=symbol,
                message=f"[DRY RUN] Closed position",
            )
        
        try:
            # Find position on exchange
            positions = self.client.positions.list_open()
            exchange_position = None
            
            for pos in positions:
                if pos.symbol == symbol:
                    exchange_position = pos
                    break
            
            if exchange_position:
                self.client.positions.close(exchange_position.position_id)
                logger.info(f"Closed position {exchange_position.position_id}")
            
            del self._positions[symbol]
            
            return ExecutionResult(
                success=True,
                action="CLOSE",
                symbol=symbol,
                message="Position closed",
            )
        
        except MudrexAPIError as e:
            logger.error(f"Failed to close position: {e}")
            return ExecutionResult(
                success=False,
                action="CLOSE",
                symbol=symbol,
                message="Failed to close position",
                error=str(e),
            )
    
    def update_trailing_stop(
        self,
        symbol: str,
        new_stop_loss: float,
    ) -> ExecutionResult:
        """Update trailing stop loss for a position."""
        if symbol not in self._positions:
            return ExecutionResult(
                success=False,
                action="UPDATE_TSL",
                symbol=symbol,
                message=f"No position found for {symbol}",
            )
        
        position = self._positions[symbol]
        old_stop = position.stop_loss
        
        # Validate stop movement
        if position.side == Signal.LONG:
            if new_stop_loss <= old_stop:
                return ExecutionResult(
                    success=True,
                    action="UPDATE_TSL",
                    symbol=symbol,
                    message="TSL not moved (would decrease)",
                )
        else:
            if new_stop_loss >= old_stop:
                return ExecutionResult(
                    success=True,
                    action="UPDATE_TSL",
                    symbol=symbol,
                    message="TSL not moved (would increase)",
                )
        
        logger.info(f"Updating TSL for {symbol}: {old_stop:.4f} -> {new_stop_loss:.4f}")
        
        if self.dry_run:
            position.stop_loss = new_stop_loss
            return ExecutionResult(
                success=True,
                action="UPDATE_TSL",
                symbol=symbol,
                message=f"[DRY RUN] Updated TSL",
                position_state=position,
            )
        
        try:
            # Find position on exchange
            positions = self.client.positions.list_open()
            
            for pos in positions:
                if pos.symbol == symbol:
                    self.client.positions.set_stoploss(
                        position_id=pos.position_id,
                        price=str(round(new_stop_loss, 4)),
                    )
                    position.stop_loss = new_stop_loss
                    
                    return ExecutionResult(
                        success=True,
                        action="UPDATE_TSL",
                        symbol=symbol,
                        message=f"Updated TSL from {old_stop:.4f} to {new_stop_loss:.4f}",
                        position_state=position,
                    )
            
            return ExecutionResult(
                success=False,
                action="UPDATE_TSL",
                symbol=symbol,
                message="Position not found on exchange",
            )
        
        except MudrexAPIError as e:
            logger.error(f"Failed to update TSL: {e}")
            return ExecutionResult(
                success=False,
                action="UPDATE_TSL",
                symbol=symbol,
                message="Failed to update TSL",
                error=str(e),
            )
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load position state from storage."""
        positions_data = state.get("positions", {})
        self._positions = {
            symbol: PositionState.from_dict(data)
            for symbol, data in positions_data.items()
        }
        logger.info(f"Loaded {len(self._positions)} positions from state")
    
    def save_state(self) -> Dict[str, Any]:
        """Save position state for storage."""
        return {
            "positions": {
                symbol: state.to_dict()
                for symbol, state in self._positions.items()
            },
            "updated_at": datetime.utcnow().isoformat(),
        }
    
    def close(self) -> None:
        """Close the Mudrex client connection."""
        if self._client:
            self._client.close()
            self._client = None
