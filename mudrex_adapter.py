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


class RateLimitCooldownError(Exception):
    """Raised when we are in rate-limit cooldown and skip making the request."""
    pass


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
    # Optional: when collecting "best open" candidates (not executed yet)
    proposed_position: Optional[Dict[str, Any]] = None
    notional: Optional[float] = None

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
        if self.proposed_position is not None:
            result["proposed_position"] = self.proposed_position
        if self.notional is not None:
            result["notional"] = self.notional
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
        self._asset_list_cache: Optional[List[Any]] = None
        self._asset_specs_last_attempt: float = 0.0
        self._asset_specs_last_error: float = 0.0
        self._asset_specs_cooldown_seconds: float = 120.0
        self._balance_cache: Optional[float] = None
        self._balance_cache_ts: float = 0.0
        self._balance_cache_ttl: float = 30.0
        # Mudrex rate limit: 2 req/s, 50/min — throttle before every API call
        self._mudrex_last_call_time: float = 0.0
        self._mudrex_call_times: List[float] = []
        # When we get 429, stop calling Mudrex until this time (avoid retries; recovery can take up to 24h)
        self._rate_limited_until: float = 0.0

    def _throttle(self) -> None:
        """Enforce Mudrex limits: 2 req/s and 50/min; skip if in rate-limit cooldown."""
        if self.dry_run:
            return
        now = time.time()
        cooldown = getattr(self.mudrex_config, "rate_limit_cooldown_seconds", 3600.0)
        if self._rate_limited_until > 0 and now < self._rate_limited_until:
            until_str = datetime.utcfromtimestamp(self._rate_limited_until).strftime("%Y-%m-%d %H:%M UTC")
            raise RateLimitCooldownError(f"Rate limited; not calling Mudrex until {until_str}")
        if self._mudrex_last_call_time > 0:
            elapsed = now - self._mudrex_last_call_time
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
                now = time.time()
        self._mudrex_call_times = [t for t in self._mudrex_call_times if now - t < 60]
        if len(self._mudrex_call_times) >= 50:
            oldest = min(self._mudrex_call_times)
            sleep_for = 60 - (now - oldest)
            if sleep_for > 0:
                logger.debug("Mudrex 50/min throttle: sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
            now = time.time()
            self._mudrex_call_times = [t for t in self._mudrex_call_times if now - t < 60]
        self._mudrex_call_times.append(now)
        self._mudrex_last_call_time = now

    def _set_rate_limit_cooldown(self) -> None:
        """Set cooldown so we stop calling Mudrex (recovery can take up to 24h)."""
        cooldown = getattr(self.mudrex_config, "rate_limit_cooldown_seconds", 3600.0)
        self._rate_limited_until = time.time() + cooldown
        until_str = datetime.utcfromtimestamp(self._rate_limited_until).strftime("%Y-%m-%d %H:%M UTC")
        logger.warning("Mudrex rate limited; backing off until %s (no retries)", until_str)

    def is_in_rate_limit_cooldown(self) -> bool:
        """True if we are in rate-limit cooldown (not making Mudrex requests)."""
        if self._rate_limited_until <= 0:
            return False
        return time.time() < self._rate_limited_until

    def rate_limit_cooldown_until_utc(self) -> Optional[str]:
        """Human-readable cooldown end time (UTC) or None if not in cooldown."""
        if not self.is_in_rate_limit_cooldown():
            return None
        return datetime.utcfromtimestamp(self._rate_limited_until).strftime("%Y-%m-%d %H:%M UTC")

    def _ensure_asset_specs(self) -> None:
        """Build symbol -> {min_quantity, quantity_step, max_leverage} from list_all() (single bulk call)."""
        if self._asset_specs_map is not None and self._asset_list_cache is not None:
            return
        now = time.time()
        if self._asset_specs_last_error and (now - self._asset_specs_last_error) < self._asset_specs_cooldown_seconds:
            return
        self._asset_specs_last_attempt = now
        try:
            self._throttle()
            assets = self.client.assets.list_all()
            self._asset_list_cache = assets
            self._asset_specs_map = {}
            for a in assets:
                self._asset_specs_map[a.symbol] = {
                    "symbol": a.symbol,
                    "min_quantity": float(a.min_quantity) if a.min_quantity else 0.001,
                    "max_leverage": int(float(a.max_leverage)) if a.max_leverage else 20,
                    "quantity_step": float(a.quantity_step) if a.quantity_step else 0.001,
                    "is_active": bool(getattr(a, "is_active", True)),
                }
            logger.info(f"Loaded asset specs for {len(self._asset_specs_map)} symbols (bulk)")
        except RateLimitCooldownError:
            if self._asset_specs_map is None:
                self._asset_specs_map = {}
            if self._asset_list_cache is None:
                self._asset_list_cache = []
        except MudrexRateLimitError:
            self._set_rate_limit_cooldown()
            self._asset_specs_last_error = now
            if self._asset_specs_map is None:
                self._asset_specs_map = {}
            if self._asset_list_cache is None:
                self._asset_list_cache = []
        except MudrexAPIError as e:
            logger.error(f"Failed to load asset list: {e}")
            self._asset_specs_last_error = now
            if self._asset_specs_map is None:
                self._asset_specs_map = {}
            if self._asset_list_cache is None:
                self._asset_list_cache = []

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
    
    def get_balance(self) -> Optional[float]:
        """Get available futures balance. Returns None on API error (e.g. rate limit) when no cache."""
        now = time.time()
        if self._balance_cache is not None and (now - self._balance_cache_ts) < self._balance_cache_ttl:
            return self._balance_cache
        try:
            self._throttle()
            balance = self.client.wallet.get_futures_balance()
            available = float(balance.available)
            logger.info(f"Futures balance: ${available:.2f}")
            self._balance_cache = available
            self._balance_cache_ts = now
            return available
        except RateLimitCooldownError:
            if self._balance_cache is not None:
                return self._balance_cache
            return None
        except MudrexRateLimitError:
            self._set_rate_limit_cooldown()
            if self._balance_cache is not None:
                return self._balance_cache
            return None
        except MudrexAPIError as e:
            logger.error(f"Failed to get balance: {e}")
            if self._balance_cache is not None:
                return self._balance_cache
            return None
    
    def get_asset_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset information from bulk list; fallback to get(symbol) only if symbol not in list."""
        self._ensure_asset_specs()
        if self._asset_specs_map and symbol in self._asset_specs_map:
            return self._asset_specs_map[symbol]
        # Avoid per-symbol API calls when bulk list isn't available (prevents rate-limit storms)
        if self._asset_specs_map is not None and len(self._asset_specs_map) == 0:
            return None
        return None

    def ensure_asset_specs_loaded(self) -> bool:
        """Return True when bulk asset specs are available (rate-limit safe)."""
        self._ensure_asset_specs()
        return bool(self._asset_specs_map)

    def get_tradable_symbols(self) -> List[str]:
        """Get active USDT symbols using cached asset list (single bulk call)."""
        self._ensure_asset_specs()
        if not self._asset_list_cache:
            return []
        return [a.symbol for a in self._asset_list_cache if a.symbol.endswith("USDT") and getattr(a, "is_active", True)]
    
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
            self._throttle()
            self.client.leverage.set(
                symbol=symbol,
                leverage=leverage,
                margin_type=self.trading_config.margin_type,
            )
            logger.info(f"Set leverage for {symbol} to {leverage}x")
            return True
        except RateLimitCooldownError:
            return False
        except MudrexRateLimitError:
            self._set_rate_limit_cooldown()
            return False
        except MudrexAPIError as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    def get_open_positions(self) -> Dict[str, PositionState]:
        """Get all open positions and sync with local cache."""
        try:
            self._throttle()
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
        except RateLimitCooldownError:
            return self._positions
        except MudrexRateLimitError:
            self._set_rate_limit_cooldown()
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
        lev = int(self.trading_config.leverage)
        lev = max(
            self.trading_config.leverage_min,
            min(self.trading_config.leverage_max, lev),
        )
        margin_pct = self.trading_config.margin_percent / 100.0
        margin = balance * margin_pct

        raw_quantity = (margin * lev) / signal_result.entry_price

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

        # Smart scaling if notional below min order value
        min_val = getattr(self.trading_config, "min_order_value", 7.0)
        notional = quantity * signal_result.entry_price
        if notional < min_val:
            scaled = self._scale_to_min_order(symbol, signal_result.entry_price, balance, asset_info, lev)
            if scaled is not None:
                quantity = scaled["quantity"]
                lev = scaled["leverage"]
            else:
                return ExecutionResult(
                    success=False,
                    action="NONE",
                    symbol=symbol,
                    message=f"Order value ${notional:.2f} below min ${min_val:.0f} (even after scaling)",
                    error="Order value below minimum after scaling",
                )

        # Open position with 1:2 RR
        return self.open_position(
            symbol=symbol,
            side=signal,
            quantity=quantity,
            entry_price=signal_result.entry_price,
            stop_loss=signal_result.stop_loss,
            take_profit=signal_result.take_profit,
            leverage=str(lev),
        )
    
    def _scale_to_min_order(
        self,
        symbol: str,
        entry_price: float,
        balance: Optional[float],
        asset_info: Dict[str, Any],
        initial_lev: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Smart Scaling Fallback: try to meet min order value ($7).

        Strategy (in order):
        1. Scale leverage up to asset max_leverage (keep 2% margin).
        2. If still below min, increase margin up to 10% of balance (safety cap).
        3. Return dict with scaled {quantity, leverage, margin_used} or None if impossible.
        """
        min_val = getattr(self.trading_config, "min_order_value", 7.0)
        asset_max_lev = int(asset_info.get("max_leverage", 20))
        effective_max_lev = min(self.trading_config.leverage_max, asset_max_lev)
        qty_step = asset_info["quantity_step"]
        min_qty = asset_info["min_quantity"]

        if balance is None or balance <= 0 or entry_price <= 0:
            return None

        base_margin_pct = self.trading_config.margin_percent / 100.0
        max_margin_pct = 0.10  # safety cap: never use more than 10% of balance per trade

        # Step 1: Try scaling leverage (keep original margin %)
        margin = balance * base_margin_pct
        for try_lev in range(initial_lev, effective_max_lev + 1):
            raw_qty = (margin * try_lev) / entry_price
            qty = self.round_quantity(raw_qty, qty_step)
            if qty < min_qty:
                continue
            notional = qty * entry_price
            if notional >= min_val:
                if try_lev != initial_lev:
                    logger.info(
                        "%s: scaled leverage %dx -> %dx (asset max %dx) to meet min $%.0f (notional $%.2f)",
                        symbol, initial_lev, try_lev, asset_max_lev, min_val, notional,
                    )
                return {"quantity": qty, "leverage": try_lev, "margin_used": margin}

        # Step 2: At max leverage, increase margin up to 10% of balance
        lev = effective_max_lev
        required_notional = min_val
        required_qty = required_notional / entry_price
        required_margin = required_notional / lev
        margin_pct_needed = required_margin / balance

        if margin_pct_needed > max_margin_pct:
            # Even 10% of balance at max leverage can't reach min order value
            return None

        margin = required_margin * 1.05  # 5% buffer so rounding doesn't push us below
        margin = min(margin, balance * max_margin_pct)
        raw_qty = (margin * lev) / entry_price
        qty = self.round_quantity(raw_qty, qty_step)
        if qty < min_qty:
            return None
        notional = qty * entry_price
        if notional < min_val:
            return None

        actual_margin_pct = (margin / balance) * 100
        logger.info(
            "%s: scaled margin %.1f%% -> %.1f%% at %dx leverage to meet min $%.0f (notional $%.2f)",
            symbol, base_margin_pct * 100, actual_margin_pct, lev, min_val, notional,
        )
        return {"quantity": qty, "leverage": lev, "margin_used": margin}

    def execute_proposed_position(
        self,
        symbol: str,
        proposed_position: dict,
        balance: Optional[float] = None,
    ) -> ExecutionResult:
        """Open position from strategy_core proposed_position (quantity, leverage precomputed).

        Smart Scaling Fallback:
        - If notional < min_order_value, scale leverage up to asset max first.
        - If still below, increase margin up to 10% of balance (safety cap).
        - Only reject if scaling is impossible.
        """
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

        # Smart scaling if notional is below minimum
        if notional < min_val:
            scaled = self._scale_to_min_order(symbol, entry_price, balance, asset_info, lev)
            if scaled is not None:
                quantity = scaled["quantity"]
                lev = scaled["leverage"]
                notional = quantity * entry_price
            else:
                # Impossible even after scaling — notify via Telegram
                msg = (
                    f"Cannot open: notional ${notional:.2f} < min ${min_val:.0f}.\n"
                    f"Even at {effective_max_lev}x leverage and 10% margin, "
                    f"balance ${balance:.2f if balance else 0:.2f} is too low."
                )
                logger.warning("%s: %s", symbol, msg)
                return ExecutionResult(
                    success=False,
                    action="NONE",
                    symbol=symbol,
                    message=msg,
                    error="Order value below minimum after scaling",
                )

        # Final validation: quantity must meet asset min after all scaling
        if quantity < asset_info["min_quantity"]:
            return ExecutionResult(
                success=False,
                action="NONE",
                symbol=symbol,
                message=f"Quantity {quantity} below asset min {asset_info['min_quantity']}",
                error="Position too small",
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
            self._throttle()
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
        
        except RateLimitCooldownError as e:
            return ExecutionResult(
                success=False,
                action=action,
                symbol=symbol,
                message="Rate limited (cooldown)",
                error=str(e),
            )
        except MudrexRateLimitError as e:
            self._set_rate_limit_cooldown()
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
            self._throttle()
            positions = self.client.positions.list_open()
            exchange_position = None

            for pos in positions:
                if pos.symbol == symbol:
                    exchange_position = pos
                    break

            if exchange_position:
                self._throttle()
                self.client.positions.close(exchange_position.position_id)
                logger.info(f"Closed position {exchange_position.position_id}")
            
            del self._positions[symbol]
            
            return ExecutionResult(
                success=True,
                action="CLOSE",
                symbol=symbol,
                message="Position closed",
            )
        except RateLimitCooldownError as e:
            return ExecutionResult(
                success=False,
                action="CLOSE",
                symbol=symbol,
                message="Rate limited (cooldown)",
                error=str(e),
            )
        except MudrexRateLimitError as e:
            self._set_rate_limit_cooldown()
            return ExecutionResult(
                success=False,
                action="CLOSE",
                symbol=symbol,
                message="Rate limited",
                error=str(e),
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

        # Skip no-op: same value (avoids 400 from API)
        if abs(new_stop_loss - old_stop) < 1e-9:
            return ExecutionResult(
                success=True,
                action="UPDATE_TSL",
                symbol=symbol,
                message="TSL unchanged (same price)",
                position_state=position,
            )
        
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
            # Rate limit: space TSL updates like order calls
            delay = getattr(self.trading_config, "order_delay_seconds", 4.0)
            if delay > 0:
                time.sleep(delay)
            # Find position on exchange
            self._throttle()
            positions = self.client.positions.list_open()
            price_str = str(round(new_stop_loss, 4))
            for pos in positions:
                if pos.symbol == symbol:
                    risk_order_id = getattr(pos, "stoploss_order_id", None)
                    self._throttle()
                    try:
                        if risk_order_id:
                            self.client.positions.edit_risk_order(
                                position_id=pos.position_id,
                                stoploss_price=price_str,
                                risk_order_id=risk_order_id,
                            )
                        else:
                            # No existing risk order id (e.g. position from state or API didn't return it); try POST to set
                            ok = self.client.positions.set_risk_order(
                                position_id=pos.position_id,
                                stoploss_price=price_str,
                                takeprofit_price=str(round(position.take_profit, 4)),
                            )
                            if not ok:
                                raise MudrexAPIError("set_risk_order returned false")
                        position.stop_loss = new_stop_loss
                        return ExecutionResult(
                            success=True,
                            action="UPDATE_TSL",
                            symbol=symbol,
                            message=f"Updated TSL from {old_stop:.4f} to {new_stop_loss:.4f}",
                            position_state=position,
                        )
                    except MudrexAPIError as e:
                        if "risk order id missing" in str(e).lower() and not risk_order_id:
                            logger.warning("%s: position has no risk order id; skipping TSL update", symbol)
                            return ExecutionResult(
                                success=True,
                                action="UPDATE_TSL",
                                symbol=symbol,
                                message="TSL skip (no risk order id on exchange)",
                                position_state=position,
                            )
                        raise
            
            return ExecutionResult(
                success=False,
                action="UPDATE_TSL",
                symbol=symbol,
                message="Position not found on exchange",
            )
        except RateLimitCooldownError as e:
            return ExecutionResult(
                success=False,
                action="UPDATE_TSL",
                symbol=symbol,
                message="Rate limited (cooldown)",
                error=str(e),
            )
        except MudrexRateLimitError as e:
            self._set_rate_limit_cooldown()
            return ExecutionResult(
                success=False,
                action="UPDATE_TSL",
                symbol=symbol,
                message="Rate limited",
                error=str(e),
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
