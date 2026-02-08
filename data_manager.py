"""
Bybit Data Manager
==================

Manages real-time OHLCV data using Bybit WebSocket (V5).
Subscribes to kline streams for symbols and maintains a cache of recent candles.
Backfills historical klines via REST so the strategy has enough bars from cycle 1.
"""

import time
import logging
import threading
from typing import Dict, List, Optional
import pandas as pd
from pybit.unified_trading import WebSocket, HTTP

logger = logging.getLogger("data_manager")

class DataManager:
    """
    Manages OHLCV data using Bybit WebSocket.
    """
    
    def __init__(self, interval: str = "5", lookback: int = 200):
        self.interval = interval  # Bybit kline interval (e.g., '1', '5', '15', '60', 'D')
        self.lookback = lookback
        self.data: Dict[str, pd.DataFrame] = {}
        self.ws: Optional[WebSocket] = None
        self._http: Optional[HTTP] = None  # REST for backfill only (public, no key)
        self._lock = threading.Lock()
        self.active_symbols: set = set()

    def _http_session(self) -> HTTP:
        if self._http is None:
            self._http = HTTP(testnet=False)
        return self._http

    def _backfill(self, symbols: List[str]) -> None:
        """Fetch last N candles via REST so we have enough data for strategy from cycle 1."""
        limit = min(200, max(self.lookback, 20))
        ok = 0
        partial = 0  # symbols with some but < limit candles
        failed = 0
        min_bars_required = 20
        for i, symbol in enumerate(symbols):
            try:
                r = self._http_session().get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=self.interval,
                    limit=limit,
                )
                result = (r or {}).get("result", {})
                rows = result.get("list") or []
                if not rows:
                    failed += 1
                    continue
                # Bybit returns newest first; we want oldest first
                rows = list(reversed(rows))
                if len(rows) < min_bars_required:
                    partial += 1
                # Each row: [startTime, open, high, low, close, volume, turnover]
                data = []
                for row in rows:
                    ts = int(row[0])
                    data.append({
                        "timestamp": pd.to_datetime(ts, unit="ms"),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    })
                df = pd.DataFrame(data).set_index("timestamp")
                with self._lock:
                    self.data[symbol] = df
                ok += 1
            except Exception as e:
                failed += 1
                logger.debug(f"Backfill {symbol}: {e}")
            if (i + 1) % 100 == 0:
                logger.info(f"Backfill progress: {i + 1}/{len(symbols)} symbols")
            time.sleep(0.10)  # ~10 req/s to stay under Bybit REST rate limits
        # Coverage and gap metrics
        logger.info(
            f"Backfill complete: {ok}/{len(symbols)} symbols, {limit} candles requested; "
            f"partial(<{min_bars_required} bars)={partial}, failed={failed}"
        )
        if failed > 0:
            logger.warning(f"Backfill failed for {failed} symbols (no data or API error)")
        
    def start(self):
        """Start the Bybit WebSocket connection."""
        if self.ws:
            return
            
        logger.info("Starting Bybit WebSocket (Linear Kline)...")
        self.ws = WebSocket(
            testnet=False,
            channel_type="linear",
        )
        
    def stop(self):
        """Stop the WebSocket connection."""
        if self.ws:
            self.ws.exit()
            self.ws = None
            logger.info("Bybit WebSocket stopped")

    def subscribe(self, symbols: List[str]):
        """Subscribe to kline streams for a list of symbols."""
        if not self.ws:
            self.start()

        new_symbols = [s for s in symbols if s not in self.active_symbols]
        if not new_symbols:
            return

        logger.info(f"Subscribing to Bybit kline.{self.interval} for {len(new_symbols)} symbols (batch)")
        # Subscribe once with full list so pybit registers callback per topic (not overwritten)
        self.ws.kline_stream(
            interval=self.interval,
            symbol=new_symbols,
            callback=self._handle_kline,
        )
        with self._lock:
            for symbol in new_symbols:
                if symbol not in self.data:
                    self.data[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.active_symbols.add(symbol)

        # Backfill historical klines so we have 20+ bars from first cycle (no ~100 min wait)
        self._backfill(new_symbols)
                
    def _handle_kline(self, message: Dict):
        """Callback for WebSocket kline messages."""
        try:
            if "data" not in message:
                return

            for kline in message["data"]:
                symbol = message["topic"].split(".")[-1]
                
                # Extract data
                ts = int(kline["start"])
                dto = pd.to_datetime(ts, unit='ms')
                
                new_row = {
                    'timestamp': dto,
                    'open': float(kline["open"]),
                    'high': float(kline["high"]),
                    'low': float(kline["low"]),
                    'close': float(kline["close"]),
                    'volume': float(kline["volume"]),
                }
                
                with self._lock:
                    df = self.data.get(symbol)
                    if df is None or df.empty:
                        self.data[symbol] = pd.DataFrame([new_row]).set_index('timestamp')
                    else:
                        # Update existing row or append
                        if dto in df.index:
                            df.loc[dto] = [new_row['open'], new_row['high'], new_row['low'], new_row['close'], new_row['volume']]
                        else:
                            df.loc[dto] = [new_row['open'], new_row['high'], new_row['low'], new_row['close'], new_row['volume']]
                            # Keep lookback
                            if len(df) > self.lookback:
                                df = df.iloc[-self.lookback:]
                            self.data[symbol] = df
                            
        except Exception as e:
            logger.error(f"Error handling kline message: {e}")

    def get_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached OHLCV data for a symbol."""
        with self._lock:
            df = self.data.get(symbol)
            if df is not None and not df.empty:
                return df.copy()
            return None

    def wait_for_data(self, symbols: List[str], timeout: int = 30):
        """Wait until data is populated for the given symbols."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            missing = False
            with self._lock:
                for s in symbols:
                    if s not in self.data or len(self.data[s]) < 2:  # Need at least 2 candles for Supertrend
                        missing = True
                        break
            if not missing:
                return True
            time.sleep(1)
        return False

    def get_data_coverage(self, symbols: List[str], min_bars: int = 20) -> dict:
        """Return coverage metrics for observability: symbols with sufficient bars and total."""
        with self._lock:
            sufficient = sum(
                1 for s in symbols
                if s in self.data and len(self.data[s]) >= min_bars
            )
        return {"symbols_with_sufficient_data": sufficient, "total": len(symbols)}
