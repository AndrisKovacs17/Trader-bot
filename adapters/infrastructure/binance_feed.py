from __future__ import annotations

import asyncio
from datetime import datetime
from itertools import islice
import logging
from typing import Awaitable, Callable

import pandas as pd
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager

from core.domain.events import MarketDataEvent


logger = logging.getLogger(__name__)


# LIVE: Legfrissebb trade-ek (utolsó pár perc, max 1000 trade)
def get_latest_trades(
    symbol: str = "BTCUSDT",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Valós idejű (live) trade-ek letöltése Binance-ről.
    Ez a legfrissebb trade-eket adja vissza (utolsó pár perc).
    
    Args:
        symbol: Trading pair (pl. BTCUSDT)
        limit: Max trade-ek száma (max 1000)
    
    Returns:
        DataFrame oszlopokkal: T (timestamp), p (price), q (quantity)
    """
    client = Client()
    trades = client.get_recent_trades(symbol=symbol, limit=min(limit, 1000))
    
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame(columns=["T", "p", "q"])
    
    # Rename columns to match expected format
    df = df.rename(columns={"time": "T", "price": "p", "qty": "q"})[["T", "p", "q"]].copy()
    df["T"] = pd.to_datetime(df["T"], unit="ms")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    return df.dropna(subset=["T", "p", "q"]).reset_index(drop=True)


# Binance aggregate trade letöltés DataFrame-be (historical data).
def get_recent_aggregate_trades(
    symbol: str = "BTCUSDT",
    start_str: str = "2 hours ago UTC",
    max_trades: int = 500,
) -> pd.DataFrame:
    # Publikus endpoint, így API kulcs nélkül is működik a legtöbb esetben.
    client = Client()
    trades_iter = client.aggregate_trade_iter(symbol=symbol, start_str=start_str)
    trades = list(islice(trades_iter, max_trades))

    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame(columns=["T", "p", "q"])

    df = df[["T", "p", "q"]].copy()
    df["T"] = pd.to_datetime(df["T"], unit="ms")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    return df.dropna(subset=["T", "p", "q"]).reset_index(drop=True)


# Egyszerű OHLC-szerű árpontok előállítása a pipeline számára.
def to_price_points(trades_df: pd.DataFrame) -> list[tuple[datetime, float]]:
    if trades_df.empty:
        return []
    return list(zip(trades_df["T"].tolist(), trades_df["p"].tolist(), strict=False))


# =====================================================
# BINANCE WEBSOCKET STREAM (REAL-TIME)
# =====================================================

class BinanceWebSocketStream:
    """
    Binance WebSocket stream for real-time market data.
    Continuously streams live trades/klines and calls callback for each update.
    AUTO-RECONNECTS on disconnect with exponential backoff.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        stream_type: str = "aggTrade",
        max_reconnects: int = 10,
        max_queue_size: int = 5000,
        websocket_max_queue: int | None = None,
    ):
        """
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            stream_type: Stream type options:
                - 'aggTrade' (aggregated trades, more stable, RECOMMENDED)
                - 'trade' (individual trades, very high frequency)
                - 'kline_1m' (1-minute candles, lowest frequency)
                - 'kline_1s' (1-second candles, if available)
            max_reconnects: Maximum reconnection attempts before giving up
            max_queue_size: Internal python-binance queue size (default 5000)
            websocket_max_queue: websockets transport queue size (None = unbounded)
        """
        self.symbol = symbol.lower()
        self.stream_type = stream_type
        self.max_reconnects = max_reconnects
        self.max_queue_size = max_queue_size
        self.websocket_max_queue = websocket_max_queue
        self.client: AsyncClient | None = None
        self.bm: BinanceSocketManager | None = None
        self.running = False
    
    async def start(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        """
        Start WebSocket stream with auto-reconnect on failure.
        
        Args:
            callback: Async function to call with each message: callback(data: dict)
        """
        self.running = True
        reconnect_count = 0
        backoff_seconds = 1
        
        while self.running and reconnect_count < self.max_reconnects:
            stream_start = asyncio.get_event_loop().time()
            message_count = 0
            
            try:
                # Create client and socket manager
                if self.client is None:
                    self.client = await AsyncClient.create()
                    self.bm = BinanceSocketManager(self.client, max_queue_size=self.max_queue_size)
                    self.bm.ws_kwargs.update(
                        {
                            "max_queue": self.websocket_max_queue,
                            "ping_interval": 20,
                            "ping_timeout": 20,
                        }
                    )
                
                # Select stream type
                if self.stream_type == "trade":
                    socket = self.bm.trade_socket(self.symbol)
                elif self.stream_type == "aggTrade":
                    socket = self.bm.aggtrade_socket(self.symbol)
                elif self.stream_type.startswith("kline"):
                    interval = self.stream_type.split("_")[1] if "_" in self.stream_type else "1m"
                    socket = self.bm.kline_socket(self.symbol, interval=interval)
                else:
                    raise ValueError(f"Unknown stream_type: {self.stream_type}. "
                                   f"Use 'aggTrade', 'trade', or 'kline_1m'")
                
                logger.info("[WebSocket] Connecting to Binance stream... (attempt %s)", reconnect_count + 1)
                
                # Start streaming
                async with socket as stream:
                    logger.info("[WebSocket] ✓ Connected! Streaming %s %s", self.symbol, self.stream_type)
                    reconnect_count = 0  # Reset on successful connect
                    backoff_seconds = 1   # Reset backoff
                    
                    while self.running:
                        try:
                            # Timeout after 60s to detect dead connections
                            msg = await asyncio.wait_for(stream.recv(), timeout=60.0)
                            
                            if msg:
                                message_count += 1
                                try:
                                    await callback(msg)
                                except Exception as callback_error:
                                    # Callback error - log and continue (don't break stream)
                                    logger.warning("[WebSocket] Callback error: %s", callback_error)
                                    await asyncio.sleep(0.01)
                        
                        except asyncio.TimeoutError:
                            # No message for 60s - check if still connected
                            elapsed = asyncio.get_event_loop().time() - stream_start
                            logger.warning(
                                "[WebSocket] ⚠ Timeout (no message for 60s). Uptime: %.1fs, Messages: %s",
                                elapsed,
                                message_count,
                            )
                            # Let the stream continue (might be low volume period)
                            continue
            
            except asyncio.CancelledError:
                logger.info("[WebSocket] Cancelled by user")
                break
            
            except Exception as stream_error:
                # Connection/stream error - attempt reconnect
                elapsed = asyncio.get_event_loop().time() - stream_start
                error_type = type(stream_error).__name__
                
                reconnect_count += 1
                logger.error("[WebSocket] ✗ Stream error after %.1fs (%s msgs)", elapsed, message_count)
                logger.error("[WebSocket]   Error type: %s", error_type)
                logger.error("[WebSocket]   Error message: %s", stream_error)
                
                if reconnect_count >= self.max_reconnects:
                    logger.error("[WebSocket] Max reconnects (%s) reached. Giving up.", self.max_reconnects)
                    break
                
                # Cleanup before reconnect
                if self.client:
                    try:
                        await self.client.close_connection()
                    except:
                        pass
                    self.client = None
                    self.bm = None
                
                # Exponential backoff
                logger.info("[WebSocket] Reconnecting in %ss...", backoff_seconds)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 30)  # Max 30s backoff
        
        # Final cleanup
        await self.stop()
    
    async def stop(self) -> None:
        """Stop WebSocket stream and cleanup."""
        self.running = False
        if self.client:
            try:
                await self.client.close_connection()
            except Exception as e:
                logger.warning("[WebSocket] Cleanup error: %s", e)
            finally:
                self.client = None
        self.bm = None
        logger.info("[WebSocket] Stopped")


def parse_trade_message(msg: dict) -> dict | None:
    """
    Parse Binance trade/aggTrade WebSocket message to standardized format.
    Supports both 'trade' and 'aggTrade' event types.
    
    Returns:
        {"symbol": "BTCUSDT", "price": 50000.0, "qty": 0.01, "time": timestamp}
    """
    event_type = msg.get("e")
    
    if event_type == "aggTrade":
        # Aggregated trade format
        return {
            "symbol": msg.get("s", ""),
            "price": float(msg.get("p", 0.0)),
            "qty": float(msg.get("q", 0.0)),
            "time": int(msg.get("T", 0)),  # Trade time in ms
            "trade_id": msg.get("a", 0),   # Aggregate trade ID
            "event": "aggTrade",
        }
    elif event_type == "trade":
        # Individual trade format
        return {
            "symbol": msg.get("s", ""),
            "price": float(msg.get("p", 0.0)),
            "qty": float(msg.get("q", 0.0)),
            "time": int(msg.get("T", 0)),  # Trade time in ms
            "trade_id": msg.get("t", 0),
            "event": "trade",
        }
    
    return None


def parse_kline_message(msg: dict) -> dict | None:
    """
    Parse Binance kline (candlestick) WebSocket message.
    
    Returns:
        {"symbol": "BTCUSDT", "price": 50000.0 (close), "time": timestamp, ...}
    """
    if msg.get("e") != "kline":
        return None
    
    kline = msg.get("k", {})
    return {
        "symbol": msg.get("s", ""),
        "price": float(kline.get("c", 0.0)),  # Close price
        "open": float(kline.get("o", 0.0)),
        "high": float(kline.get("h", 0.0)),
        "low": float(kline.get("l", 0.0)),
        "volume": float(kline.get("v", 0.0)),
        "time": int(kline.get("T", 0)),  # Close time in ms
        "is_closed": kline.get("x", False),  # Is candle closed?
    }


def to_market_data_event(parsed: dict, source: str = "binance-websocket") -> MarketDataEvent | None:
    """Convert parsed trade/kline payload into a typed MarketDataEvent."""
    symbol = str(parsed.get("symbol", "")).strip()
    if not symbol:
        return None
    return MarketDataEvent(
        payload={
            "symbol": symbol,
            "price": float(parsed.get("price", 0.0)),
            "qty": float(parsed.get("qty", 0.0)),
            "timestamp": int(parsed.get("time", 0)),
        },
        source=source,
        correlation_id=f"ws-trade-{parsed.get('trade_id', 'na')}",
    )

