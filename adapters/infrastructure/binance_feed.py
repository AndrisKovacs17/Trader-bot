from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from itertools import islice
import logging
from typing import Awaitable, Callable

import pandas as pd
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager

from core.domain.events import MarketDataEvent


logger = logging.getLogger(__name__)


def _utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _parse_time_window(start_str: str) -> tuple[int, int]:
    """Parse coarse time windows like '3 days ago UTC' or '12 hours ago UTC'."""
    end = datetime.now(timezone.utc)
    tokens = str(start_str).strip().lower().split()
    if len(tokens) >= 3 and tokens[2] == "ago":
        try:
            amount = int(tokens[0])
        except ValueError:
            amount = 1
        unit = tokens[1]
        if unit.startswith("day"):
            start = end - timedelta(days=amount)
        elif unit.startswith("hour"):
            start = end - timedelta(hours=amount)
        elif unit.startswith("min"):
            start = end - timedelta(minutes=amount)
        else:
            start = end - timedelta(days=1)
    else:
        # Safe default window if format is unknown.
        start = end - timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


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


def get_aggregate_trades_time_window(
    symbol: str = "BTCUSDT",
    start_ms: int | None = None,
    end_ms: int | None = None,
    start_str: str | None = None,
    max_trades: int = 200000,
) -> pd.DataFrame:
    """Fetch aggregate trades over a real time window (not just first N after start)."""
    client = Client()

    if start_ms is None or end_ms is None:
        parsed_start, parsed_end = _parse_time_window(start_str or "1 day ago UTC")
        if start_ms is None:
            start_ms = parsed_start
        if end_ms is None:
            end_ms = parsed_end

    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if start_ms >= end_ms:
        return pd.DataFrame(columns=["T", "p", "q"])

    trades: list[dict] = []
    from_id: int | None = None
    cursor_ms = start_ms

    while cursor_ms <= end_ms and len(trades) < max_trades:
        remaining = max_trades - len(trades)
        limit = max(1, min(1000, remaining))
        if from_id is not None:
            # Binance aggTrades endpoint rejects fromId mixed with time bounds.
            params: dict[str, int | str] = {
                "symbol": symbol,
                "limit": limit,
                "fromId": from_id,
            }
        else:
            params = {
                "symbol": symbol,
                "limit": limit,
                "startTime": cursor_ms,
                "endTime": end_ms,
            }

        batch = client.get_aggregate_trades(**params)
        if not batch:
            break

        appended = 0
        for trade in batch:
            trade_ts = int(trade.get("T", 0))
            if trade_ts < start_ms:
                continue
            if trade_ts > end_ms:
                continue
            trades.append(trade)
            appended += 1

        last = batch[-1]
        last_id = int(last.get("a", 0))
        last_ts = int(last.get("T", cursor_ms))
        from_id = last_id + 1
        cursor_ms = max(cursor_ms + 1, last_ts + 1)

        if appended == 0 and last_ts >= end_ms:
            break
        if len(batch) < limit:
            break

    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame(columns=["T", "p", "q"])

    df = df[["T", "p", "q"]].copy()
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    return df.dropna(subset=["T", "p", "q"]).sort_values("T").reset_index(drop=True)


def get_historical_klines_time_window(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_str: str = "1 day ago UTC",
    end_str: str | None = None,
    max_bars: int = 200000,
) -> pd.DataFrame:
    """Fetch historical klines and return a normalized close-price dataframe.

    Returned columns:
    - T: candle close time (UTC timestamp)
    - p: candle close price
    - q: candle volume
    """
    client = Client()
    raw = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str,
    )
    if not raw:
        return pd.DataFrame(columns=["T", "p", "q"])

    if max_bars > 0 and len(raw) > max_bars:
        raw = raw[-max_bars:]

    rows: list[dict[str, int | float]] = []
    for candle in raw:
        if len(candle) < 7:
            continue
        close_ts = int(candle[6])
        close_price = float(candle[4])
        volume = float(candle[5])
        rows.append({"T": close_ts, "p": close_price, "q": volume})

    if not rows:
        return pd.DataFrame(columns=["T", "p", "q"])

    df = pd.DataFrame(rows)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    return df.dropna(subset=["T", "p", "q"]).sort_values("T").reset_index(drop=True)


def build_fixed_bars_from_trades(trades_df: pd.DataFrame, bar_seconds: int = 1) -> pd.DataFrame:
    """Convert trade ticks to fixed-width bars using time-based resampling."""
    if trades_df.empty:
        return pd.DataFrame(columns=["T", "p", "q"])

    sec = max(int(bar_seconds), 1)
    freq = f"{sec}s"

    df = trades_df.copy()
    df["T"] = pd.to_datetime(df["T"], utc=True)
    df = df.sort_values("T")
    df = df.set_index("T")

    bars = pd.DataFrame()
    bars["p"] = df["p"].resample(freq).last()
    bars["q"] = df["q"].resample(freq).sum()
    bars = bars.dropna(subset=["p"])
    bars["q"] = bars["q"].fillna(0.0)
    bars = bars.reset_index()
    return bars[["T", "p", "q"]]


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
            stream_start = asyncio.get_running_loop().time()
            message_count = 0
            
            try:
                # Create client and socket manager
                if self.client is None:
                    self.client = await AsyncClient.create()
                    self.bm = BinanceSocketManager(self.client, max_queue_size=self.max_queue_size)
                    if hasattr(self.bm, "ws_kwargs") and isinstance(self.bm.ws_kwargs, dict):
                        ws_extra: dict = {"ping_interval": 20, "ping_timeout": 20}
                        if self.websocket_max_queue is not None:
                            ws_extra["max_queue"] = self.websocket_max_queue
                        self.bm.ws_kwargs.update(ws_extra)
                
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
                            elapsed = asyncio.get_running_loop().time() - stream_start
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
                elapsed = asyncio.get_running_loop().time() - stream_start
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


def get_historical_funding_rates(
    symbol: str = "BTCUSDT",
    start_str: str = "90 days ago UTC",
    max_rows: int = 5000,
) -> dict[int, float]:
    """Fetch historical funding rates from Binance Futures (USDM).

    Returns a dict mapping timestamp_ms → funding_rate (float).
    Funding is settled every 8 hours (3 values/day).  If the symbol is not a
    perpetual contract, or the endpoint is unavailable, returns an empty dict
    so callers can fall back gracefully.
    """
    try:
        from binance.client import Client as _Client
        client = _Client()
        parsed_start, parsed_end = _parse_time_window(start_str)
        result: dict[int, float] = {}
        cursor = parsed_start
        while cursor < parsed_end and len(result) < max_rows:
            batch = client.futures_funding_rate(
                symbol=symbol,
                startTime=cursor,
                endTime=parsed_end,
                limit=min(1000, max_rows - len(result)),
            )
            if not batch:
                break
            for entry in batch:
                ts = int(entry.get("fundingTime", 0))
                rate = float(entry.get("fundingRate", 0.0))
                result[ts] = rate
            # Advance cursor past last entry
            last_ts = int(batch[-1].get("fundingTime", cursor))
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            if len(batch) < 1000:
                break
        return result
    except Exception:
        return {}


def get_historical_klines_with_taker(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_str: str = "90 days ago UTC",
    max_bars: int = 30000,
) -> pd.DataFrame:
    """Like get_historical_klines_time_window but also returns taker buy volume.

    Kline API field indices:
      [0] open_time, [1] open, [2] high, [3] low, [4] close, [5] volume,
      [6] close_time, [7] quote_asset_volume, [8] num_trades,
      [9] taker_buy_base_volume, [10] taker_buy_quote_volume

    Extra returned columns beyond T/p/q:
      - taker_buy_vol  : taker buy base asset volume
      - num_trades     : number of trades in the bar
    """
    from binance.client import Client as _Client
    client = _Client()
    raw = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
    )
    if not raw:
        return pd.DataFrame(columns=["T", "o", "h", "l", "p", "q", "taker_buy_vol", "num_trades"])
    if max_bars > 0 and len(raw) > max_bars:
        raw = raw[-max_bars:]

    rows: list[dict] = []
    for candle in raw:
        if len(candle) < 11:
            continue
        rows.append({
            "T": int(candle[6]),
            "o": float(candle[1]),
            "h": float(candle[2]),
            "l": float(candle[3]),
            "p": float(candle[4]),
            "q": float(candle[5]),
            "taker_buy_vol": float(candle[9]),
            "num_trades": int(candle[8]),
        })
    if not rows:
        return pd.DataFrame(columns=["T", "o", "h", "l", "p", "q", "taker_buy_vol", "num_trades"])

    df = pd.DataFrame(rows)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    return df.dropna(subset=["T", "p", "q"]).sort_values("T").reset_index(drop=True)


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

