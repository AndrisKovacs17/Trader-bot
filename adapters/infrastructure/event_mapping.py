from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Iterator
from uuid import uuid4

import pandas as pd

from core.domain.events import MarketDataEvent, NewsSentimentEvent


def _normalize_utc_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e11:
            numeric = numeric / 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


# News DataFrame -> Event lista.
def news_df_to_events(news_df: pd.DataFrame, correlation_prefix: str = "news") -> list[NewsSentimentEvent]:
    return list(iter_news_df_to_events(news_df=news_df, correlation_prefix=correlation_prefix))


def iter_news_df_to_events(news_df: pd.DataFrame, correlation_prefix: str = "news") -> Iterator[NewsSentimentEvent]:
    for _, row in news_df.iterrows():
        raw_time = row.get("Time")
        event_time = _normalize_utc_timestamp(raw_time)
        yield NewsSentimentEvent(
            payload={
                "time": raw_time,
                "title": row.get("Title"),
                "source": row.get("Source"),
                "sentiment_score": row.get("Sentiment_Score"),
                "sentiment": row.get("Sentiment"),
            },
            source="news-feed",
            correlation_id=f"{correlation_prefix}-{uuid4()}",
            ts_event=event_time,
        )


# Price pontok -> MarketData event lista.
def price_points_to_market_events(
    points: Iterable[tuple],
    symbol: str = "BTCUSDT",
    correlation_id: str = "binance-live",
) -> list[MarketDataEvent]:
    return list(
        iter_price_points_to_market_events(
            points=points,
            symbol=symbol,
            correlation_id=correlation_id,
        )
    )


def iter_price_points_to_market_events(
    points: Iterable[tuple],
    symbol: str = "BTCUSDT",
    correlation_id: str = "binance-live",
) -> Iterator[MarketDataEvent]:
    for ts, price in points:
        yield MarketDataEvent(
            payload={"symbol": symbol, "price": float(price)},
            source="binance",
            correlation_id=f"{correlation_id}-{uuid4()}",
            ts_event=_normalize_utc_timestamp(ts),
        )
