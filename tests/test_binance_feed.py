#!/usr/bin/env python3
"""Regression tests for Binance aggregate trade window pagination."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.infrastructure import binance_feed


def _mk_trade(trade_id: int, ts_ms: int, price: str = "100.0", qty: str = "0.1") -> dict:
    return {"a": trade_id, "T": ts_ms, "p": price, "q": qty}


def test_time_window_fetch_does_not_mix_fromid_with_time_bounds(monkeypatch) -> None:
    captured_calls: list[dict[str, int | str]] = []

    first_batch = [
        _mk_trade(1, 900),
        _mk_trade(2, 920),
        _mk_trade(3, 940),
        _mk_trade(4, 960),
        _mk_trade(5, 980),
        _mk_trade(6, 1000),
    ]
    second_batch = [_mk_trade(7, 1010)]

    class FakeClient:
        def __init__(self) -> None:
            self._calls = 0

        def get_aggregate_trades(self, **params):
            self._calls += 1
            captured_calls.append(dict(params))
            if self._calls == 1:
                return first_batch
            if self._calls == 2:
                return second_batch
            return []

    monkeypatch.setattr(binance_feed, "Client", FakeClient)

    df = binance_feed.get_aggregate_trades_time_window(
        symbol="BTCUSDT",
        start_ms=1000,
        end_ms=1050,
        max_trades=6,
    )

    assert len(captured_calls) >= 2
    first = captured_calls[0]
    second = captured_calls[1]

    assert "startTime" in first
    assert "endTime" in first
    assert "fromId" not in first

    assert "fromId" in second
    assert "startTime" not in second
    assert "endTime" not in second

    assert list(df.columns) == ["T", "p", "q"]
    assert len(df) == 2


def test_time_window_fetch_filters_out_of_window_and_sorts(monkeypatch) -> None:
    # Out-of-window values should be filtered even if exchange returns them.
    batch = [
        _mk_trade(10, 999),
        _mk_trade(11, 1001, "101.0", "0.2"),
        _mk_trade(12, 1000, "100.5", "0.3"),
        _mk_trade(13, 1002, "101.5", "0.4"),
        _mk_trade(14, 2000),
    ]

    class FakeClient:
        def get_aggregate_trades(self, **params):
            return batch

    monkeypatch.setattr(binance_feed, "Client", FakeClient)

    df = binance_feed.get_aggregate_trades_time_window(
        symbol="BTCUSDT",
        start_ms=1000,
        end_ms=1002,
        max_trades=5,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    timestamps = [int(getattr(ts, "value", 0) // 1_000_000) for ts in df["T"].tolist()]
    assert timestamps == [1000, 1001, 1002]


def _mk_kline(close_time_ms: int, close_price: str, volume: str) -> list:
    # Binance kline shape: [openTime, open, high, low, close, volume, closeTime, ...]
    return [
        close_time_ms - 60000,
        "0",
        "0",
        "0",
        close_price,
        volume,
        close_time_ms,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


def test_historical_klines_window_maps_close_and_volume(monkeypatch) -> None:
    sample = [
        _mk_kline(1200, "101.5", "0.25"),
        _mk_kline(1000, "100.0", "0.50"),
        _mk_kline(1100, "100.8", "0.40"),
    ]

    class FakeClient:
        def get_historical_klines(self, **params):
            return sample

    monkeypatch.setattr(binance_feed, "Client", FakeClient)

    df = binance_feed.get_historical_klines_time_window(
        symbol="BTCUSDT",
        interval="1m",
        start_str="1 day ago UTC",
        max_bars=100,
    )

    assert list(df.columns) == ["T", "p", "q"]
    assert len(df) == 3
    timestamps = [int(getattr(ts, "value", 0) // 1_000_000) for ts in df["T"].tolist()]
    assert timestamps == [1000, 1100, 1200]
    assert [float(v) for v in df["p"].tolist()] == [100.0, 100.8, 101.5]
    assert [float(v) for v in df["q"].tolist()] == [0.5, 0.4, 0.25]


def test_historical_klines_window_respects_max_bars_tail(monkeypatch) -> None:
    sample = [
        _mk_kline(1000, "100.0", "1.0"),
        _mk_kline(2000, "101.0", "1.0"),
        _mk_kline(3000, "102.0", "1.0"),
        _mk_kline(4000, "103.0", "1.0"),
    ]

    class FakeClient:
        def get_historical_klines(self, **params):
            return sample

    monkeypatch.setattr(binance_feed, "Client", FakeClient)

    df = binance_feed.get_historical_klines_time_window(
        symbol="BTCUSDT",
        interval="1m",
        start_str="1 day ago UTC",
        max_bars=2,
    )

    assert len(df) == 2
    timestamps = [int(getattr(ts, "value", 0) // 1_000_000) for ts in df["T"].tolist()]
    assert timestamps == [3000, 4000]
