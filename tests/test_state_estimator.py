#!/usr/bin/env python3
"""Unit tests for KCAStateEstimator in core/ml/services.py.

Targets the state-estimation logic that is currently uncovered:
  - update() with various payload shapes
  - reset() / get_state()
  - Prediction / EstimatedState value objects
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.time_source import SimulatedTimeSource
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.models import Instrument
from core.ml.services import EstimatedState, KCAStateEstimator, Prediction


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ctx(symbol: str = "BTCUSDT") -> RunContext:
    instr = Instrument(symbol=symbol)
    wallet = SimulationWallet(initial_cash=10_000.0)
    wallet.mark_price(instr, 50_000.0)
    return RunContext(
        correlation_id="test",
        instrument=instr,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=10,
    )


def _market_event(symbol: str = "BTCUSDT", price: float = 50_000.0, qty: float = 1.5,
                  ts_ms: int | None = None):
    from core.domain.events import MarketDataEvent
    payload: dict = {"price": price, "qty": qty}
    if ts_ms is not None:
        payload["timestamp"] = ts_ms
    return MarketDataEvent(
        event_type="MarketData",
        payload={"symbol": symbol, **payload},
    )


# ── EstimatedState value object ───────────────────────────────────────────────

def test_estimated_state_fields() -> None:
    state = EstimatedState(x=[1.0, 2.0], P=[[1.0]], features={"price": 100.0}, confidence=0.9)
    assert state.x == [1.0, 2.0]
    assert state.confidence == 0.9
    assert state.features["price"] == 100.0
    assert state.ts is not None


# ── Prediction value object ───────────────────────────────────────────────────

def test_prediction_fields() -> None:
    pred = Prediction(mu=0.01, sigma=0.005, prob_up=0.65, regime="trend", horizon=12)
    assert pred.mu == 0.01
    assert pred.prob_up == 0.65
    assert pred.regime == "trend"
    assert pred.confirm_score == 0.0


# ── KCAStateEstimator.update ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_returns_estimated_state() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    event = _market_event(price=50_000.0, qty=2.0)
    state = await estimator.update(event, ctx)
    assert isinstance(state, EstimatedState)
    assert state.features["price"] == 50_000.0
    assert state.features["volume"] == 2.0


@pytest.mark.asyncio
async def test_update_computes_return_zero_for_first_tick() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    state = await estimator.update(_market_event(price=100.0), ctx)
    assert state.features["return"] == 0.0


@pytest.mark.asyncio
async def test_update_computes_return_for_second_tick() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    await estimator.update(_market_event(price=100.0), ctx)
    state2 = await estimator.update(_market_event(price=110.0), ctx)
    assert abs(state2.features["return"] - 0.1) < 1e-9


@pytest.mark.asyncio
async def test_update_handles_explicit_timestamp() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    ts_ms = 1_700_000_000_000
    state = await estimator.update(_market_event(ts_ms=ts_ms), ctx)
    assert state.features["timestamp_ms"] == ts_ms


@pytest.mark.asyncio
async def test_update_falls_back_to_event_ts_when_payload_ts_zero() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    state = await estimator.update(_market_event(ts_ms=0), ctx)
    # Should fall back to event datetime-based ms value (non-zero)
    assert state.features["timestamp_ms"] > 0


@pytest.mark.asyncio
async def test_update_stores_ohlc_from_payload() -> None:
    from core.domain.events import MarketDataEvent
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    payload = {"price": 500.0, "qty": 1.0, "o": 490.0, "h": 510.0, "l": 480.0}
    event = MarketDataEvent(event_type="MarketData", payload=payload)
    state = await estimator.update(event, ctx)
    assert state.features["h"] >= 510.0
    assert state.features["l"] <= 480.0
    assert state.features["o"] == 490.0


@pytest.mark.asyncio
async def test_update_taker_buy_vol_field() -> None:
    from core.domain.events import MarketDataEvent
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    payload = {"price": 50_000.0, "qty": 1.0, "V": 0.75}
    event = MarketDataEvent(event_type="MarketData", payload=payload)
    state = await estimator.update(event, ctx)
    assert state.features["taker_buy_vol"] == 0.75


@pytest.mark.asyncio
async def test_update_taker_buy_vol_minus_one_when_absent() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    state = await estimator.update(_market_event(price=50_000.0, qty=1.0), ctx)
    assert state.features["taker_buy_vol"] == -1.0


# ── KCAStateEstimator.reset / get_state ───────────────────────────────────────

@pytest.mark.asyncio
async def test_get_state_after_update() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    await estimator.update(_market_event(price=42_000.0), ctx)
    state = estimator.get_state(ctx.instrument)
    assert state.features["price"] == 42_000.0


@pytest.mark.asyncio
async def test_reset_removes_state() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    await estimator.update(_market_event(price=42_000.0), ctx)
    estimator.reset(ctx.instrument)
    assert "BTCUSDT" not in estimator.models


@pytest.mark.asyncio
async def test_get_state_raises_after_reset() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    await estimator.update(_market_event(price=1_000.0), ctx)
    estimator.reset(ctx.instrument)
    with pytest.raises(KeyError):
        estimator.get_state(ctx.instrument)


@pytest.mark.asyncio
async def test_multiple_symbols_isolated() -> None:
    estimator = KCAStateEstimator()
    ctx_btc = _make_ctx("BTCUSDT")
    ctx_eth = _make_ctx("ETHUSDT")
    await estimator.update(_market_event(symbol="BTCUSDT", price=50_000.0), ctx_btc)
    await estimator.update(_market_event(symbol="ETHUSDT", price=3_000.0), ctx_eth)
    assert estimator.models["BTCUSDT"].features["price"] == 50_000.0
    assert estimator.models["ETHUSDT"].features["price"] == 3_000.0


@pytest.mark.asyncio
async def test_reset_unknown_symbol_is_no_op() -> None:
    estimator = KCAStateEstimator()
    instr = Instrument(symbol="XYZUSDT")
    estimator.reset(instr)  # should not raise


@pytest.mark.asyncio
async def test_state_vector_structure() -> None:
    estimator = KCAStateEstimator()
    ctx = _make_ctx()
    state = await estimator.update(_market_event(price=1_234.0, qty=5.0), ctx)
    # x = [price, return, volume]
    assert len(state.x) == 3
    assert state.x[0] == 1_234.0
    assert state.x[2] == 5.0
