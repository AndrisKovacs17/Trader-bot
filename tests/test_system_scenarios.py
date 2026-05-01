#!/usr/bin/env python3
"""System/E2E tests: realistic multi-step scenarios with full wiring.

These tests exercise the system as a whole unit with simulated feeds,
covering realistic user-facing scenarios (backtest-style replay, multi-bar
sequences, position lifecycle, wallet PnL).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.broker import MockBrokerGateway
from adapters.testing.time_source import SimulatedTimeSource
from core.application.execution import ExecutionUseCase
from core.application.ports import IEventBusPort
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.events import Event, ExecutionCompletedEvent
from core.domain.models import Fill, Instrument, Order, Signal
from core.domain.risk import (
    ConfidenceRule,
    MaxAbsPositionRule,
    MaxNotionalRule,
    MaxQtyRule,
    RiskPolicy,
    SlippageGuardRule,
)
from core.domain.strategy import ThresholdStrategy
from core.ml.services import KCAStateEstimator, Prediction


class InMemoryBus(IEventBusPort):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._subs: dict[str, list] = {}

    async def publish(self, event: Event) -> None:
        self.events.append(event)
        for h in self._subs.get(event.event_type, []):
            await h(event)

    def subscribe(self, event_type: str, handler) -> None:
        self._subs.setdefault(event_type, []).append(handler)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


# ── scenario: profitable long trade ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_profitable_long_trade() -> None:
    """Full system: buy at 50k, price rises to 51k, sell → positive PnL."""
    bus = InMemoryBus()
    broker = MockBrokerGateway(initial_cash=100_000.0, bus=bus)
    execution = ExecutionUseCase(broker, bus)
    await broker.connect()

    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=100_000.0)
    wallet.mark_price(instrument, 50_000.0)
    pos_store = PositionStore()
    ctx = RunContext(
        correlation_id="sys-1",
        instrument=instrument,
        state_store=StateStore(),
        position_store=pos_store,
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=10,
    )

    async def _on_fill(evt):
        if evt.event_type == "OrderFilled":
            d = evt.payload
            fill = Fill(order_id=d["order_id"], qty=d["qty"], price=d["price"],
                        fee=d.get("fee", 0.0), side=d["side"])
            pos_store.apply_fill(fill, instrument)
            wallet.apply_fill(fill, instrument)
        await execution.handle(evt, ctx)

    bus.subscribe("OrderFilled", _on_fill)

    # BUY at 50k
    await execution.submit(Order(instrument=instrument, side="BUY", qty=0.1, limit_price=50_000.0), ctx)

    # Price rises to 51k
    wallet.mark_price(instrument, 51_000.0)

    # SELL at 51k
    await execution.submit(Order(instrument=instrument, side="SELL", qty=0.1, limit_price=51_000.0), ctx)

    pos = pos_store.get(instrument)
    assert abs(pos.qty) < 1e-9, "Position should be flat after round trip"
    assert pos.realized_pnl > 0, "Should have positive PnL on profitable trade"
    completed = [e for e in bus.events if isinstance(e, ExecutionCompletedEvent)]
    assert len(completed) == 2


# ── scenario: losing short trade ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_short_position_lifecycle() -> None:
    """Open short at 50k, price falls to 49k, cover → positive PnL for short."""
    bus = InMemoryBus()
    broker = MockBrokerGateway(initial_cash=100_000.0, bus=bus)
    execution = ExecutionUseCase(broker, bus)
    await broker.connect()

    instrument = Instrument(symbol="ETHUSDT")
    wallet = SimulationWallet(initial_cash=100_000.0)
    wallet.mark_price(instrument, 50_000.0)
    pos_store = PositionStore()
    ctx = RunContext(
        correlation_id="sys-2",
        instrument=instrument,
        state_store=StateStore(),
        position_store=pos_store,
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=10,
    )

    async def _on_fill(evt):
        if evt.event_type == "OrderFilled":
            d = evt.payload
            fill = Fill(order_id=d["order_id"], qty=d["qty"], price=d["price"],
                        fee=d.get("fee", 0.0), side=d["side"])
            pos_store.apply_fill(fill, instrument)
            wallet.apply_fill(fill, instrument)
        await execution.handle(evt, ctx)

    bus.subscribe("OrderFilled", _on_fill)

    await execution.submit(Order(instrument=instrument, side="SELL", qty=0.1, limit_price=50_000.0), ctx)
    wallet.mark_price(instrument, 49_000.0)
    await execution.submit(Order(instrument=instrument, side="BUY", qty=0.1, limit_price=49_000.0), ctx)

    pos = pos_store.get(instrument)
    assert abs(pos.qty) < 1e-9
    assert pos.realized_pnl > 0


# ── scenario: multi-bar replay with strategy ─────────────────────────────────

@pytest.mark.asyncio
async def test_system_multi_bar_strategy_replay() -> None:
    """Simulate 10 bars with rising probability → strategy fires once."""
    bus = InMemoryBus()
    broker = MockBrokerGateway(initial_cash=100_000.0, bus=bus)
    execution = ExecutionUseCase(broker, bus)
    await broker.connect()

    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=100_000.0)
    wallet.mark_price(instrument, 50_000.0)
    pos_store = PositionStore()
    time_src = SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc))
    policy = RiskPolicy(rules=[
        ConfidenceRule(min_confidence=0.5),
        MaxQtyRule(max_qty=0.5, min_qty=0.0001),
        MaxNotionalRule(max_notional=100_000.0),
        MaxAbsPositionRule(max_abs_position_qty=2.0),
    ])
    strategy = ThresholdStrategy(
        threshold=0.60,
        min_edge=0.05,
        min_confidence=0.55,
        min_cooldown_seconds=0.0,
        min_bars_between_signals=0,
    )
    fills: list[Fill] = []

    async def _on_fill(evt):
        if evt.event_type == "OrderFilled":
            d = evt.payload
            f = Fill(order_id=d["order_id"], qty=d["qty"], price=d["price"],
                     fee=d.get("fee", 0.0), side=d["side"])
            fills.append(f)
            pos_store.apply_fill(f, instrument)
        await execution.handle(evt, RunContext(
            correlation_id="sys-3",
            instrument=instrument,
            state_store=StateStore(),
            position_store=pos_store,
            wallet=wallet,
            config=Config(),
            time_source=time_src,
            bar_index=i,
        ))

    bus.subscribe("OrderFilled", _on_fill)

    # 10 bars with gradually increasing prob_up
    signals_fired = 0
    for i in range(1, 11):
        prob_up = 0.50 + i * 0.04  # 0.54 .. 0.90
        pred = Prediction(mu=0.01 * i, sigma=0.005, prob_up=min(prob_up, 1.0),
                          regime="trend", horizon=1)
        ctx = RunContext(
            correlation_id=f"sys-3-bar{i}",
            instrument=instrument,
            state_store=StateStore(),
            position_store=pos_store,
            wallet=wallet,
            config=Config(),
            time_source=time_src,
            bar_index=i,
        )
        sig = strategy.on_prediction(pred, ctx)
        if sig is not None:
            decision = policy.evaluate(sig, ctx)
            if decision.status.value == "APPROVED":
                signals_fired += 1
                await execution.submit(decision.order, ctx)
        time_src.advance(timedelta(minutes=5))

    # At least some signals were generated and executed
    assert signals_fired >= 1
    assert len(fills) >= 1


# ── scenario: risk blocks all signals below confidence ────────────────────────

def test_system_risk_blocks_all_low_confidence_signals() -> None:
    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=10_000.0)
    wallet.mark_price(instrument, 50_000.0)
    pos_store = PositionStore()
    ctx = RunContext(
        correlation_id="sys-4",
        instrument=instrument,
        state_store=StateStore(),
        position_store=pos_store,
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=10,
    )
    policy = RiskPolicy(rules=[ConfidenceRule(min_confidence=0.9)])

    for conf in [0.5, 0.6, 0.7, 0.8]:
        sig = Signal(instrument=instrument, side="BUY", strength=0.1,
                     confidence=conf, horizon=1, reason="test")
        decision = policy.evaluate(sig, ctx)
        assert decision.status.value == "BLOCKED", f"confidence={conf} should be blocked"


# ── scenario: state estimator feeds predictor mock ───────────────────────────

@pytest.mark.asyncio
async def test_system_state_estimator_update_sequence() -> None:
    """State estimator correctly tracks price series through sequential updates."""
    from core.domain.events import MarketDataEvent

    estimator = KCAStateEstimator()
    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=10_000.0)
    wallet.mark_price(instrument, 50_000.0)
    ctx = RunContext(
        correlation_id="sys-5",
        instrument=instrument,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=10,
    )

    prices = [50_000.0, 50_100.0, 49_900.0, 50_200.0, 50_050.0]
    states = []
    for p in prices:
        evt = MarketDataEvent(event_type="MarketData",
                              payload={"price": p, "qty": 1.0})
        state = await estimator.update(evt, ctx)
        states.append(state)

    # Returns are computed relative to previous price
    assert states[0].features["return"] == 0.0
    assert abs(states[1].features["return"] - 0.002) < 1e-6
    assert states[1].features["price"] == 50_100.0


# ── scenario: wallet equity tracks position value ─────────────────────────────

def test_system_wallet_equity_reflects_position() -> None:
    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=100_000.0)
    wallet.mark_price(instrument, 50_000.0)
    pos_store = PositionStore()

    # Manually apply a fill
    fill = Fill(order_id="x", qty=0.5, price=50_000.0, fee=0.0, side="BUY")
    wallet.apply_fill(fill, instrument)
    pos_store.apply_fill(fill, instrument)

    wallet.mark_price(instrument, 52_000.0)
    equity = wallet.equity(pos_store)
    # cash reduced by 25000 (0.5*50000), position worth 0.5*52000=26000
    # equity = (100000-25000) + 26000 = 101000
    assert equity == pytest.approx(101_000.0, abs=1e-2)


# ── scenario: config validation ───────────────────────────────────────────────

def test_system_config_validation_raises_on_empty_symbols() -> None:
    from core.application.stores import Config
    cfg = Config(symbols=[])
    with pytest.raises(ValueError, match="symbol"):
        cfg.validate()


def test_system_config_get_nested_key() -> None:
    from core.application.stores import Config
    cfg = Config(strategy={"threshold": 0.65})
    assert cfg.get("strategy.threshold") == 0.65
    assert cfg.get("strategy.nonexistent", "default") == "default"
    assert cfg.get("no.such.path", 42) == 42
