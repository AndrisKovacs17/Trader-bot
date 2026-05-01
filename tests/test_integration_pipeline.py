#!/usr/bin/env python3
"""Integration tests: full signal → risk → execution → fill → state pipeline.

Verifies cross-layer interactions without mocking internal seams.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.broker import MockBrokerGateway
from adapters.testing.time_source import SimulatedTimeSource
from core.application.execution import ExecutionUseCase
from core.application.ports import IEventBusPort
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.events import Event, ExecutionCompletedEvent, OrderFilledEvent
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
from core.ml.services import Prediction


# ── in-memory event bus ───────────────────────────────────────────────────────

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


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def instrument() -> Instrument:
    return Instrument(symbol="BTCUSDT")


@pytest.fixture
def bus() -> InMemoryBus:
    return InMemoryBus()


@pytest.fixture
def broker(bus: InMemoryBus) -> MockBrokerGateway:
    return MockBrokerGateway(initial_cash=100_000.0, bus=bus)


@pytest.fixture
def execution(broker: MockBrokerGateway, bus: InMemoryBus) -> ExecutionUseCase:
    return ExecutionUseCase(broker, bus)


@pytest.fixture
def ctx(instrument: Instrument) -> RunContext:
    wallet = SimulationWallet(initial_cash=100_000.0)
    wallet.mark_price(instrument, 50_000.0)
    return RunContext(
        correlation_id="int-test",
        instrument=instrument,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(
            strategy={
                "threshold": 0.55,
                "min_edge": 0.01,
                "min_confidence": 0.51,
                "min_expected_value": 0.0,
                "min_signal_score": 0.0,
                "max_sigma": 5.0,
                "min_sigma": 1e-8,
                "max_strength": 0.0,
                "use_edge_score": False,
                "min_cooldown_seconds": 0.0,
                "min_bars_between_signals": 0,
                "allowed_horizons": [],
                "horizon_scale_mode": "inverse",
                "flip_extra_entry": 0.0,
            },
            risk_limits={
                "max_qty": 1.0,
                "min_qty": 0.0001,
                "min_confidence": 0.5,
                "max_notional": 100_000.0,
                "max_abs_position_qty": 5.0,
                "max_slippage_bps": 100.0,
            },
        ),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=10,
    )


def _policy() -> RiskPolicy:
    return RiskPolicy(rules=[
        ConfidenceRule(min_confidence=0.5),
        SlippageGuardRule(max_slippage_bps=100.0),
        MaxQtyRule(max_qty=1.0, min_qty=0.0001),
        MaxNotionalRule(max_notional=100_000.0),
        MaxAbsPositionRule(max_abs_position_qty=5.0),
    ])


# ── integration: strategy → signal ───────────────────────────────────────────

def test_strategy_generates_buy_signal(ctx: RunContext) -> None:
    strategy = ThresholdStrategy(threshold=0.55, min_edge=0.01, min_confidence=0.51)
    pred = Prediction(mu=0.03, sigma=0.01, prob_up=0.75, regime="trend", horizon=1)
    sig = strategy.on_prediction(pred, ctx)
    assert sig is not None
    assert sig.side == "BUY"
    assert sig.instrument.symbol == "BTCUSDT"


def test_strategy_generates_sell_signal(ctx: RunContext) -> None:
    strategy = ThresholdStrategy(threshold=0.55, min_edge=0.01, min_confidence=0.51)
    pred = Prediction(mu=-0.03, sigma=0.01, prob_up=0.25, regime="trend", horizon=1)
    sig = strategy.on_prediction(pred, ctx)
    assert sig is not None
    assert sig.side == "SELL"


# ── integration: signal → risk policy → order ─────────────────────────────────

def test_risk_policy_approves_valid_signal(ctx: RunContext, instrument: Instrument) -> None:
    policy = _policy()
    sig = Signal(
        instrument=instrument,
        side="BUY",
        strength=0.1,
        confidence=0.8,
        horizon=1,
        reason="test",
    )
    ctx.wallet.mark_price(instrument, 50_000.0)
    decision = policy.evaluate(sig, ctx)
    assert decision.status.value == "APPROVED"
    assert decision.order is not None
    assert decision.order.side == "BUY"


def test_risk_policy_blocks_low_confidence(ctx: RunContext, instrument: Instrument) -> None:
    policy = _policy()
    sig = Signal(
        instrument=instrument,
        side="BUY",
        strength=0.1,
        confidence=0.3,
        horizon=1,
        reason="test",
    )
    decision = policy.evaluate(sig, ctx)
    assert decision.status.value == "BLOCKED"


# ── integration: execution → fill → wallet/position update ────────────────────

@pytest.mark.asyncio
async def test_buy_order_updates_position(
    broker: MockBrokerGateway,
    execution: ExecutionUseCase,
    bus: InMemoryBus,
    ctx: RunContext,
    instrument: Instrument,
) -> None:
    await broker.connect()

    async def _on_fill(evt):
        if evt.event_type == "OrderFilled":
            d = evt.payload
            fill = Fill(order_id=d["order_id"], qty=d["qty"], price=d["price"],
                        fee=d.get("fee", 0.0), side=d["side"])
            ctx.position_store.apply_fill(fill, instrument)
            ctx.wallet.apply_fill(fill, instrument)
        await execution.handle(evt, ctx)

    bus.subscribe("OrderFilled", _on_fill)

    order = Order(instrument=instrument, side="BUY", qty=0.1, limit_price=50_000.0)
    await execution.submit(order, ctx)

    pos = ctx.position_store.get(instrument)
    assert pos.qty == pytest.approx(0.1, abs=1e-9)


@pytest.mark.asyncio
async def test_sell_after_buy_closes_position(
    broker: MockBrokerGateway,
    execution: ExecutionUseCase,
    bus: InMemoryBus,
    ctx: RunContext,
    instrument: Instrument,
) -> None:
    await broker.connect()

    async def _on_fill(evt):
        if evt.event_type == "OrderFilled":
            d = evt.payload
            fill = Fill(order_id=d["order_id"], qty=d["qty"], price=d["price"],
                        fee=d.get("fee", 0.0), side=d["side"])
            ctx.position_store.apply_fill(fill, instrument)
            ctx.wallet.apply_fill(fill, instrument)
        await execution.handle(evt, ctx)

    bus.subscribe("OrderFilled", _on_fill)

    await execution.submit(Order(instrument=instrument, side="BUY", qty=0.2, limit_price=50_000.0), ctx)
    await execution.submit(Order(instrument=instrument, side="SELL", qty=0.2, limit_price=51_000.0), ctx)

    pos = ctx.position_store.get(instrument)
    assert abs(pos.qty) < 1e-9
    assert pos.realized_pnl > 0  # sold higher


@pytest.mark.asyncio
async def test_wallet_cash_decreases_on_buy(
    broker: MockBrokerGateway,
    execution: ExecutionUseCase,
    bus: InMemoryBus,
    ctx: RunContext,
    instrument: Instrument,
) -> None:
    await broker.connect()
    initial_cash = ctx.wallet.cash

    async def _on_fill(evt):
        if evt.event_type == "OrderFilled":
            fill_data = evt.payload
            fill = Fill(
                order_id=fill_data["order_id"],
                qty=fill_data["qty"],
                price=fill_data["price"],
                fee=fill_data.get("fee", 0.0),
                side=fill_data["side"],
            )
            ctx.wallet.apply_fill(fill, instrument)
        await execution.handle(evt, ctx)

    bus.subscribe("OrderFilled", _on_fill)
    await execution.submit(Order(instrument=instrument, side="BUY", qty=0.1, limit_price=50_000.0), ctx)

    assert ctx.wallet.cash < initial_cash


# ── integration: multi-instrument isolation ───────────────────────────────────

@pytest.mark.asyncio
async def test_two_instruments_independent_positions(
    broker: MockBrokerGateway,
    execution: ExecutionUseCase,
    bus: InMemoryBus,
) -> None:
    await broker.connect()
    btc = Instrument(symbol="BTCUSDT")
    eth = Instrument(symbol="ETHUSDT")

    wallet = SimulationWallet(initial_cash=100_000.0)
    wallet.mark_price(btc, 50_000.0)
    wallet.mark_price(eth, 3_000.0)
    pos_store = PositionStore()

    ctx_btc = RunContext(
        correlation_id="btc",
        instrument=btc,
        state_store=StateStore(),
        position_store=pos_store,
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=1,
    )
    ctx_eth = RunContext(
        correlation_id="eth",
        instrument=eth,
        state_store=StateStore(),
        position_store=pos_store,
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=1,
    )

    fills: list = []

    async def _on_fill(evt):
        fills.append(evt)

    bus.subscribe("OrderFilled", _on_fill)

    await execution.submit(Order(instrument=btc, side="BUY", qty=0.1, limit_price=50_000.0), ctx_btc)
    await execution.submit(Order(instrument=eth, side="BUY", qty=1.0, limit_price=3_000.0), ctx_eth)

    for evt in fills:
        d = evt.payload
        fill = Fill(order_id=d["order_id"], qty=d["qty"], price=d["price"],
                    fee=d.get("fee", 0.0), side=d["side"])
        instr = btc if d.get("symbol", "") == "BTCUSDT" else eth
        pos_store.apply_fill(fill, instr)

    assert pos_store.get(btc).qty == pytest.approx(0.1, abs=1e-9)
    assert pos_store.get(eth).qty == pytest.approx(1.0, abs=1e-9)


# ── integration: full pipeline strategy → risk → execution ───────────────────

@pytest.mark.asyncio
async def test_full_pipeline_strategy_risk_execution(
    broker: MockBrokerGateway,
    execution: ExecutionUseCase,
    bus: InMemoryBus,
    ctx: RunContext,
    instrument: Instrument,
) -> None:
    await broker.connect()

    async def _on_fill(evt):
        await execution.handle(evt, ctx)

    bus.subscribe("OrderFilled", _on_fill)

    strategy = ThresholdStrategy(threshold=0.55, min_edge=0.01, min_confidence=0.51)
    policy = _policy()
    pred = Prediction(mu=0.03, sigma=0.01, prob_up=0.80, regime="trend", horizon=1)

    sig = strategy.on_prediction(pred, ctx)
    assert sig is not None

    decision = policy.evaluate(sig, ctx)
    assert decision.status.value == "APPROVED"

    await execution.submit(decision.order, ctx)

    completed = [e for e in bus.events if isinstance(e, ExecutionCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].payload["state"] == "FILLED"
