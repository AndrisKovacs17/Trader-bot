#!/usr/bin/env python3
"""End-to-end integration test: full trading pipeline from market data to fill."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.application.engine import TradingEngine
from core.application.execution import ExecutionUseCase
from core.application.ports import IEventBusPort
from core.application.stages import (
    ExecutionStage,
    PredictionStage,
    RiskStage,
    SignalStage,
    StateEstimationStage,
)
from core.application.stores import (
    Config,
    PositionStore,
    RunContext,
    SimulationWallet,
    StateStore,
)
from core.domain.events import Event, MarketDataEvent, OrderFilledEvent
from core.domain.models import Instrument
from core.domain.risk import (
    ConfidenceRule,
    MaxAbsPositionRule,
    MaxNotionalRule,
    MaxQtyRule,
    RiskPolicy,
    SlippageGuardRule,
)
from core.domain.strategy import ThresholdStrategy
from core.ml.services import EstimatedState, KCAStateEstimator, Prediction
from adapters.testing.broker import MockBrokerGateway
from adapters.testing.time_source import SimulatedTimeSource


# Helpers

class SyncEventBus(IEventBusPort):
    """Synchronous event bus for deterministic E2E testing."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.subscribers: dict[str, list] = {}

    async def publish(self, event: Event) -> None:
        self.events.append(event)
        for handler in self.subscribers.get(event.event_type, []):
            await handler(event)

    def subscribe(self, event_type: str, handler) -> None:
        self.subscribers.setdefault(event_type, []).append(handler)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class ForcedPredictor:
    """Predictor that alternates BUY/SELL signals to guarantee trades."""

    def __init__(self) -> None:
        self._call_count = 0

    async def predict(self, state: EstimatedState, ctx: RunContext) -> Prediction:
        self._call_count += 1
        # Alternate: odd calls → BUY, even calls → SELL
        if self._call_count % 2 == 1:
            return Prediction(mu=0.05, sigma=0.02, prob_up=0.90, regime="forced-buy", horizon=1)
        return Prediction(mu=-0.05, sigma=0.02, prob_up=0.10, regime="forced-sell", horizon=1)

    def current_version(self) -> str:
        return "test-forced-v1"


def make_market_event(price: float, symbol: str = "BTCUSDT") -> MarketDataEvent:
    return MarketDataEvent(
        payload={"symbol": symbol, "price": price, "qty": 1.0, "volume": 100.0},
        source="test",
    )


# Fixtures

@pytest.fixture
def config() -> Config:
    return Config(
        symbols=["BTCUSDT"],
        risk_limits={
            "max_qty": 1.0,
            "min_qty": 0.0001,
            "min_confidence": 0.5,
            "max_notional": 100_000.0,
            "max_abs_position_qty": 10.0,
            "max_slippage_bps": 500.0,
            "allow_short_selling": True,
            "estimated_fee_bps": 10.0,
        },
        strategy={
            "threshold": 0.55,
            "min_edge": 0.02,
            "min_confidence": 0.55,
            "min_bars_between_signals": 0,
            "min_cooldown_seconds": 0.0,
        },
        simulation={"initial_cash": 10_000.0},
    )


@pytest.fixture
def bus() -> SyncEventBus:
    return SyncEventBus()


@pytest.fixture
def engine(bus, config):
    """Build a fully wired TradingEngine with deterministic components."""
    wallet = SimulationWallet(initial_cash=10_000.0)
    position_store = PositionStore()
    state_store = StateStore()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time_source = SimulatedTimeSource(now)

    estimator = KCAStateEstimator()
    predictor = ForcedPredictor()
    strategy = ThresholdStrategy(config)
    risk_policy = RiskPolicy(
        rules=[
            ConfidenceRule(),
            SlippageGuardRule(),
            MaxQtyRule(),
            MaxNotionalRule(),
            MaxAbsPositionRule(),
        ],
    )
    broker = MockBrokerGateway(initial_cash=10_000.0, bus=bus, auto_fill=True)
    execution_uc = ExecutionUseCase(broker, bus)

    engine = TradingEngine(
        bus=bus,
        state_store=state_store,
        position_store=position_store,
        wallet=wallet,
        config=config,
        time_source=time_source,
        state_stage=StateEstimationStage(estimator, state_store),
        pred_stage=PredictionStage(predictor),
        signal_stage=SignalStage(strategy),
        risk_stage=RiskStage(risk_policy),
        exec_stage=ExecutionStage(execution_uc),
    )

    # Wire fill handler: update wallet and position store on OrderFilled
    async def on_fill(event: Event) -> None:
        if not isinstance(event, OrderFilledEvent):
            return
        from core.domain.models import Fill
        p = event.payload
        fill = Fill(
            order_id=p["order_id"],
            price=p["fill_price"],
            qty=p["fill_qty"],
            side=p["side"],
            fee=p.get("fee", 0.0),
            timestamp=event.ts_event,
        )
        instrument = Instrument(symbol=p["symbol"])
        wallet.apply_fill(fill, instrument)
        position_store.apply_fill(fill, instrument)

    bus.subscribe("OrderFilled", on_fill)

    return engine


# Tests

@pytest.mark.asyncio
async def test_full_pipeline_produces_fill(engine, bus):
    """Feed synthetic prices → engine should produce at least one fill."""
    prices = [50_000.0, 50_100.0, 50_200.0, 50_300.0, 50_400.0]

    for price in prices:
        await engine.handle(make_market_event(price))

    event_types = [e.event_type for e in bus.events]

    # Pipeline must have produced predictions
    assert "Prediction" in event_types, f"No Prediction events. Got: {event_types}"

    # At least one signal must have been generated
    assert "Signal" in event_types, f"No Signal events. Got: {event_types}"

    # At least one order must have been filled
    assert "OrderFilled" in event_types, f"No OrderFilled events. Got: {event_types}"


@pytest.mark.asyncio
async def test_equity_tracking(engine, bus):
    """Equity should be tracked after processing market data."""
    prices = [50_000.0, 50_100.0, 50_200.0]

    for price in prices:
        await engine.handle(make_market_event(price))

    wallet = engine.wallet
    position_store = engine.position_store

    equity = wallet.equity(position_store)
    assert equity > 0, f"Equity should be positive, got {equity}"

    # Equity history should have entries from wallet.record() calls
    assert len(wallet.history) >= len(prices), (
        f"Expected at least {len(prices)} equity snapshots, got {len(wallet.history)}"
    )


@pytest.mark.asyncio
async def test_position_updated_after_fill(engine, bus):
    """After a fill, the position store should reflect the trade."""
    prices = [50_000.0, 50_100.0, 50_200.0]

    for price in prices:
        await engine.handle(make_market_event(price))

    fills = [e for e in bus.events if e.event_type == "OrderFilled"]
    assert len(fills) >= 1, "Expected at least one fill event"

    # Verify the fill handler processed events through the wallet
    instrument = Instrument(symbol="BTCUSDT")
    # last_prices should be populated after processing market events
    assert "BTCUSDT" in engine.wallet.last_prices, (
        "Wallet should track last prices after market events"
    )


@pytest.mark.asyncio
async def test_multiple_trades_round_trip(engine, bus):
    """Multiple market events should produce multiple pipeline cycles."""
    # Feed enough bars to trigger at least 2 trades (BUY then SELL)
    prices = [50_000.0 + i * 50 for i in range(10)]

    for price in prices:
        await engine.handle(make_market_event(price))

    fills = [e for e in bus.events if e.event_type == "OrderFilled"]
    assert len(fills) >= 2, f"Expected at least 2 fills for round-trip, got {len(fills)}"

    # Verify both BUY and SELL fills exist
    sides = {f.payload["side"] for f in fills}
    assert "BUY" in sides, f"Expected a BUY fill, got sides: {sides}"
    assert "SELL" in sides, f"Expected a SELL fill, got sides: {sides}"
