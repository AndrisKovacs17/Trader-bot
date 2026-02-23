#!/usr/bin/env python3
"""ExecutionUseCase and pipeline integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.broker import MockBrokerGateway
from adapters.testing.time_source import SimulatedTimeSource
from core.application.execution import ExecutionUseCase
from core.application.ports import IEventBusPort
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.models import Instrument, Order
from core.domain.events import Event, ExecutionCompletedEvent, OrderEvent, OrderFilledEvent


class InMemoryEventBus(IEventBusPort):
    """Simple in-memory event bus for testing."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.subscribers: dict[str, list] = {}

    async def publish(self, event: Event) -> None:
        self.events.append(event)
        handlers = self.subscribers.get(event.event_type, [])
        for handler in handlers:
            await handler(event)

    def subscribe(self, event_type: str, handler) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.fixture
def broker():
    return MockBrokerGateway(initial_cash=10000.0)


@pytest.fixture
def bus():
    return InMemoryEventBus()


@pytest.fixture
def execution_uc(broker, bus):
    return ExecutionUseCase(broker, bus)


@pytest.fixture
def ctx():
    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=10000.0)
    wallet.mark_price(instrument, 50000.0)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return RunContext(
        correlation_id="corr-1",
        instrument=instrument,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(now),
        bar_index=1,
    )


class TestExecutionUseCase:
    """ExecutionUseCase behavioral tests."""

    @pytest.mark.asyncio
    async def test_submit_valid_order(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """Submit valid order should publish OrderSubmitted event."""
        await execution_uc.broker.connect()

        order = Order(
            instrument=ctx.instrument,
            side="BUY",
            qty=0.5,
            limit_price=50000.0,
        )

        await execution_uc.submit(order, ctx)

        # Should publish OrderSubmitted event
        submitted_events = [e for e in execution_uc.bus.events if e.event_type == "OrderSubmitted"]
        assert len(submitted_events) == 1
        assert submitted_events[0].payload["qty"] == 0.5

    @pytest.mark.asyncio
    async def test_submit_invalid_qty(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """Submit with qty <= 0 should be rejected at Order creation."""
        # Order constructor validates qty > 0, so we catch that
        with pytest.raises(ValueError, match="qty must be > 0"):
            Order(
                instrument=ctx.instrument,
                side="BUY",
                qty=-1.0,
                limit_price=50000.0,
            )

    @pytest.mark.asyncio
    async def test_submit_invalid_side(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """Submit with invalid side should be rejected at Order creation."""
        # Order constructor validates side, so we catch that
        with pytest.raises(ValueError, match="side must be BUY or SELL"):
            Order(
                instrument=ctx.instrument,
                side="INVALID",
                qty=0.5,
                limit_price=50000.0,
            )

    @pytest.mark.asyncio
    async def test_pending_order_count(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """pending_order_count should track submitted orders."""
        await execution_uc.broker.connect()

        order = Order(
            instrument=ctx.instrument,
            side="BUY",
            qty=0.5,
            limit_price=50000.0,
        )

        assert execution_uc.pending_order_count() == 0
        await execution_uc.submit(order, ctx)
        assert execution_uc.pending_order_count() == 1

    @pytest.mark.asyncio
    async def test_handle_fill_removes_pending(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """Handle OrderFilled should remove order from pending."""
        await execution_uc.broker.connect()

        order = Order(
            instrument=ctx.instrument,
            side="BUY",
            qty=0.5,
            limit_price=50000.0,
        )

        await execution_uc.submit(order, ctx)
        assert execution_uc.pending_order_count() == 1

        # Simulate fill
        fill_event = OrderFilledEvent(
            payload={
                "order_id": order.order_id,
                "broker_id": "mock-1",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "qty": 0.5,
                "price": 50000.0,
                "fee": 25.0,
            },
            source="broker",
            correlation_id="corr-1",
        )

        await execution_uc.handle(fill_event, ctx)
        assert execution_uc.pending_order_count() == 0

        # Should publish ExecutionComplete
        complete_events = [e for e in execution_uc.bus.events if isinstance(e, ExecutionCompletedEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].payload["qty"] == 0.5

    @pytest.mark.asyncio
    async def test_broker_error_handling(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """Broker error should publish OrderRejected."""
        execution_uc.broker = MockBrokerGateway(reject_probability=1.0)
        await execution_uc.broker.connect()

        order = Order(
            instrument=ctx.instrument,
            side="BUY",
            qty=0.5,
            limit_price=50000.0,
        )
        await execution_uc.submit(order, ctx)
        
        rejected_events = [e for e in execution_uc.bus.events if e.event_type == "OrderRejected"]
        assert len(rejected_events) == 1

    @pytest.mark.asyncio
    async def test_multiple_orders_tracking(self, execution_uc: ExecutionUseCase, ctx: RunContext) -> None:
        """Multiple orders should be tracked independently."""
        await execution_uc.broker.connect()

        order1 = Order(
            instrument=ctx.instrument,
            side="BUY",
            qty=0.5,
            limit_price=50000.0,
        )
        order2 = Order(
            instrument=ctx.instrument,
            side="SELL",
            qty=0.3,
            limit_price=51000.0,
        )

        await execution_uc.submit(order1, ctx)
        await execution_uc.submit(order2, ctx)
        assert execution_uc.pending_order_count() == 2

        # Fill first order
        fill_event1 = OrderFilledEvent(
            payload={
                "order_id": order1.order_id,
                "broker_id": "mock-1",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "qty": 0.5,
                "price": 50000.0,
                "fee": 25.0,
            },
            source="broker",
            correlation_id="corr-1",
        )
        await execution_uc.handle(fill_event1, ctx)
        assert execution_uc.pending_order_count() == 1

        # Fill second order
        fill_event2 = OrderFilledEvent(
            payload={
                "order_id": order2.order_id,
                "broker_id": "mock-2",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "qty": 0.3,
                "price": 51000.0,
                "fee": 15.3,
            },
            source="broker",
            correlation_id="corr-1",
        )
        await execution_uc.handle(fill_event2, ctx)
        assert execution_uc.pending_order_count() == 0
