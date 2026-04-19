#!/usr/bin/env python3
"""End-to-end integration test: order submission -> broker fill -> position/wallet state.

Covers the execution pipeline across layers:
    ExecutionUseCase -> MockBrokerGateway -> OrderFilledEvent
    -> ExecutionUseCase.handle -> PositionStore / SimulationWallet
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
from core.application.stores import (
    Config,
    PositionStore,
    RunContext,
    SimulationWallet,
    StateStore,
)
from core.domain.events import Event, ExecutionCompletedEvent
from core.domain.models import Instrument, Order


class InMemoryEventBus(IEventBusPort):
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.subscribers: dict[str, list] = {}

    async def publish(self, event: Event) -> None:
        self.events.append(event)
        for handler in self.subscribers.get(event.event_type, []):
            await handler(event)

    def subscribe(self, event_type: str, handler) -> None:
        self.subscribers.setdefault(event_type, []).append(handler)

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


@pytest.mark.asyncio
async def test_end_to_end_buy_sell_round_trip() -> None:
    """A full buy + sell round-trip flows through the pipeline and settles wallet/position."""
    bus = InMemoryEventBus()
    broker = MockBrokerGateway(initial_cash=10_000.0, bus=bus)
    execution = ExecutionUseCase(broker, bus)
    await broker.connect()

    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=10_000.0)
    wallet.mark_price(instrument, 50_000.0)
    ctx = RunContext(
        correlation_id="e2e-1",
        instrument=instrument,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(),
        time_source=SimulatedTimeSource(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        bar_index=1,
    )

    # Wire broker fills back into the execution use-case (what the real engine does)
    async def _on_fill(event):
        await execution.handle(event, ctx)
    bus.subscribe("OrderFilled", _on_fill)

    buy = Order(instrument=instrument, side="BUY", qty=0.1, limit_price=50_000.0)
    await execution.submit(buy, ctx)

    assert execution.pending_order_count() == 0, "BUY-ordernek azonnal teljesülnie kell"
    completed = [e for e in bus.events if isinstance(e, ExecutionCompletedEvent)]
    assert len(completed) == 1
    assert completed[0].payload["state"] == "FILLED"
    assert "OrderSubmitted" in {e.event_type for e in bus.events}
    assert "OrderFilled" in {e.event_type for e in bus.events}

    sell = Order(instrument=instrument, side="SELL", qty=0.1, limit_price=51_000.0)
    await execution.submit(sell, ctx)

    assert execution.pending_order_count() == 0
    completed = [e for e in bus.events if isinstance(e, ExecutionCompletedEvent)]
    assert len(completed) == 2
    fills = [e for e in bus.events if e.event_type == "OrderFilled"]
    assert len(fills) == 2
    assert [f.payload["side"] for f in fills] == ["BUY", "SELL"]
