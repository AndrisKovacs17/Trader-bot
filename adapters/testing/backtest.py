from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.application.ports import IBrokerGatewayPort, IEventBusPort, IEventHandlerPort, IEventStorePort
from core.domain.events import Event
from core.domain.models import Fill, Order
from core.ops.contracts import HealthStatus
from adapters.testing.time_source import SimulatedTimeSource


# Backtest futás eredménye.
@dataclass(slots=True)
class BacktestResult:
    metrics: dict
    equity_curve: list[float] = field(default_factory=list)
    trades: list[Fill] = field(default_factory=list)


# Minimal backtest engine adapter.
class BacktestEngine:
    def __init__(
        self,
        event_store: IEventStorePort,
        handler: IEventHandlerPort,
        bus: IEventBusPort | None = None,
        time_source: SimulatedTimeSource | None = None,
    ) -> None:
        self.event_store = event_store
        self.handler = handler
        self.bus = bus or MockEventBus()
        self.time_source = time_source or SimulatedTimeSource(datetime.now(timezone.utc))

    async def run(self, from_ts: datetime, to_ts: datetime) -> BacktestResult:
        events = await self.event_store.replay(from_ts, to_ts, instrument=None)
        for event_type in self.handler.supported_types():
            self.bus.subscribe(event_type, self.handler)
        await self.bus.start()
        for event in events:
            self.time_source.set(event.ts_event)
            await self.step(event)
        await self.bus.stop()
        return BacktestResult(metrics={"events": len(events)})

    async def step(self, event: Event) -> None:
        await self.bus.publish(event)


class MockBrokerGateway(IBrokerGatewayPort):
    async def connect(self) -> None:
        return None

    async def send_order(self, order: Order) -> str:
        return f"mock-{order.order_id}"

    async def cancel(self, order_id: str) -> None:
        _ = order_id

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, details={"gateway": "mock"})


class MockEventBus(IEventBusPort):
    def __init__(self) -> None:
        self.handlers: dict[str, list[IEventHandlerPort]] = {}
        self.started = False

    async def publish(self, event: Event) -> None:
        for handler in self.handlers.get(event.event_type, []):
            await handler.handle(event)

    def subscribe(self, event_type: str, handler: IEventHandlerPort) -> None:
        self.handlers.setdefault(event_type, []).append(handler)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


class FixedTimeSource:
    def __init__(self, fixed_time: datetime) -> None:
        self.fixed_time = fixed_time

    def now(self) -> datetime:
        return self.fixed_time
