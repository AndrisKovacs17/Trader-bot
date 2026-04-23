from __future__ import annotations

from datetime import datetime
from typing import Protocol, TYPE_CHECKING

from core.domain.events import Event, OrderEvent
from core.domain.models import Instrument, Order
from core.ml.services import EstimatedState, IModelUpdatePort
from core.ops.contracts import HealthStatus

if TYPE_CHECKING:
    from core.application.stores import RunContext


# EVENT HANDLING PORTS

# Event handler port.
class IEventHandlerPort(Protocol):
    async def handle(self, event: Event) -> None:
        ...

    def supported_types(self) -> set[str]:
        ...


# Event bus port.
class IEventBusPort(Protocol):
    async def publish(self, event: Event) -> None:
        ...

    def subscribe(self, event_type: str, handler: IEventHandlerPort) -> None:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...


# STORAGE PORTS (Hexagonal boundary)

# State repository port: Persists estimated state
class IStateRepository(Protocol):
    def get(self, instrument: Instrument) -> EstimatedState:
        ...

    def save(self, instrument: Instrument, state: EstimatedState) -> None:
        ...


# Event store port.
class IEventStorePort(Protocol):
    async def append(self, event: Event) -> None:
        ...

    async def append_batch(self, events: list[Event]) -> None:
        ...

    async def read_stream(self, filters: dict) -> list[Event]:
        ...

    async def replay(self, from_ts: datetime, to_ts: datetime, instrument: Instrument | None) -> list[Event]:
        ...


# TRADING INFRASTRUCTURE PORTS

# Broker gateway port.
class IBrokerGatewayPort(Protocol):
    async def connect(self) -> None:
        ...

    async def send_order(self, order: Order) -> str:
        ...

    async def cancel(self, order_id: str) -> None:
        ...

    async def health(self) -> HealthStatus:
        ...


# Time source port.
class ITimeSource(Protocol):
    def now(self) -> datetime:
        ...


# USE CASE PORTS (Application layer)

# Execution use-case port.
class IExecutionUseCase(Protocol):
    async def submit(self, order: Order, ctx: RunContext) -> None:
        ...

    async def handle(self, order_event: OrderEvent, ctx: RunContext) -> None:
        ...


# Re-export model update port from ML layer for convenience
__all__ = [
    "IEventHandlerPort",
    "IEventBusPort",
    "IStateRepository",
    "IEventStorePort",
    "IBrokerGatewayPort",
    "ITimeSource",
    "IExecutionUseCase",
    "IModelUpdatePort",
]
