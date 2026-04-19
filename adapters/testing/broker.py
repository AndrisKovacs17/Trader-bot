"""Mock broker gateway for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.application.ports import IBrokerGatewayPort, IEventBusPort
from core.domain.events import OrderFilledEvent
from core.domain.models import Order
from core.ops.contracts import HealthStatus

if TYPE_CHECKING:
    pass


class MockBrokerGateway(IBrokerGatewayPort):
    """
    Mock broker for backtesting and unit tests.

    Behavior:
    - Instantly fills orders after OrderSubmitted was published
    - Always succeeds unless configured to reject
    - No slippage by default (configurable)
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        reject_probability: float = 0.0,
        bus: IEventBusPort | None = None,
        auto_fill: bool = True,
    ) -> None:
        self.initial_cash = initial_cash
        self.reject_probability = reject_probability
        self._order_counter = 0
        self.is_connected = False
        self.bus = bus
        self.auto_fill = auto_fill

    async def connect(self) -> None:
        """Connect to mock broker (instant)."""
        self.is_connected = True

    async def send_order(self, order: Order) -> str:
        """
        Send order to mock broker.

        Returns:
            Broker order ID (same as input order_id in mock)
        """
        import random

        self._order_counter += 1
        broker_order_id = f"mock-{self._order_counter}"

        if random.random() < self.reject_probability:
            raise ValueError(f"Mock broker rejected order: {order.order_id}")

        return broker_order_id

    async def emit_post_submit_events(self, *, order: Order, broker_order_id: str, correlation_id: str = "") -> None:
        if not self.auto_fill or self.bus is None:
            return

        price = float(order.limit_price) if order.limit_price is not None else max(float(order.instrument.tick_size), 1.0)
        fee = order.qty * price * 0.001
        await self.bus.publish(
            OrderFilledEvent(
                payload={
                    "order_id": order.order_id,
                    "broker_id": broker_order_id,
                    "symbol": order.instrument.symbol,
                    "side": order.side,
                    "qty": order.qty,
                    "price": price,
                    "fee": fee,
                },
                source="broker",
                correlation_id=correlation_id,
            )
        )

    async def cancel(self, order_id: str) -> None:
        """Cancel order (mock: no-op, fills already happened)."""
        _ = order_id

    async def health(self) -> HealthStatus:
        """Health check."""
        return HealthStatus(
            ok=self.is_connected,
            details={
                "status": "healthy" if self.is_connected else "disconnected",
                "gateway": "mock",
                "message": "Mock broker" if self.is_connected else "Not connected",
            },
        )
