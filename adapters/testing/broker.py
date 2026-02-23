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
    - Instantly "fills" all orders (IOC execution)
    - Always succeeds unless configured to reject
    - No slippage by default (configurable)
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        reject_probability: float = 0.0,
        bus: IEventBusPort | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.reject_probability = reject_probability
        self._order_counter = 0
        self.is_connected = False
        self.bus = bus

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

        # Random rejection based on configuration
        if random.random() < self.reject_probability:
            raise ValueError(f"Mock broker rejected order: {order.order_id}")

        # In backtest/mock: instant fill
        fill = {
            "broker_order_id": broker_order_id,
            "original_order_id": order.order_id,
            "order_id": order.order_id,
            "symbol": order.instrument.symbol,
            "side": order.side,
            "qty": order.qty,
            "price": order.limit_price if order.limit_price else 0.0,
            "fee": (order.qty * (order.limit_price if order.limit_price else 0.0) * 0.001),  # 0.1% fee
        }
        if self.bus is not None:
            await self.bus.publish(
                OrderFilledEvent(
                    payload={
                        "order_id": order.order_id,
                        "broker_id": broker_order_id,
                        "symbol": order.instrument.symbol,
                        "side": order.side,
                        "qty": order.qty,
                        "price": order.limit_price if order.limit_price else 0.0,
                        "fee": (order.qty * (order.limit_price if order.limit_price else 0.0) * 0.001),
                    },
                    source="broker",
                )
            )

        return broker_order_id

    async def cancel(self, order_id: str) -> None:
        """Cancel order (mock: no-op, fills already happened)."""
        pass

    async def health(self) -> HealthStatus:
        """Health check."""
        return HealthStatus(
            status="healthy" if self.is_connected else "disconnected",
            message="Mock broker" if self.is_connected else "Not connected",
        )
