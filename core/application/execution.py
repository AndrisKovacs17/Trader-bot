"""Execution use case: order submission and fill handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.application.ports import IBrokerGatewayPort, IEventBusPort, IExecutionUseCase
from core.domain.events import ExecutionCompletedEvent, OrderEvent, OrderFilledEvent, OrderRejectedEvent, OrderSubmittedEvent
from core.domain.models import Order

if TYPE_CHECKING:
    from core.application.stores import RunContext


class ExecutionUseCase(IExecutionUseCase):
    """
    Execute orders through broker gateway.

    Responsibilities:
    - Validate orders before submission
    - Send orders to broker (IBrokerGatewayPort)
    - Track order status
    - Publish order events (OrderSubmitted, OrderRejected)
    - Handle order fills (OrderFilled events)
    """

    def __init__(self, broker: IBrokerGatewayPort, bus: IEventBusPort) -> None:
        self.broker = broker
        self.bus = bus
        self._pending_by_broker: dict[str, Order] = {}
        self._pending_by_order: dict[str, str] = {}

    async def submit(self, order: Order, ctx: RunContext) -> None:
        """
        Submit order to broker.

        Flow:
        1. Validation (qty > 0, price > 0, side in BUY/SELL)
        2. Send to broker gateway
        3. Track in pending_orders
        4. Publish OrderSubmitted event
        """
        # Validation
        if order.qty <= 0:
            await self.bus.publish(
                OrderEvent(
                    event_type="OrderRejected",
                    payload={
                        "order_id": order.order_id,
                        "reason": "Invalid quantity",
                        "symbol": order.instrument.symbol,
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )
            return

        if order.side not in ("BUY", "SELL"):
            await self.bus.publish(
                OrderEvent(
                    event_type="OrderRejected",
                    payload={
                        "order_id": order.order_id,
                        "reason": "Invalid side",
                        "symbol": order.instrument.symbol,
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )
            return

        # Send to broker
        try:
            broker_order_id = await self.broker.send_order(order)
            self._pending_by_broker[broker_order_id] = order
            self._pending_by_order[order.order_id] = broker_order_id

            # Publish OrderSubmitted event
            await self.bus.publish(
                OrderSubmittedEvent(
                    payload={
                        "order_id": order.order_id,
                        "broker_id": broker_order_id,
                        "symbol": order.instrument.symbol,
                        "side": order.side,
                        "qty": order.qty,
                        "limit_price": order.limit_price,
                        "time_in_force": "IOC",
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )

        except Exception as e:
            # Broker error: publish rejection
            await self.bus.publish(
                OrderRejectedEvent(
                    payload={
                        "order_id": order.order_id,
                        "reason": str(e),
                        "symbol": order.instrument.symbol,
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )

    async def handle(self, order_event: OrderEvent, ctx: RunContext) -> None:
        """
        Handle order event (fill, rejection, etc).

        For fills: position & wallet already updated by engine,
        but we can perform additional audit/logging here.
        """
        if isinstance(order_event, OrderFilledEvent):
            broker_id = str(order_event.payload.get("broker_id", ""))
            order_id = str(order_event.payload.get("order_id", ""))
            if broker_id and broker_id in self._pending_by_broker:
                self._pending_by_broker.pop(broker_id, None)
                if order_id:
                    self._pending_by_order.pop(order_id, None)
            elif order_id in self._pending_by_order:
                broker_id = self._pending_by_order.pop(order_id)
                self._pending_by_broker.pop(broker_id, None)

            # Optional: additional logging, fees audit, etc.
            qty = float(order_event.payload.get("qty", 0.0))
            price = float(order_event.payload.get("price", 0.0))
            fee = float(order_event.payload.get("fee", 0.0))

            # Publish for observability
            await self.bus.publish(
                ExecutionCompletedEvent(
                    payload={
                        "order_id": order_id,
                        "qty": qty,
                        "price": price,
                        "fee": fee,
                        "notional": qty * price,
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )

        elif isinstance(order_event, OrderRejectedEvent):
            broker_id = str(order_event.payload.get("broker_id", ""))
            order_id = str(order_event.payload.get("order_id", ""))
            if broker_id and broker_id in self._pending_by_broker:
                self._pending_by_broker.pop(broker_id, None)
                if order_id:
                    self._pending_by_order.pop(order_id, None)
            elif order_id in self._pending_by_order:
                broker_id = self._pending_by_order.pop(order_id)
                self._pending_by_broker.pop(broker_id, None)

    def pending_order_count(self) -> int:
        """Return count of pending orders (for monitoring)."""
        return len(self._pending_by_broker)
