"""Execution use case: order submission and fill handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.application.ports import IBrokerGatewayPort, IEventBusPort, IExecutionUseCase
from core.domain.events import ExecutionCompletedEvent, OrderEvent, OrderFilledEvent, OrderRejectedEvent, OrderSubmittedEvent
from core.domain.models import Fill, Order

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

    def _track_order(self, order: Order, broker_order_id: str) -> None:
        self._pending_by_broker[broker_order_id] = order
        self._pending_by_order[order.order_id] = broker_order_id

    def _untrack_order(self, order_id: str = "", broker_id: str = "") -> Order | None:
        if broker_id and broker_id in self._pending_by_broker:
            order = self._pending_by_broker.pop(broker_id)
            self._pending_by_order.pop(order.order_id, None)
            return order
        if order_id and order_id in self._pending_by_order:
            resolved_broker_id = self._pending_by_order.pop(order_id)
            return self._pending_by_broker.pop(resolved_broker_id, None)
        return None

    def _resolve_order(self, order_id: str = "", broker_id: str = "") -> Order | None:
        if broker_id:
            order = self._pending_by_broker.get(broker_id)
            if order is not None:
                return order
        if order_id:
            broker_ref = self._pending_by_order.get(order_id)
            if broker_ref is not None:
                return self._pending_by_broker.get(broker_ref)
        return None

    async def submit(self, order: Order, ctx: RunContext) -> None:
        """
        Submit order to broker.

        Flow:
        1. Validation (qty > 0, price > 0, side in BUY/SELL)
        2. Send to broker gateway
        3. Track in pending_orders
        4. Publish OrderSubmitted event
        """
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

        try:
            broker_order_id = await self.broker.send_order(order)
            self._track_order(order, broker_order_id)

            await self.bus.publish(
                OrderSubmittedEvent(
                    payload={
                        "order_id": order.order_id,
                        "broker_id": broker_order_id,
                        "symbol": order.instrument.symbol,
                        "side": order.side,
                        "qty": order.qty,
                        "limit_price": order.limit_price,
                        "time_in_force": order.time_in_force,
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )

            emit_post_submit_events = getattr(self.broker, "emit_post_submit_events", None)
            if callable(emit_post_submit_events):
                await emit_post_submit_events(
                    order=order,
                    broker_order_id=broker_order_id,
                    correlation_id=ctx.correlation_id,
                )

        except Exception as e:
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

        For fills: position & wallet already updated by execution handler,
        but we can perform additional audit/logging here.
        """
        if isinstance(order_event, OrderFilledEvent):
            broker_id = str(order_event.payload.get("broker_id", ""))
            order_id = str(order_event.payload.get("order_id", ""))
            tracked_order = self._resolve_order(order_id=order_id, broker_id=broker_id)

            if tracked_order is not None:
                fill = Fill(
                    order_id=tracked_order.order_id,
                    qty=abs(float(order_event.payload.get("qty", 0.0))),
                    price=float(order_event.payload.get("price", 0.0)),
                    fee=float(order_event.payload.get("fee", 0.0)),
                    side=str(order_event.payload.get("side", tracked_order.side)).upper(),
                )
                tracked_order.apply_fill(fill)
                if tracked_order.is_active():
                    return
                self._untrack_order(order_id=tracked_order.order_id, broker_id=broker_id)
            else:
                self._untrack_order(order_id=order_id, broker_id=broker_id)

            qty = abs(float(order_event.payload.get("qty", 0.0)))
            price = float(order_event.payload.get("price", 0.0))
            fee = float(order_event.payload.get("fee", 0.0))
            state = tracked_order.state.value if tracked_order is not None else "FILLED"
            remaining_qty = tracked_order.remaining_qty() if tracked_order is not None else 0.0

            await self.bus.publish(
                ExecutionCompletedEvent(
                    payload={
                        "order_id": order_id,
                        "qty": qty,
                        "price": price,
                        "fee": fee,
                        "notional": qty * price,
                        "state": state,
                        "remaining_qty": remaining_qty,
                    },
                    source="execution",
                    correlation_id=ctx.correlation_id,
                )
            )
            return

        if isinstance(order_event, OrderRejectedEvent) or order_event.event_type == "OrderCancelled":
            broker_id = str(order_event.payload.get("broker_id", ""))
            order_id = str(order_event.payload.get("order_id", ""))
            self._untrack_order(order_id=order_id, broker_id=broker_id)

    def pending_order_count(self) -> int:
        """Return count of pending orders (for monitoring)."""
        return len(self._pending_by_broker)
