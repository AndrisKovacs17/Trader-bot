from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.application.ports import IEventBusPort, IEventHandlerPort, IExecutionUseCase
from core.application.stores import PositionStore, SimulationWallet
from core.domain.events import Event, OrderFilledEvent, OrderRejectedEvent, PerformanceUpdateEvent
from core.domain.models import Fill, Instrument

if TYPE_CHECKING:
    from core.application.stores import Config

@dataclass(slots=True)
class _ExecutionEventContext:
    correlation_id: str


class ExecutionOrderEventHandler(IEventHandlerPort):
    """Infrastructure adapter: routes broker order events to ExecutionUseCase."""

    def __init__(
        self,
        execution: IExecutionUseCase,
        position_store: PositionStore,
        wallet: SimulationWallet,
        bus: IEventBusPort,
        config: Config,
    ) -> None:
        self.execution = execution
        self.position_store = position_store
        self.wallet = wallet
        self.bus = bus
        self.config = config

    async def handle(self, event: Event) -> None:
        if isinstance(event, OrderFilledEvent):
            await self._apply_fill_side_effects(event)
            await self.execution.handle(event, _ExecutionEventContext(correlation_id=event.correlation_id))
            return

        if not isinstance(event, OrderRejectedEvent):
            return
        await self.execution.handle(event, _ExecutionEventContext(correlation_id=event.correlation_id))

    def supported_types(self) -> set[str]:
        return {"OrderFilled", "OrderRejected"}

    async def _apply_fill_side_effects(self, event: OrderFilledEvent) -> None:
        symbol = str(event.payload.get("symbol", self.config.symbols[0]))
        instrument = Instrument(symbol=symbol)
        side = str(event.payload.get("side", "BUY")).upper()
        fill = Fill(
            order_id=str(event.payload.get("order_id", "")),
            qty=abs(float(event.payload.get("qty", 0.0))),
            price=float(event.payload.get("price", 0.0)),
            fee=float(event.payload.get("fee", 0.0)),
            side=side,
        )

        self.position_store.apply_fill(fill, instrument)
        self.wallet.apply_fill(fill, instrument)
        self.wallet.record(self.position_store)

        current_equity = self.wallet.equity(self.position_store)

        await self.bus.publish(
            PerformanceUpdateEvent(
                payload={
                    "order_id": fill.order_id,
                    "qty": fill.qty,
                    "price": fill.price,
                    "fee": fill.fee,
                    "side": side,
                    "current_cash": self.wallet.cash,
                    "current_equity": current_equity,
                },
                source="execution-handler",
                correlation_id=event.correlation_id or event.event_id,
            )
        )
