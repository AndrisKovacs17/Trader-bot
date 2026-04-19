"""Live Binance broker gateway adapter.

UML-aligned broker port implementation:
- connect
- send_order
- cancel
- health

This adapter is intentionally infrastructure-only. Real API wiring can be injected
via async callbacks (submit/cancel/ping) without changing core/application code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

from core.application.ports import IBrokerGatewayPort
from core.domain.models import Order
from core.ops.contracts import HealthStatus

if TYPE_CHECKING:
    pass


SubmitOrderFn = Callable[[dict[str, Any]], Awaitable[str]]
CancelOrderFn = Callable[[str], Awaitable[None]]
PingFn = Callable[[], Awaitable[dict[str, Any] | None]]


class BinanceBrokerGateway(IBrokerGatewayPort):
    """
    Live Binance broker integration.

    Production implementation requires:
    - CCXT or python-binance wrapper
    - WebSocket stream for fills
    - Order status polling
    - Error handling & reconnection logic
    - Rate limiting
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        submit_order_fn: SubmitOrderFn | None = None,
        cancel_order_fn: CancelOrderFn | None = None,
        ping_fn: PingFn | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._submit_order_fn = submit_order_fn
        self._cancel_order_fn = cancel_order_fn
        self._ping_fn = ping_fn
        self._is_connected = False

    async def connect(self) -> None:
        """Mark adapter connected after minimal credential validation."""
        if not self.api_key or not self.api_secret:
            raise RuntimeError("Binance credentials are missing")
        self._is_connected = True

    def _require_connected(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Not connected to Binance")

    async def send_order(self, order: Order) -> str:
        """
        Send limit order to Binance.

        Args:
            order: Order to submit

        Returns:
            Binance order ID

        Raises:
            ValueError: Order validation error
            RuntimeError: Broker error
        """
        self._require_connected()

        symbol = order.instrument.symbol
        side = order.side.upper()
        payload = {
            "symbol": symbol,
            "side": side,
            "qty": float(order.qty),
            "order_type": str(order.order_type),
            "limit_price": (float(order.limit_price) if order.limit_price is not None else None),
            "time_in_force": str(order.time_in_force),
            "client_tag": str(order.client_tag),
            "client_order_id": str(order.order_id),
        }

        if self._submit_order_fn is None:
            raise RuntimeError("Binance submit adapter not configured")

        broker_order_id = await self._submit_order_fn(payload)
        if not broker_order_id:
            raise RuntimeError("Binance returned empty order id")
        return str(broker_order_id)

    async def cancel(self, order_id: str) -> None:
        """Cancel order by ID."""
        self._require_connected()
        if self._cancel_order_fn is None:
            raise RuntimeError("Binance cancel adapter not configured")
        await self._cancel_order_fn(order_id)

    async def health(self) -> HealthStatus:
        """Check Binance API health."""
        if not self._is_connected:
            return HealthStatus(ok=False, details={"status": "disconnected", "venue": "binance", "testnet": self.testnet})

        ping_ok = True
        if self._ping_fn is not None:
            try:
                await self._ping_fn()
            except Exception:
                ping_ok = False

        return HealthStatus(
            ok=ping_ok,
            details={
                "status": "healthy" if ping_ok else "degraded",
                "venue": "binance",
                "testnet": self.testnet,
                "submit_configured": self._submit_order_fn is not None,
                "cancel_configured": self._cancel_order_fn is not None,
            },
        )

    async def __aenter__(self) -> "BinanceBrokerGateway":
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self._is_connected = False


class BrokerGatewayImpl(BinanceBrokerGateway):
    """UML-aligned alias for the concrete broker adapter."""
