from __future__ import annotations

from core.analytics.services import ModelDiagnosticsService, PerformanceTracker
from core.application.ports import IEventHandlerPort
from core.application.stores import PositionStore, SimulationWallet
from core.domain.events import Event, MarketDataEvent, PerformanceUpdateEvent, PredictionEvent
from core.domain.models import Fill
from core.ml.services import Prediction
from core.ops.contracts import IMetrics


# Nem blokkoló observability handler: event busról hallgat.
class EventTapHandler(IEventHandlerPort):
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        diagnostics: ModelDiagnosticsService,
        metrics: IMetrics,
        wallet: SimulationWallet | None = None,
        position_store: PositionStore | None = None,
    ) -> None:
        self.performance_tracker = performance_tracker
        self.diagnostics = diagnostics
        self.metrics = metrics
        self.wallet = wallet
        self.position_store = position_store

    async def handle(self, event: Event) -> None:
        # Typed event dispatch for observability updates.
        if isinstance(event, MarketDataEvent):
            symbol = str(event.payload.get("symbol", "UNKNOWN"))
            price = float(event.payload.get("price", 0.0))
            if price > 0:
                self.diagnostics.on_market_price(symbol, price)
                if self.wallet is not None and self.position_store is not None:
                    self.performance_tracker.on_equity(
                        current_equity=self.wallet.equity(self.position_store),
                        current_cash=self.wallet.cash,
                    )
            return

        if isinstance(event, PredictionEvent):
            pred = Prediction(
                mu=float(event.payload.get("mu", 0.0)),
                sigma=float(event.payload.get("sigma", 0.0)),
                prob_up=float(event.payload.get("prob_up", 0.5)),
                regime=str(event.payload.get("regime", "normal")),
                horizon=int(event.payload.get("horizon", 1)),
            )
            symbol = str(event.payload.get("symbol", "UNKNOWN"))
            self.performance_tracker.on_prediction(pred)
            self.diagnostics.track_prediction(symbol, pred)
            self.metrics.counter(
                "prediction_events",
                {
                    "regime": pred.regime,
                    "horizon": str(pred.horizon),
                },
            )

        elif isinstance(event, PerformanceUpdateEvent):
            # Authoritative fill event published by execution handler
            fill = Fill(
                order_id=str(event.payload.get("order_id", "")),
                qty=abs(float(event.payload.get("qty", 0.0))),
                price=float(event.payload.get("price", 0.0)),
                fee=float(event.payload.get("fee", 0.0)),
                side=str(event.payload.get("side", "BUY")).upper(),
            )
            current_equity = float(event.payload.get("current_equity", 0.0))
            current_cash = float(event.payload.get("current_cash", 0.0))
            self.performance_tracker.on_fill(fill, current_equity=current_equity, current_cash=current_cash)
            self.metrics.counter(
                "fill_events",
                {
                    "side": fill.side,
                },
            )

    def supported_types(self) -> set[str]:
        return {"MarketData", "Prediction", "PerformanceUpdate"}
