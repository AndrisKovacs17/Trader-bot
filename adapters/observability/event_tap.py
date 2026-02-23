from __future__ import annotations

from core.analytics.services import ModelDiagnosticsService, PerformanceTracker
from core.application.ports import IEventHandlerPort
from core.domain.events import Event, OrderFilledEvent, PerformanceUpdateEvent, PredictionEvent
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
    ) -> None:
        self.performance_tracker = performance_tracker
        self.diagnostics = diagnostics
        self.metrics = metrics

    async def handle(self, event: Event) -> None:
        # Typed event dispatch for observability updates.
        if isinstance(event, PredictionEvent):
            pred = Prediction(
                mu=float(event.payload.get("mu", 0.0)),
                sigma=float(event.payload.get("sigma", 0.0)),
                prob_up=float(event.payload.get("prob_up", 0.5)),
                regime=str(event.payload.get("regime", "normal")),
                horizon=int(event.payload.get("horizon", 1)),
            )
            self.performance_tracker.on_prediction(pred)
            self.diagnostics.analyze_prediction(pred)
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
            self.performance_tracker.on_fill(fill, current_equity=current_equity)
            self.metrics.counter(
                "fill_events",
                {
                    "side": fill.side,
                },
            )

    def supported_types(self) -> set[str]:
        return {"Prediction", "PerformanceUpdate"}
