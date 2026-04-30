from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Any

from core.application.ports import (
    IEventBusPort,
    IEventHandlerPort,
    IStateRepository,
    ITimeSource,
)
from core.application.stages import (
    IExecutionStage,
    IPredictionStage,
    IRiskStage,
    ISignalStage,
    IStateEstimationStage,
)
from core.application.stores import RunContext
from core.domain.events import (
    Event,
    EngineErrorEvent,
    MarketDataEvent,
    PredictionEvent,
    SignalEvent,
)
from core.domain.models import Instrument

if TYPE_CHECKING:
    from core.application.stores import Config, PositionStore, SimulationWallet

logger = logging.getLogger(__name__)


# TRADING ENGINE (Facade / Orchestrator)
class TradingEngine(IEventHandlerPort):
    """
    Main trading orchestrator (facade pattern).

    Responsibilities:
    - Consume market data events
    - Orchestrate pipeline stages (market data -> state -> prediction -> signal -> risk -> execution)
    - Publish trading events to bus
    """

    def __init__(
        self,
        bus: IEventBusPort,
        state_store: IStateRepository,
        position_store: Any,
        wallet: Any,
        config: Any,
        time_source: ITimeSource,
        state_stage: IStateEstimationStage,
        pred_stage: IPredictionStage,
        signal_stage: ISignalStage,
        risk_stage: IRiskStage,
        exec_stage: IExecutionStage,
        replay_mode: bool = False,
    ) -> None:
        self.bus = bus
        self.time_source = time_source
        self.state_store = state_store
        self.position_store = position_store
        self.wallet = wallet
        self.config = config
        self.replay_mode = replay_mode
        self.state_stage = state_stage
        self.pred_stage = pred_stage
        self.signal_stage = signal_stage
        self.risk_stage = risk_stage
        self.exec_stage = exec_stage
        self.bar_index_counter: int = 0

    def supported_types(self) -> set[str]:
        return {
            MarketDataEvent.EVENT_TYPE,
        }

    def reset_bar_index(self, start: int = 0) -> None:
        self.bar_index_counter = start

    async def handle(self, event: Event) -> None:
        """
        Main event handler: process market data events.

        Uses isinstance() for strong type checking.
        All published events use ctx.correlation_id.
        """
        if not isinstance(event, MarketDataEvent):
            return

        ctx = self._build_context(event)
        raw_price = event.payload.get("price", 0.0)
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            price = 0.0

        if price <= 0:
            await self._publish_market_data_error(ctx, event, raw_price)
            logger.warning("Ignoring invalid market data price for %s: %r", ctx.instrument.symbol, raw_price)
            return

        logger.info("[PIPELINE] MarketData: %s @ %s | bar_index=%d", ctx.instrument.symbol, price, ctx.bar_index)
        self.wallet.mark_price(ctx.instrument, price)
        self.wallet.record(self.position_store, ts=event.ts_event)

        try:
            state = await self.state_stage.process(event, ctx)
            pred = await self.pred_stage.process(state, ctx)

            await self.bus.publish(
                PredictionEvent(
                    payload={
                        "symbol": ctx.instrument.symbol,
                        "mu": pred.mu,
                        "sigma": pred.sigma,
                        "prob_up": pred.prob_up,
                        "regime": pred.regime,
                        "horizon": pred.horizon,
                        "confirm_score": pred.confirm_score,
                    },
                    source="engine",
                    correlation_id=ctx.correlation_id,
                )
            )

            signal = await self.signal_stage.process(pred, ctx)
            logger.debug(
                "[PIPELINE] Signal: %s | %s @ %s",
                signal is not None,
                signal.side if signal else "NONE",
                signal.strength if signal else "?",
            )
            if signal is not None:
                await self.bus.publish(
                    SignalEvent(
                        payload={
                            "symbol": signal.instrument.symbol,
                            "side": signal.side,
                            "strength": signal.strength,
                            "confidence": signal.confidence,
                            "reason": signal.reason,
                        },
                        source="engine",
                        correlation_id=ctx.correlation_id,
                    )
                )

            order, risk_decision = await self.risk_stage.process(signal, ctx)
            logger.debug("[PIPELINE] Risk: order=%s decision=%s", order is not None, risk_decision is not None)

            if risk_decision is not None:
                for risk_event in risk_decision.events_to_emit:
                    await self.bus.publish(risk_event)

            if order is not None:
                price_hint = order.limit_price if order.limit_price is not None else ctx.wallet.last_prices.get(ctx.instrument.symbol)
                logger.info("[PIPELINE] EXECUTING: %s %s @ %s", order.side, order.qty, price_hint)
                await self.exec_stage.process(order, ctx)
            else:
                logger.debug("[PIPELINE] NO ORDER: risk stage blocked")

        except Exception as e:
            await self.bus.publish(
                EngineErrorEvent(
                    payload={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                        "stage": "pipeline",
                    },
                    source="engine",
                    correlation_id=ctx.correlation_id,
                )
            )
            logger.error("Engine pipeline error for %s: %s", ctx.instrument.symbol, e, exc_info=True)

    async def _publish_market_data_error(self, ctx: RunContext, event: MarketDataEvent, raw_price: object) -> None:
        await self.bus.publish(
            EngineErrorEvent(
                payload={
                    "error_type": "InvalidMarketData",
                    "error_message": f"Invalid non-positive market price: {raw_price!r}",
                    "traceback": "",
                    "stage": "market-data-validate",
                    "symbol": ctx.instrument.symbol,
                    "event_id": event.event_id,
                },
                source="engine",
                correlation_id=ctx.correlation_id,
            )
        )

    def _build_context(self, event: Event) -> RunContext:
        self.bar_index_counter += 1
        symbol = str(event.payload.get("symbol", self.config.symbols[0]))
        instrument = Instrument(symbol=symbol)
        return RunContext(
            correlation_id=event.correlation_id or event.event_id,
            instrument=instrument,
            state_store=self.state_store,
            position_store=self.position_store,
            wallet=self.wallet,
            config=self.config,
            time_source=self.time_source,
            bar_index=self.bar_index_counter,
        )
