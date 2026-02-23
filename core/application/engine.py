from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone

from core.application.ports import (
    IEventBusPort,
    IEventHandlerPort,
    ITimeSource,
)
from core.application.stages import (
    ExecutionStage,
    IExecutionStage,
    IMarketDataStage,
    IPredictionStage,
    IRiskStage,
    ISignalStage,
    IStateEstimationStage,
    MarketDataStage,
    PredictionStage,
    RiskStage,
    SignalStage,
    StateEstimationStage,
)
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.events import (
    Event,
    EngineErrorEvent,
    MarketDataEvent,
    ModelLifecycleAppliedEvent,
    ModelUpdatedEvent,
    PredictionEvent,
    SignalEvent,
)
from core.domain.models import Instrument
from core.ml.services import IModelLifecycle


# =====================================================
# TRADING ENGINE (Facade / Orchestrator)
# =====================================================

class TradingEngine(IEventHandlerPort):
    """
    Main trading orchestrator (facade pattern).
    
    Responsibilities:
    - Consume market data events
    - Orchestrate pipeline stages (market data -> state -> prediction -> signal -> risk -> execution)
    - Publish trading events to bus
    - Coordinate model lifecycle (apply pending ML model updates via dedicated event)
    
    Key Design Points:
    - Uses isinstance() for strong type checking instead of string comparison
    - Explicit bar_index counter to avoid side effects from wallet.record()
    - RiskStage returns tuple(order, decision) to eliminate implicit state dependency
    - No re-wrapping of events; publishes subclass instances directly
    - None checks before stage execution
    """
    
    def __init__(
        self,
        bus: IEventBusPort,
        state_store: StateStore,
        position_store: PositionStore,
        wallet: SimulationWallet,
        config: Config,
        time_source: ITimeSource,
        market_stage: IMarketDataStage,
        state_stage: IStateEstimationStage,
        pred_stage: IPredictionStage,
        signal_stage: ISignalStage,
        risk_stage: IRiskStage,
        exec_stage: IExecutionStage,
        model_lifecycle: IModelLifecycle | None = None,
        replay_mode: bool = False,
    ) -> None:
        self.bus = bus
        self.time_source = time_source
        self.state_store = state_store
        self.position_store = position_store
        self.wallet = wallet
        self.config = config
        self.replay_mode = replay_mode
        self.market_stage = market_stage
        self.state_stage = state_stage
        self.pred_stage = pred_stage
        self.signal_stage = signal_stage
        self.risk_stage = risk_stage
        self.exec_stage = exec_stage
        self.model_lifecycle = model_lifecycle  # Optional ML model lifecycle manager
        
        # Explicit bar_index counter (point 6)
        self.bar_index_counter: int = 0

        # Execution is wired via injected execution stage (UML-aligned)

    def supported_types(self) -> set[str]:
        """
        Return set of supported event type strings.
        
        NOTE: Design compromize—engine uses isinstance() for strong typing,
        but event bus typically routes by string event_type.
        Future: Migrate to supported_classes() → set[type[Event]] if bus supports type-based routing.
        
        Maps directly from event subclass EVENT_TYPE constants for consistency
        and to avoid hardcoded string duplication.
        """
        return {
            MarketDataEvent.EVENT_TYPE,   # "MarketData"
            ModelLifecycleAppliedEvent.EVENT_TYPE,  # "ModelLifecycleApplied"
        }

    def reset_bar_index(self, start: int = 0) -> None:
        """
        Reset bar_index_counter for replay or new session.
        
        Point 5: Use this during replay initialization to reset bar numbering.
        Ensures strategy cooldown and horizon logic references correct bar indices.
        
        Args:
            start: Bar number to start from (typically 0 for new session, or last_bar+1 for continuation)
        """
        self.bar_index_counter = start

    async def handle(self, event: Event) -> None:
        """
        Main event handler: process market data or model lifecycle events.
        
        Uses isinstance() for strong type checking (point 1).
        Routes ModelLifecycleAppliedEvent to dedicated handler (point 7).
        All published events use ctx.correlation_id (point 6).
        """
        # Point 7: Handle model lifecycle updates via dedicated event (not mid-pipeline)
        if isinstance(event, ModelLifecycleAppliedEvent):
            await self._handle_model_lifecycle_applied(event)
            return

        # Point 2: Expect MarketDataEvent type directly
        if not isinstance(event, MarketDataEvent):
            return

        # Build context with explicit bar_index (point 6)
        ctx = self._build_context(event)
        logging.info(f"[PIPELINE] MarketData: {ctx.instrument.symbol} @ {event.payload.get('price', '?')} | bar_index={ctx.bar_index}")
        
        # Update wallet mark prices
        self.wallet.mark_price(ctx.instrument, float(event.payload.get("price", 0.0)))
        self.wallet.record(self.position_store)

        # =====================================================
        # PIPELINE: Market -> State -> Prediction -> Signal -> Risk -> Execution
        # Point 3: Pipeline exception boundary - failure doesn't kill engine
        # =====================================================
        try:
            await self.market_stage.process(event, ctx)
            state = await self.state_stage.process(event, ctx)
            pred = await self.pred_stage.process(state, ctx)
            
            # Point 3: Event creation without redundant event_type (default from EVENT_TYPE)
            # Point 6: Use ctx.correlation_id consistently
            await self.bus.publish(
                PredictionEvent(
                    payload={
                        "symbol": ctx.instrument.symbol,
                        "mu": pred.mu,
                        "sigma": pred.sigma,
                        "prob_up": pred.prob_up,
                        "regime": pred.regime,
                        "horizon": pred.horizon,
                    },
                    source="engine",
                    correlation_id=ctx.correlation_id,
                )
            )

            # Generate signal
            signal = await self.signal_stage.process(pred, ctx)
            logging.debug(f"[PIPELINE] Signal: {signal is not None} | {signal.side if signal else 'NONE'} @ {signal.strength if signal else '?'}")
            if signal is not None:
                # Point 3: No redundant event_type parameter
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

            # Apply risk checks - Point 3: Explicit tuple return eliminates implicit state
            order, risk_decision = await self.risk_stage.process(signal, ctx)
            logging.debug(f"[PIPELINE] Risk: order={order is not None}, decision={risk_decision is not None if risk_decision else 'NONE'}")
            
            # Point 4: Publish RiskEvent subclasses directly, no re-wrapping
            if risk_decision is not None:
                for risk_event in risk_decision.events_to_emit:
                    await self.bus.publish(risk_event)
            
            # Point 8: Explicit None check before execution
            if order is not None:
                price_hint = order.limit_price if order.limit_price is not None else ctx.wallet.last_prices.get(ctx.instrument.symbol)
                logging.info(f"[PIPELINE] EXECUTING: {order.side} {order.qty} @ {price_hint}")
                await self.exec_stage.process(order, ctx)
            else:
                logging.debug(f"[PIPELINE] NO ORDER: risk stage blocked")
            
        except Exception as e:
            # Publish error event for observability and replay
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
            # Log error but don't re-raise—engine stays alive
            logging.error(f"Engine pipeline error for {ctx.instrument.symbol}: {e}", exc_info=True)

    def _build_context(self, event: Event) -> RunContext:
        """
        Build execution context from event.
        
        Point 6: Use explicit bar_index_counter instead of len(wallet.history)
        to avoid side effects if wallet.record() is called elsewhere.
        
        NOTE: bar_index_counter only increments on MarketData ticks, not on OrderFilled events.
        This is intentional—fill events share the same bar_index as their triggering market tick,
        ensuring consistency in strategy/cooldown logic that depends on bar indices.
        """
        self.bar_index_counter += 1
        symbol = event.payload.get("symbol", self.config.symbols[0])
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

    async def _handle_model_lifecycle_applied(self, event: ModelLifecycleAppliedEvent) -> None:
        """
        Handle model lifecycle event: apply pending ML model updates.
        
        Point 7: Dedicated event path for model updates (not mid-pipeline).
        ModelLifecycleAppliedEvent is the trigger; we trust the lifecycle manager's decision.
        No redundant has_pending_update() check—the event is the contract.
        
        Point 2: Publish ModelUpdatedEvent with version metadata (old_version, new_version, applied_at, etc.)
        from apply_pending_update() return value.
        """
        if not self.model_lifecycle:
            return
        
        # Event is the trigger; apply directly and capture version metadata
        update_metadata = await self.model_lifecycle.apply_pending_update()
        
        # Build payload with version info for audit trail
        payload = {
            "status": "applied",
            "applied_at": datetime.now(timezone.utc).isoformat(),
        }
        if update_metadata:
            payload.update(update_metadata)  # Merge old_version, new_version, model_id, etc.
        
        # Publish confirmation using ModelUpdatedEvent (no redundant event_type)
        # Point 4: Ensure correlation_id with fallback to event_id
        await self.bus.publish(
            ModelUpdatedEvent(
                payload=payload,
                source="engine",
                correlation_id=event.correlation_id or event.event_id,
            )
        )