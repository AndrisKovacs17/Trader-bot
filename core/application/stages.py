from __future__ import annotations

from typing import Optional, Protocol, TYPE_CHECKING

from core.application.ports import IExecutionUseCase, IStateRepository
from core.domain.events import MarketDataEvent
from core.domain.models import Order, RiskDecision, Signal
from core.domain.risk import RiskPolicy
from core.domain.strategy import IStrategy
from core.ml.services import EstimatedState, IPredictor, IStateEstimator, Prediction

if TYPE_CHECKING:
    from core.application.stores import RunContext


# Stage interface-ek: a pipeline olvashatóvá tétele.
class IMarketDataStage(Protocol):
    async def process(self, md: MarketDataEvent, ctx: RunContext) -> None:
        ...

    def name(self) -> str:
        ...


class IStateEstimationStage(Protocol):
    async def process(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        ...

    def name(self) -> str:
        ...


class IPredictionStage(Protocol):
    async def process(self, state: EstimatedState, ctx: RunContext) -> Prediction:
        ...

    def name(self) -> str:
        ...


class ISignalStage(Protocol):
    async def process(self, pred: Prediction, ctx: RunContext) -> Optional[Signal]:
        ...

    def name(self) -> str:
        ...


class IRiskStage(Protocol):
    async def process(self, signal: Optional[Signal], ctx: RunContext) -> tuple[Optional[Order], Optional[RiskDecision]]:
        ...

    def name(self) -> str:
        ...


class IExecutionStage(Protocol):
    async def process(self, order: Optional[Order], ctx: RunContext) -> None:
        ...

    def name(self) -> str:
        ...


class MarketDataStage:
    async def process(self, md: MarketDataEvent, ctx: RunContext) -> None:
        # MVP-ben ez a stage csak placeholder normalizálásra/validációra.
        _ = md
        _ = ctx

    def name(self) -> str:
        return "market-data"


# State estimation stage: estimator + store írás.
class StateEstimationStage:
    def __init__(self, estimator: IStateEstimator, state_store: IStateRepository) -> None:
        self.estimator = estimator
        self.state_store = state_store

    async def process(self, md: MarketDataEvent, ctx: RunContext) -> EstimatedState:
        state = await self.estimator.update(md, ctx)
        self.state_store.save(ctx.instrument, state)
        return state

    def name(self) -> str:
        return "state-estimation"


# Prediction stage.
class PredictionStage:
    def __init__(self, predictor: IPredictor) -> None:
        self.predictor = predictor

    async def process(self, state: EstimatedState, ctx: RunContext) -> Prediction:
        return await self.predictor.predict(state, ctx)

    def name(self) -> str:
        return "prediction"


# Signal stage.
class SignalStage:
    def __init__(self, strategy: IStrategy) -> None:
        self.strategy = strategy

    async def process(self, pred: Prediction, ctx: RunContext) -> Optional[Signal]:
        return self.strategy.on_prediction(pred, ctx)

    def name(self) -> str:
        return "signal"


# Risk stage.
class RiskStage:
    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    async def process(self, signal: Optional[Signal], ctx: RunContext) -> tuple[Optional[Order], Optional[RiskDecision]]:
        if signal is None:
            return None, None
        decision = self.policy.evaluate(signal, ctx)
        return decision.order, decision

    def name(self) -> str:
        return "risk"


# Execution stage.
class ExecutionStage:
    def __init__(self, execution: IExecutionUseCase) -> None:
        self.execution = execution

    async def process(self, order: Optional[Order], ctx: RunContext) -> None:
        if order is None:
            return
        await self.execution.submit(order, ctx)

    def name(self) -> str:
        return "execution"
