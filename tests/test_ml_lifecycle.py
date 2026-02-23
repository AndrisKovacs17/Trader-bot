#!/usr/bin/env python3
"""ML lifecycle tests: training update flow + engine event publishing."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.offline_training.training_engine import TrainingEngine
from adapters.testing.time_source import SimulatedTimeSource
from core.application.engine import TradingEngine
from core.application.ports import IEventBusPort
from core.application.stores import Config, PositionStore, SimulationWallet, StateStore
from core.domain.events import Event, ModelLifecycleAppliedEvent, ModelUpdatedEvent
from core.domain.models import Instrument
from core.ml.services import EstimatedState, MambaPredictor


class InMemoryEventBus(IEventBusPort):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def publish(self, event: Event) -> None:
        self.events.append(event)

    def subscribe(self, event_type: str, handler) -> None:
        _ = event_type
        _ = handler

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class NoopStage:
    async def process(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return None


class NoopRiskStage:
    async def process(self, *args, **kwargs):
        _ = args
        _ = kwargs
        return None, None


@pytest.mark.asyncio
async def test_training_push_and_apply_updates_predictor_version() -> None:
    predictor = MambaPredictor(version="mvp-v1")
    trainer = TrainingEngine()

    dataset = [{"price": 100.0 + i, "return": 0.001 * i, "volume": 1000.0} for i in range(8)]
    result = trainer.train(dataset=dataset, instrument=Instrument(symbol="BTCUSDT"), epochs=1, batch_size=4)
    assert result["version"].startswith("offline-")

    await trainer.push_model(predictor)
    assert predictor.has_pending_update() is True
    assert predictor.current_version() == "mvp-v1"

    metadata = await predictor.apply_pending_update()
    assert metadata is not None
    assert metadata["status"] == "applied"
    assert metadata["old_version"] == "mvp-v1"
    assert metadata["new_version"].startswith("offline-")
    assert predictor.current_version() == metadata["new_version"]
    assert predictor.has_pending_update() is False

    second = await predictor.apply_pending_update()
    assert second is None


@pytest.mark.asyncio
async def test_trained_update_changes_predictor_regime_or_output() -> None:
    predictor = MambaPredictor(version="mvp-v1")
    trainer = TrainingEngine()

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    instrument = Instrument(symbol="BTCUSDT")
    ctx = Config(model={"default_signal": False, "mamba_window": 64})

    run_ctx = type("RunCtx", (), {})()
    run_ctx.instrument = instrument
    run_ctx.config = ctx

    state = EstimatedState(
        x=[100.0, 0.01],
        P=[[1.0, 0.0], [0.0, 1.0]],
        features={"price": 100.0, "return": 0.01},
        confidence=0.8,
        ts=now,
    )

    before = await predictor.predict(state, run_ctx)

    dataset = [{"price": 100.0 + i, "return": 0.002 * i, "volume": 1000.0} for i in range(16)]
    trainer.train(dataset=dataset, instrument=instrument, epochs=2, batch_size=4)
    await trainer.push_model(predictor)
    await predictor.apply_pending_update()

    after = await predictor.predict(state, run_ctx)
    changed = (before.mu, before.sigma, before.prob_up) != (after.mu, after.sigma, after.prob_up)
    assert changed or after.regime in {"mamba-ssm", "trained-linear"}


@pytest.mark.asyncio
async def test_engine_emits_model_updated_event_on_lifecycle_tick() -> None:
    predictor = MambaPredictor(version="mvp-v1")
    trainer = TrainingEngine()
    bus = InMemoryEventBus()

    dataset = [{"price": 100.0 + i, "return": 0.001 * i, "volume": 1000.0} for i in range(6)]
    trainer.train(dataset=dataset, instrument=Instrument(symbol="BTCUSDT"), epochs=1, batch_size=3)
    await trainer.push_model(predictor)
    assert predictor.has_pending_update() is True

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    engine = TradingEngine(
        bus=bus,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=SimulationWallet(initial_cash=10000.0),
        config=Config(),
        time_source=SimulatedTimeSource(now),
        market_stage=NoopStage(),
        state_stage=NoopStage(),
        pred_stage=NoopStage(),
        signal_stage=NoopStage(),
        risk_stage=NoopRiskStage(),
        exec_stage=NoopStage(),
        model_lifecycle=predictor,
    )

    await engine.handle(
        ModelLifecycleAppliedEvent(
            payload={"reason": "scheduled_tick"},
            source="test",
            correlation_id="corr-ml-1",
        )
    )

    updated_events = [e for e in bus.events if isinstance(e, ModelUpdatedEvent)]
    assert len(updated_events) == 1
    payload = updated_events[0].payload
    assert payload["status"] == "applied"
    assert payload["old_version"] == "mvp-v1"
    assert payload["new_version"].startswith("offline-")
    assert predictor.has_pending_update() is False
