#!/usr/bin/env python3
"""ThresholdStrategy behavior tests with config-driven guards and sizing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.time_source import SimulatedTimeSource
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.models import Fill, Instrument
from core.domain.strategy import ThresholdStrategy
from core.ml.services import Prediction


@pytest.fixture
def strategy() -> ThresholdStrategy:
    return ThresholdStrategy(threshold=0.55)


@pytest.fixture
def ctx() -> RunContext:
    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=10000.0)
    wallet.mark_price(instrument, 50000.0)
    cfg = Config(
        strategy={
            "threshold": 0.55,
            "min_edge": 0.02,
            "min_confidence": 0.55,
            "min_expected_value": 0.0,
            "max_sigma": 1.0,
            "min_sigma": 1e-8,
            "max_strength": 0.0,
            "use_edge_score": False,
            "min_cooldown_seconds": 0.0,
            "min_bars_between_signals": 0,
            "allowed_horizons": [],
            "bar_count": 1,
        }
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return RunContext(
        correlation_id="corr-1",
        instrument=instrument,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=cfg,
        time_source=SimulatedTimeSource(now),
        bar_index=1,  # Explicit bar index to satisfy warmup_required=1
    )


def make_pred(prob_up: float, mu: float, sigma: float, horizon: int = 1) -> Prediction:
    return Prediction(mu=mu, sigma=sigma, prob_up=prob_up, regime="normal", horizon=horizon)


def test_threshold_comes_from_config(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["threshold"] = 0.60
    pred = make_pred(prob_up=0.56, mu=0.02, sigma=0.01)
    signal = strategy.on_prediction(pred, ctx)
    assert signal is None


def test_edge_confidence_expected_value_guards(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["min_edge"] = 0.06
    pred = make_pred(prob_up=0.54, mu=0.03, sigma=0.01)
    assert strategy.on_prediction(pred, ctx) is None

    ctx.config.strategy["min_edge"] = 0.01
    ctx.config.strategy["min_confidence"] = 0.80
    pred = make_pred(prob_up=0.60, mu=0.03, sigma=0.01)
    assert strategy.on_prediction(pred, ctx) is None

    ctx.config.strategy["min_confidence"] = 0.55
    ctx.config.strategy["min_expected_value"] = 0.05
    pred = make_pred(prob_up=0.65, mu=0.01, sigma=0.01)
    assert strategy.on_prediction(pred, ctx) is None


def test_uncertainty_guard_blocks_high_sigma(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["max_sigma"] = 0.2
    pred = make_pred(prob_up=0.8, mu=0.04, sigma=0.5)
    assert strategy.on_prediction(pred, ctx) is None


def test_volatility_aware_strength(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    pred = make_pred(prob_up=0.8, mu=0.02, sigma=0.01, horizon=1)
    signal = strategy.on_prediction(pred, ctx)
    assert signal is not None
    assert signal.side == "BUY"
    # strength = abs(mu)/sigma = 2.0
    assert abs(signal.strength - 2.0) < 1e-12


def test_position_aware_skip_same_direction(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.position_store.apply_fill(Fill(order_id="o1", qty=1.0, price=50000.0, fee=0.0, side="BUY"), ctx.instrument)
    pred = make_pred(prob_up=0.8, mu=0.02, sigma=0.01)
    assert strategy.on_prediction(pred, ctx) is None


def test_position_aware_allows_scale_in_when_enabled(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["allow_scale_in"] = True
    ctx.position_store.apply_fill(Fill(order_id="o1", qty=1.0, price=50000.0, fee=0.0, side="BUY"), ctx.instrument)
    pred = make_pred(prob_up=0.8, mu=0.02, sigma=0.01)
    signal = strategy.on_prediction(pred, ctx)
    assert signal is not None
    assert signal.side == "BUY"
    assert "intent=scale_in_long" in signal.reason


def test_neutral_exit_closes_open_position_in_middle_band(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["enable_neutral_exit"] = True
    ctx.config.strategy["neutral_exit_fraction"] = 1.0
    ctx.position_store.apply_fill(Fill(order_id="o1", qty=0.8, price=50000.0, fee=0.0, side="BUY"), ctx.instrument)

    pred = make_pred(prob_up=0.53, mu=0.001, sigma=0.02)
    signal = strategy.on_prediction(pred, ctx)

    assert signal is not None
    assert signal.side == "SELL"
    assert abs(signal.strength - 0.8) < 1e-12
    assert "intent=neutral_exit" in signal.reason


def test_strategy_diagnostics_track_blocks_and_decisions(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    blocked_pred = make_pred(prob_up=0.53, mu=0.001, sigma=0.02)
    assert strategy.on_prediction(blocked_pred, ctx) is None

    ok_pred = make_pred(prob_up=0.8, mu=0.02, sigma=0.01)
    signal = strategy.on_prediction(ok_pred, ctx)
    assert signal is not None

    diag = strategy.diagnostics()
    assert diag["block_reason_hits"].get("between-threshold-bands", 0) >= 1
    assert diag["decision_hits"].get("enter", 0) >= 1


def test_mu_prob_sign_agreement_gate_blocks_mismatch(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["enforce_mu_prob_agreement"] = True
    pred = make_pred(prob_up=0.80, mu=-0.01, sigma=0.02)
    signal = strategy.on_prediction(pred, ctx)
    assert signal is None


def test_max_holding_forces_exit(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["enable_neutral_exit"] = False
    ctx.config.strategy["max_holding_seconds"] = 1.0
    ctx.position_store.apply_fill(Fill(order_id="o1", qty=0.4, price=50000.0, fee=0.0, side="BUY"), ctx.instrument)

    # First call initializes holding timer for open position.
    _ = strategy.on_prediction(make_pred(prob_up=0.55, mu=0.001, sigma=0.02), ctx)

    ctx.time_source.set(ctx.time_source.now() + timedelta(seconds=2))
    forced_exit = strategy.on_prediction(make_pred(prob_up=0.55, mu=0.001, sigma=0.02), ctx)

    assert forced_exit is not None
    assert forced_exit.side == "SELL"
    assert abs(forced_exit.strength - 0.4) < 1e-12
    assert "intent=max_holding_exit" in forced_exit.reason


def test_position_aware_close_or_flip_signal(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.position_store.apply_fill(Fill(order_id="o1", qty=1.5, price=50000.0, fee=0.0, side="BUY"), ctx.instrument)
    pred = make_pred(prob_up=0.2, mu=0.01, sigma=0.05)
    signal = strategy.on_prediction(pred, ctx)
    assert signal is not None
    assert signal.side == "SELL"
    assert signal.strength >= 1.5
    assert "intent=close_or_flip_long_to_short" in signal.reason


def test_cooldown_blocks_second_signal(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["min_cooldown_seconds"] = 60.0
    pred = make_pred(prob_up=0.8, mu=0.02, sigma=0.01)

    first = strategy.on_prediction(pred, ctx)
    assert first is not None

    second = strategy.on_prediction(pred, ctx)
    assert second is None

    ctx.time_source.set(ctx.time_source.now() + timedelta(seconds=61))
    third = strategy.on_prediction(pred, ctx)
    assert third is not None


def test_horizon_filter(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    ctx.config.strategy["allowed_horizons"] = [2, 4]
    pred_h1 = make_pred(prob_up=0.8, mu=0.02, sigma=0.01, horizon=1)
    pred_h2 = make_pred(prob_up=0.8, mu=0.02, sigma=0.01, horizon=2)
    assert strategy.on_prediction(pred_h1, ctx) is None
    assert strategy.on_prediction(pred_h2, ctx) is not None


def test_warmup_enforced(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    # Test with bar_index=0 (below warmup_required=1)
    ctx_zero = RunContext(
        correlation_id=ctx.correlation_id,
        instrument=ctx.instrument,
        state_store=ctx.state_store,
        position_store=ctx.position_store,
        wallet=ctx.wallet,
        config=ctx.config,
        time_source=ctx.time_source,
        bar_index=0,
    )
    pred = make_pred(prob_up=0.8, mu=0.02, sigma=0.01)
    assert strategy.on_prediction(pred, ctx_zero) is None

    # Test with bar_index=1 (meets warmup_required)
    ctx_one = RunContext(
        correlation_id=ctx.correlation_id,
        instrument=ctx.instrument,
        state_store=ctx.state_store,
        position_store=ctx.position_store,
        wallet=ctx.wallet,
        config=ctx.config,
        time_source=ctx.time_source,
        bar_index=1,
    )
    assert strategy.on_prediction(pred, ctx_one) is not None


def test_min_sigma_prevents_extreme_strength(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    """Very small sigma should be clamped to min_sigma to prevent division-by-zero strength."""
    ctx.config.strategy["min_sigma"] = 0.001
    pred_tiny_sigma = make_pred(prob_up=0.8, mu=0.02, sigma=1e-10)  # Extreme tiny sigma
    signal = strategy.on_prediction(pred_tiny_sigma, ctx)
    assert signal is not None
    # Strength = |mu| / max(sigma, min_sigma) / horizon
    # = 0.02 / 0.001 / 1 = 20.0, but should be bounded
    assert signal.strength > 0.0
    assert signal.strength < 1000.0  # Should not explode to infinity


def test_max_strength_caps_output(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    """max_strength parameter should cap final strength value."""
    ctx.config.strategy["max_strength"] = 5.0
    pred_high = make_pred(prob_up=0.9, mu=0.5, sigma=0.01)  # Would produce high strength
    signal = strategy.on_prediction(pred_high, ctx)
    assert signal is not None
    assert signal.strength <= 5.0


def test_use_edge_score_mode(strategy: ThresholdStrategy, ctx: RunContext) -> None:
    """use_edge_score=True should use edge*|mu| instead of |mu|*confidence for signal_score."""
    ctx.config.strategy["use_edge_score"] = True
    ctx.config.strategy["min_signal_score"] = 0.005
    
    # Edge = 0.25, mu = 0.02 → score = 0.25 * 0.02 = 0.005 (should pass)
    pred_edge = make_pred(prob_up=0.75, mu=0.02, sigma=0.01)
    signal = strategy.on_prediction(pred_edge, ctx)
    assert signal is not None
    
    # Edge = 0.05, mu = 0.02 → score = 0.05 * 0.02 = 0.001 (should block)
    pred_low_edge = make_pred(prob_up=0.55, mu=0.02, sigma=0.01)
    assert strategy.on_prediction(pred_low_edge, ctx) is None


def test_thread_safe_cooldown_tracking() -> None:
    """Concurrent on_prediction calls should not race on cooldown state."""
    from concurrent.futures import ThreadPoolExecutor
    
    strategy = ThresholdStrategy(threshold=0.55, min_cooldown_seconds=0.5)
    instrument = Instrument(symbol="BTCUSDT")
    wallet = SimulationWallet(initial_cash=10000.0)
    wallet.mark_price(instrument, 50000.0)
    
    cfg = Config(strategy={"min_cooldown_seconds": 0.5})
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ctx = RunContext(
        correlation_id="corr-1",
        instrument=instrument,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=cfg,
        time_source=SimulatedTimeSource(now),
        bar_index=1,
    )
    
    pred = Prediction(mu=0.02, sigma=0.01, prob_up=0.8, regime="normal", horizon=1)
    
    def worker(_id: int) -> bool:
        sig = strategy.on_prediction(pred, ctx)
        return sig is not None
    
    # First call should emit signal
    first = strategy.on_prediction(pred, ctx)
    assert first is not None
    
    # Concurrent calls within cooldown window should all block
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, range(10)))
    
    # All should be blocked by cooldown
    assert all(r is False for r in results)
