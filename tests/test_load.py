#!/usr/bin/env python3
"""Load tests: throughput and concurrency under high volume.

These tests verify that core components handle sustained load without
degradation, deadlocks, or race conditions.

Run with: pytest tests/test_load.py -v
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.time_source import SimulatedTimeSource
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.models import Fill, Instrument, Signal
from core.domain.risk import (
    ConfidenceRule,
    MaxAbsPositionRule,
    MaxNotionalRule,
    MaxQtyRule,
    RiskPolicy,
    SlippageGuardRule,
)
from core.domain.strategy import ThresholdStrategy
from core.ml.services import Prediction


BASE_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
INSTRUMENT = Instrument(symbol="BTCUSDT")


def _make_ctx(bar_index: int = 10) -> RunContext:
    wallet = SimulationWallet(initial_cash=1_000_000.0)
    wallet.mark_price(INSTRUMENT, 50_000.0)
    return RunContext(
        correlation_id=f"load-{bar_index}",
        instrument=INSTRUMENT,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(strategy={
            "threshold": 0.55,
            "min_edge": 0.01,
            "min_confidence": 0.51,
            "min_expected_value": 0.0,
            "min_signal_score": 0.0,
            "max_sigma": 10.0,
            "min_sigma": 1e-8,
            "max_strength": 0.0,
            "use_edge_score": False,
            "min_cooldown_seconds": 0.0,
            "min_bars_between_signals": 0,
            "allowed_horizons": [],
            "horizon_scale_mode": "inverse",
            "flip_extra_entry": 0.0,
        }),
        time_source=SimulatedTimeSource(BASE_NOW),
        bar_index=bar_index,
    )


def _policy() -> RiskPolicy:
    return RiskPolicy(rules=[
        ConfidenceRule(min_confidence=0.5),
        MaxQtyRule(max_qty=1.0, min_qty=0.0001),
        MaxNotionalRule(max_notional=1_000_000.0),
        MaxAbsPositionRule(max_abs_position_qty=100.0),
    ])


# ── load: strategy throughput ─────────────────────────────────────────────────

def test_strategy_handles_10k_predictions() -> None:
    """ThresholdStrategy must process 10 000 predictions without error."""
    strategy = ThresholdStrategy(threshold=0.55, min_edge=0.01, min_confidence=0.51)
    ctx = _make_ctx(bar_index=20_000)
    signals_generated = 0
    N = 10_000

    for i in range(N):
        # Alternate bullish / bearish predictions
        prob_up = 0.80 if i % 2 == 0 else 0.20
        mu = 0.02 if i % 2 == 0 else -0.02
        pred = Prediction(mu=mu, sigma=0.005, prob_up=prob_up, regime="trend", horizon=1)
        sig = strategy.on_prediction(pred, ctx)
        if sig is not None:
            signals_generated += 1

    assert signals_generated > 0


def test_strategy_throughput_above_5k_per_second() -> None:
    """Strategy must process ≥ 5 000 predictions per second on any hardware."""
    strategy = ThresholdStrategy(threshold=0.55, min_edge=0.01, min_confidence=0.51)
    ctx = _make_ctx(bar_index=100_000)
    pred = Prediction(mu=0.03, sigma=0.005, prob_up=0.80, regime="trend", horizon=1)
    N = 5_000

    start = time.perf_counter()
    for _ in range(N):
        strategy.on_prediction(pred, ctx)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Strategy too slow: {N} calls took {elapsed:.3f}s (threshold 1.0s)"


# ── load: risk policy throughput ──────────────────────────────────────────────

def test_risk_policy_handles_5k_evaluations() -> None:
    """RiskPolicy must evaluate 5 000 signals without error."""
    policy = _policy()
    ctx = _make_ctx()
    N = 5_000
    approved = 0

    for i in range(N):
        sig = Signal(
            instrument=INSTRUMENT,
            side="BUY",
            strength=0.1,
            confidence=0.8,
            horizon=1,
            reason=f"load|seq={i}",  # unique reason avoids dedup window
        )
        decision = policy.evaluate(sig, ctx)
        if decision.status.value == "APPROVED":
            approved += 1

    assert approved == N


def test_risk_policy_throughput_above_2k_per_second() -> None:
    """Risk policy must process ≥ 2 000 evaluations per second."""
    policy = _policy()
    ctx = _make_ctx()
    N = 2_000

    start = time.perf_counter()
    for i in range(N):
        sig = Signal(instrument=INSTRUMENT, side="BUY", strength=0.1, confidence=0.8,
                     horizon=1, reason=f"load|seq={i}")
        policy.evaluate(sig, ctx)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"RiskPolicy too slow: {N} calls took {elapsed:.3f}s"


# ── load: concurrent strategy calls (thread-safety) ──────────────────────────

def test_strategy_thread_safe_concurrent_calls() -> None:
    """ThresholdStrategy must be thread-safe: concurrent calls produce no exceptions."""
    strategy = ThresholdStrategy(threshold=0.55, min_edge=0.01, min_confidence=0.51,
                                 min_cooldown_seconds=0.0, min_bars_between_signals=0)
    THREADS = 8
    CALLS_PER_THREAD = 500
    errors: list[Exception] = []
    barrier = Barrier(THREADS)

    def _worker(thread_id: int) -> int:
        ctx = _make_ctx(bar_index=10 + thread_id * 1000)
        barrier.wait()  # synchronise start
        count = 0
        try:
            for i in range(CALLS_PER_THREAD):
                prob = 0.80 if i % 2 == 0 else 0.20
                pred = Prediction(mu=0.02, sigma=0.005, prob_up=prob,
                                  regime="trend", horizon=1)
                sig = strategy.on_prediction(pred, ctx)
                if sig is not None:
                    count += 1
        except Exception as exc:
            errors.append(exc)
        return count

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(_worker, t) for t in range(THREADS)]
        results = [f.result() for f in as_completed(futures)]

    assert errors == [], f"Thread errors: {errors}"
    assert sum(results) > 0


# ── load: concurrent risk policy evaluations ──────────────────────────────────

def test_risk_policy_thread_safe_concurrent_evaluations() -> None:
    """RiskPolicy must handle concurrent evaluations without deadlock or error."""
    THREADS = 4
    CALLS_PER_THREAD = 500
    errors: list[Exception] = []
    barrier = Barrier(THREADS)

    def _worker(thread_id: int = 0) -> int:
        policy = _policy()  # each thread gets its own stateless policy
        ctx = _make_ctx()
        barrier.wait()
        count = 0
        try:
            for i in range(CALLS_PER_THREAD):
                sig = Signal(instrument=INSTRUMENT, side="BUY", strength=0.1,
                             confidence=0.8, horizon=1, reason=f"concurrent|t={thread_id}|i={i}")
                d = policy.evaluate(sig, ctx)
                if d.status.value == "APPROVED":
                    count += 1
        except Exception as exc:
            errors.append(exc)
        return count

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = [pool.submit(_worker, t) for t in range(THREADS)]
        results = [f.result() for f in as_completed(futures)]

    assert errors == [], f"Thread errors: {errors}"
    total = sum(results)
    assert total == THREADS * CALLS_PER_THREAD


# ── load: position store under concurrent fills ───────────────────────────────

def test_position_store_sequential_fill_consistency() -> None:
    """Apply 1 000 sequential fills and verify final position is exact."""
    pos_store = PositionStore()
    # 500 buys + 500 sells of 0.001 each → flat
    for i in range(1000):
        side = "BUY" if i % 2 == 0 else "SELL"
        fill = Fill(order_id=f"fill-{i}", qty=0.001, price=50_000.0, fee=0.0, side=side)
        pos_store.apply_fill(fill, INSTRUMENT)

    pos = pos_store.get(INSTRUMENT)
    assert abs(pos.qty) < 1e-9, f"Expected flat position, got qty={pos.qty}"


# ── load: wallet equity under many mark-price updates ────────────────────────

def test_wallet_equity_stable_under_many_mark_price_updates() -> None:
    """Wallet equity computation is stable under 10k price updates."""
    wallet = SimulationWallet(initial_cash=100_000.0)
    pos_store = PositionStore()
    fill = Fill(order_id="f0", qty=1.0, price=50_000.0, fee=0.0, side="BUY")
    pos_store.apply_fill(fill, INSTRUMENT)
    wallet.apply_fill(fill, INSTRUMENT)

    for i in range(10_000):
        price = 50_000.0 + (i % 1000) * 10.0
        wallet.mark_price(INSTRUMENT, max(price, 0.01))

    equity = wallet.equity(pos_store)
    # Should be ~cash + 1 * last_price = (100000 - 50000) + last_price
    last_price = 50_000.0 + (9999 % 1000) * 10.0
    expected = (100_000.0 - 50_000.0) + last_price
    assert abs(equity - expected) < 1.0


# ── load: model_store save/load cycle ────────────────────────────────────────

def test_model_store_rapid_save_load_cycles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """model_store handles 100 rapid save→load cycles correctly."""
    import core.ml.model_store as ms
    import struct

    monkeypatch.setenv("KCA_MODEL_DIR", str(tmp_path))

    for i in range(100):
        version = f"v{i:04d}"
        encoded = version.encode("utf-8")
        weights = struct.pack(">Q", len(encoded)) + encoded + b"payload"
        ms.save_active(weights, version=version, metadata={"iter": i})

    result = ms.load_active()
    assert result is not None
    raw, meta = result
    assert meta["version"] == "v0099"
    assert meta["iter"] == 99
    assert ms.parse_version(raw) == "v0099"
