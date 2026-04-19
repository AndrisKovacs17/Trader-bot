#!/usr/bin/env python3
"""Pytest risk policy coverage: block paths, adjustments, short direction, dedup, counters."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.time_source import SimulatedTimeSource
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.models import Fill, Instrument, RiskReasonCode, RiskSeverity, RiskStatus, Signal
from core.domain.risk import (
    ConfidenceRule,
    MaxAbsPositionRule,
    MaxNotionalRule,
    MaxQtyRule,
    RiskPolicy,
    SlippageGuardRule,
)


@pytest.fixture
def base_instrument() -> Instrument:
    return Instrument(symbol="BTCUSDT")


@pytest.fixture
def base_ctx(base_instrument: Instrument) -> RunContext:
    state_store = StateStore()
    position_store = PositionStore()
    wallet = SimulationWallet(initial_cash=10000.0)
    wallet.mark_price(base_instrument, 50000.0)
    config = Config(
        risk_limits={
            "max_qty": 1.0,
            "min_qty": 0.0001,
            "min_confidence": 0.5,
            "max_notional": 1000.0,
            "max_abs_position_qty": 2.0,
            "max_slippage_bps": 50.0,
        }
    )
    return RunContext(
        correlation_id="corr-1",
        instrument=base_instrument,
        state_store=state_store,
        position_store=position_store,
        wallet=wallet,
        config=config,
        time_source=SimulatedTimeSource(datetime.now(timezone.utc)),
    )


@pytest.fixture
def default_policy() -> RiskPolicy:
    return RiskPolicy(
        rules=[
            ConfidenceRule(min_confidence=0.5),
            SlippageGuardRule(max_slippage_bps=50.0),
            MaxQtyRule(max_qty=1.0, min_qty=0.0001),
            MaxNotionalRule(max_notional=1000.0),
            MaxAbsPositionRule(max_abs_position_qty=2.0),
        ]
    )


def make_signal(
    instrument: Instrument,
    *,
    side: str = "BUY",
    strength: float = 0.1,
    confidence: float = 0.9,
    reason: str = "entry",
    horizon: int = 1,
) -> Signal:
    return Signal(
        instrument=instrument,
        side=side,
        strength=strength,
        confidence=confidence,
        horizon=horizon,
        reason=reason,
    )


@pytest.mark.parametrize(
    "confidence, expected_status, expected_reason",
    [
        (0.2, RiskStatus.BLOCKED, RiskReasonCode.LOW_CONFIDENCE.value),
        (0.9, RiskStatus.APPROVED, None),
    ],
)
def test_confidence_rule_paths(
    base_ctx: RunContext,
    default_policy: RiskPolicy,
    confidence: float,
    expected_status: RiskStatus,
    expected_reason: str | None,
) -> None:
    signal = make_signal(base_ctx.instrument, confidence=confidence)
    decision = default_policy.evaluate(signal, base_ctx)
    assert decision.status == expected_status
    payload = decision.events_to_emit[0].payload
    if expected_reason is not None:
        assert payload.get("reason_code") == expected_reason
        assert payload.get("severity") == RiskSeverity.MEDIUM.value


def test_max_qty_rule_blocks_on_non_positive_strength(base_ctx: RunContext) -> None:
    policy = RiskPolicy(rules=[MaxQtyRule(max_qty=1.0, min_qty=0.0001)])
    signal = make_signal(base_ctx.instrument, strength=0.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    payload = decision.events_to_emit[0].payload
    assert payload.get("reason_code") == RiskReasonCode.STRENGTH_NON_POSITIVE.value


def test_max_abs_position_rule_blocks_with_existing_position(base_ctx: RunContext) -> None:
    fill = Fill(order_id="seed", qty=2.0, price=50000.0, fee=0.0, side="BUY")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)

    policy = RiskPolicy(rules=[MaxAbsPositionRule(max_abs_position_qty=2.0)])
    signal = make_signal(base_ctx.instrument, side="BUY", strength=0.5)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    payload = decision.events_to_emit[0].payload
    assert payload.get("reason_code") == RiskReasonCode.MAX_ABS_POSITION_EXCEEDED.value


def test_max_abs_position_rule_short_direction_adjusts_qty(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["allow_short_selling"] = True
    fill = Fill(order_id="seed", qty=1.9, price=50000.0, fee=0.0, side="BUY")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)

    policy = RiskPolicy(rules=[MaxAbsPositionRule(max_abs_position_qty=2.0)])
    signal = make_signal(base_ctx.instrument, side="SELL", strength=10.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.APPROVED
    assert decision.order is not None
    assert math.isclose(decision.order.qty, 3.9, rel_tol=0.0, abs_tol=1e-12)


def test_max_abs_position_rule_long_to_short_flip_is_limited(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["allow_short_selling"] = True
    fill = Fill(order_id="seed", qty=1.5, price=50000.0, fee=0.0, side="BUY")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)

    policy = RiskPolicy(rules=[MaxAbsPositionRule(max_abs_position_qty=2.0)])
    signal = make_signal(base_ctx.instrument, side="SELL", strength=10.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.APPROVED
    assert decision.order is not None
    # from +1.5 long to -2.0 short max requires 3.5 sell qty
    assert math.isclose(decision.order.qty, 3.5, rel_tol=0.0, abs_tol=1e-12)


def test_max_abs_position_rule_short_extend_beyond_limit_blocks(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["allow_short_selling"] = True
    fill = Fill(order_id="seed", qty=2.0, price=50000.0, fee=0.0, side="SELL")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)

    policy = RiskPolicy(rules=[MaxAbsPositionRule(max_abs_position_qty=2.0)])
    signal = make_signal(base_ctx.instrument, side="SELL", strength=0.5)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    payload = decision.events_to_emit[0].payload
    assert payload.get("reason_code") == RiskReasonCode.MAX_ABS_POSITION_EXCEEDED.value


def test_slippage_guard_blocks(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["max_slippage_bps"] = 0.001
    policy = RiskPolicy(rules=[SlippageGuardRule(max_slippage_bps=0.001)])
    signal = make_signal(base_ctx.instrument, strength=1.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    payload = decision.events_to_emit[0].payload
    assert payload.get("reason_code") == RiskReasonCode.SLIPPAGE_BPS_EXCEEDED.value


def test_notional_adjusts_qty_with_isclose(base_ctx: RunContext) -> None:
    policy = RiskPolicy(rules=[MaxNotionalRule(max_notional=1000.0)])
    signal = make_signal(base_ctx.instrument, strength=1.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.APPROVED
    assert decision.order is not None
    assert math.isclose(decision.order.qty, 1000.0 / 50000.0, rel_tol=0.0, abs_tol=1e-12)


def test_partial_adjustment_and_min_qty_interaction_blocks(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["min_qty"] = 0.0001
    base_ctx.config.risk_limits["max_notional"] = 0.1  # adjusted qty = 0.1/50000 = 0.000002
    policy = RiskPolicy(rules=[MaxNotionalRule(max_notional=1000.0)])
    signal = make_signal(base_ctx.instrument, strength=1.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    payload = decision.events_to_emit[0].payload
    assert payload.get("rule") == "post-adjustment-min-qty"
    assert payload.get("reason_code") == RiskReasonCode.QTY_NON_POSITIVE.value


def test_duplicate_signal_blocked_and_counters(base_ctx: RunContext, default_policy: RiskPolicy) -> None:
    signal = make_signal(base_ctx.instrument, strength=0.1, confidence=0.9)

    first = default_policy.evaluate(signal, base_ctx)
    second = default_policy.evaluate(signal, base_ctx)

    assert first.status == RiskStatus.APPROVED
    assert second.status == RiskStatus.BLOCKED
    payload = second.events_to_emit[0].payload
    assert payload.get("reason_code") == RiskReasonCode.DUPLICATE_SIGNAL.value

    diagnostics = default_policy.diagnostics()
    assert diagnostics["block_reason_hits"].get(RiskReasonCode.DUPLICATE_SIGNAL.value, 0) >= 1
    assert diagnostics["rule_hits"].get("confidence", 0) >= 2


def test_dedup_key_contains_model_version(base_ctx: RunContext, default_policy: RiskPolicy) -> None:
    signal = make_signal(base_ctx.instrument, strength=0.1, confidence=0.9)

    first = default_policy.evaluate(signal, base_ctx)
    assert first.status == RiskStatus.APPROVED

    base_ctx.config.model["version"] = "mamba-v2"
    second = default_policy.evaluate(signal, base_ctx)
    assert second.status == RiskStatus.APPROVED


def test_concurrent_rule_hit_counters_are_consistent(base_ctx: RunContext) -> None:
    policy = RiskPolicy(
        rules=[
            ConfidenceRule(min_confidence=0.5),
            MaxQtyRule(max_qty=1.0, min_qty=0.0001),
        ]
    )
    base_ctx.config.risk_limits["min_qty"] = 0.0001

    def _run_once(i: int) -> RiskStatus:
        local_ctx = RunContext(
            correlation_id=f"corr-{i}",
            instrument=base_ctx.instrument,
            state_store=base_ctx.state_store,
            position_store=base_ctx.position_store,
            wallet=base_ctx.wallet,
            config=base_ctx.config,
            time_source=base_ctx.time_source,
        )
        signal = make_signal(base_ctx.instrument, strength=0.2, confidence=0.9, reason=f"r{i}")
        return policy.evaluate(signal, local_ctx).status

    runs = 200
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(_run_once, range(runs)))

    assert all(status == RiskStatus.APPROVED for status in statuses)
    diagnostics = policy.diagnostics()
    assert diagnostics["rule_hits"].get("confidence", 0) == runs
    assert diagnostics["rule_hits"].get("max-qty", 0) == runs


def test_insufficient_cash_blocks_buy(base_ctx: RunContext) -> None:
    base_ctx.wallet.cash = 1.0
    policy = RiskPolicy(rules=[])
    signal = make_signal(base_ctx.instrument, side="BUY", strength=0.1)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    assert decision.events_to_emit[0].payload.get("reason_code") == RiskReasonCode.INSUFFICIENT_CASH.value


def test_sell_is_clamped_to_existing_position_when_short_disabled(base_ctx: RunContext) -> None:
    fill = Fill(order_id="seed", qty=1.5, price=50000.0, fee=0.0, side="BUY")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)
    base_ctx.config.risk_limits["allow_short_selling"] = False
    policy = RiskPolicy(rules=[])
    signal = make_signal(base_ctx.instrument, side="SELL", strength=10.0)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.APPROVED
    assert decision.order is not None
    assert math.isclose(decision.order.qty, 1.5, rel_tol=0.0, abs_tol=1e-12)


def test_short_entry_blocked_when_short_disabled(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["allow_short_selling"] = False
    policy = RiskPolicy(rules=[])
    signal = make_signal(base_ctx.instrument, side="SELL", strength=0.5)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    assert decision.events_to_emit[0].payload.get("reason_code") == RiskReasonCode.SHORT_SELL_NOT_ALLOWED.value


def test_max_qty_below_min_is_blocked(base_ctx: RunContext) -> None:
    policy = RiskPolicy(rules=[MaxQtyRule(max_qty=1.0, min_qty=0.25)])
    signal = make_signal(base_ctx.instrument, strength=0.1)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    assert decision.events_to_emit[0].payload.get("reason_code") == RiskReasonCode.QTY_NON_POSITIVE.value


def test_same_side_weak_reentry_is_blocked(base_ctx: RunContext) -> None:
    fill = Fill(order_id="seed", qty=0.5, price=50000.0, fee=0.0, side="BUY")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)
    base_ctx.config.risk_limits["weak_reentry_edge_threshold"] = 0.08
    base_ctx.config.risk_limits["max_same_side_scale_in"] = -1

    policy = RiskPolicy(rules=[])
    weak_reentry_signal = make_signal(
        base_ctx.instrument,
        side="BUY",
        strength=0.1,
        confidence=0.9,
        reason="threshold-strategy|edge=0.0500|intent=scale_in_long",
    )

    decision = policy.evaluate(weak_reentry_signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    assert decision.events_to_emit[0].payload.get("reason_code") == RiskReasonCode.WEAK_REENTRY_SIGNAL.value


def test_same_side_scale_in_limit_is_enforced(base_ctx: RunContext) -> None:
    fill = Fill(order_id="seed", qty=0.5, price=50000.0, fee=0.0, side="BUY")
    base_ctx.position_store.apply_fill(fill, base_ctx.instrument)
    base_ctx.config.risk_limits["max_same_side_scale_in"] = 1
    base_ctx.config.risk_limits["same_side_min_edge_improvement"] = 0.0
    base_ctx.config.risk_limits["weak_reentry_edge_threshold"] = 0.0

    policy = RiskPolicy(rules=[])
    first = make_signal(
        base_ctx.instrument,
        side="BUY",
        strength=0.1,
        confidence=0.9,
        reason="threshold-strategy|edge=0.2000|intent=scale_in_long",
    )
    second = make_signal(
        base_ctx.instrument,
        side="BUY",
        strength=0.1,
        confidence=0.9,
        reason="threshold-strategy|edge=0.2500|intent=scale_in_long",
    )

    first_decision = policy.evaluate(first, base_ctx)
    assert first_decision.status == RiskStatus.APPROVED

    second_decision = policy.evaluate(second, base_ctx)
    assert second_decision.status == RiskStatus.BLOCKED
    assert second_decision.events_to_emit[0].payload.get("reason_code") == RiskReasonCode.SCALE_IN_LIMIT_EXCEEDED.value


def test_shadow_mode_blocks_trading_before_threshold(base_ctx: RunContext) -> None:
    base_ctx.config.risk_limits["shadow_predictions_required"] = 10
    base_ctx.bar_index = 5
    policy = RiskPolicy(rules=[])
    signal = make_signal(base_ctx.instrument, side="BUY", strength=0.1, confidence=0.9)

    decision = policy.evaluate(signal, base_ctx)
    assert decision.status == RiskStatus.BLOCKED
    assert decision.events_to_emit[0].payload.get("reason_code") == RiskReasonCode.SHADOW_MODE_ACTIVE.value
