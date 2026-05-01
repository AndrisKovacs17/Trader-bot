#!/usr/bin/env python3
"""Advanced unit tests for ThresholdStrategy — covers branches missing from test_strategy.py.

Target: strategy.py 79 % → ≥ 92 %

Newly covered paths:
  - stop_loss_bps trigger (long & short)
  - stop_loss sigma-scaled path
  - max_holding_seconds exit
  - neutral_exit with position
  - neutral_exit_fraction < 1.0
  - neutral_exit disabled (enable_neutral_exit=False)
  - flip_extra_entry (close+flip)
  - allow_scale_in BUY with long pos
  - allow_scale_in SELL with short pos
  - same-side block (allow_scale_in=False, long pos, BUY pred)
  - short-entry disabled
  - allowed_horizons filter
  - enforce_mu_prob_agreement block
  - min_confirm_score filter
  - use_edge_score=True
  - horizon_scale_mode=none
  - max_strength cap
  - reset() clears diagnostics
  - diagnostics() returns hit counters
  - bar cooldown block
  - time cooldown block
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.testing.time_source import SimulatedTimeSource
from core.application.stores import Config, PositionStore, RunContext, SimulationWallet, StateStore
from core.domain.models import Fill, Instrument, Position
from core.domain.strategy import ThresholdStrategy
from core.ml.services import Prediction


# ── fixtures ──────────────────────────────────────────────────────────────────

BASE_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_ctx(
    symbol: str = "BTCUSDT",
    price: float = 50_000.0,
    bar_index: int = 10,
    now: datetime = BASE_NOW,
    extra_strategy: dict | None = None,
) -> RunContext:
    instr = Instrument(symbol=symbol)
    wallet = SimulationWallet(initial_cash=10_000.0)
    wallet.mark_price(instr, price)
    cfg_strategy: dict = {
        "threshold": 0.55,
        "min_edge": 0.01,
        "min_confidence": 0.51,
        "min_expected_value": 0.0,
        "min_signal_score": 0.0,
        "max_sigma": 5.0,
        "min_sigma": 1e-8,
        "max_strength": 0.0,
        "use_edge_score": False,
        "min_cooldown_seconds": 0.0,
        "min_bars_between_signals": 0,
        "allowed_horizons": [],
        "horizon_scale_mode": "inverse",
        "flip_extra_entry": 0.0,
        "stop_loss_bps": 0.0,
        "stop_loss_sigma_scale": 0.0,
        "bar_count": bar_index,
    }
    if extra_strategy:
        cfg_strategy.update(extra_strategy)
    return RunContext(
        correlation_id="test",
        instrument=instr,
        state_store=StateStore(),
        position_store=PositionStore(),
        wallet=wallet,
        config=Config(strategy=cfg_strategy),
        time_source=SimulatedTimeSource(now),
        bar_index=bar_index,
    )


def _open_long(ctx: RunContext, qty: float = 0.1, price: float = 50_000.0) -> None:
    fill = Fill(order_id="f1", qty=qty, price=price, fee=0.0, side="BUY")
    ctx.position_store.apply_fill(fill, ctx.instrument)
    ctx.wallet.mark_price(ctx.instrument, price)


def _open_short(ctx: RunContext, qty: float = 0.1, price: float = 50_000.0) -> None:
    fill = Fill(order_id="f2", qty=qty, price=price, fee=0.0, side="SELL")
    ctx.position_store.apply_fill(fill, ctx.instrument)
    ctx.wallet.mark_price(ctx.instrument, price)


def _pred(
    prob_up: float = 0.65,
    mu: float = 0.03,
    sigma: float = 0.01,
    horizon: int = 1,
    confirm_score: float = 0.0,
) -> Prediction:
    return Prediction(mu=mu, sigma=sigma, prob_up=prob_up, regime="normal",
                      horizon=horizon, confirm_score=confirm_score)


# ── stop-loss ─────────────────────────────────────────────────────────────────

def test_stop_loss_triggers_sell_on_long_position() -> None:
    ctx = _make_ctx(price=50_000.0, extra_strategy={"stop_loss_bps": 100.0})
    _open_long(ctx, qty=0.1, price=50_000.0)
    # Drop price to 49k → drawdown = 1000/50000 = 200 bps > 100 bps → stop
    ctx.wallet.mark_price(ctx.instrument, 49_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.65), ctx)
    assert sig is not None
    assert sig.side == "SELL"
    assert "stop_loss_exit" in sig.reason


def test_stop_loss_triggers_buy_on_short_position() -> None:
    ctx = _make_ctx(price=50_000.0, extra_strategy={"stop_loss_bps": 100.0})
    _open_short(ctx, qty=0.1, price=50_000.0)
    ctx.wallet.mark_price(ctx.instrument, 51_000.0)
    # short: drawdown = (51000-50000)/50000 = 200 bps > 100 bps
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.35), ctx)
    assert sig is not None
    assert sig.side == "BUY"


def test_stop_loss_not_triggered_when_drawdown_below_threshold() -> None:
    ctx = _make_ctx(price=49_990.0, extra_strategy={"stop_loss_bps": 500.0})
    _open_long(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    # drawdown only ~2 bps, stop is 500 bps
    sig = strategy.on_prediction(_pred(prob_up=0.65), ctx)
    # Should generate normal signal, not stop-loss
    assert sig is None or "stop_loss_exit" not in (sig.reason or "")


def test_stop_loss_sigma_scaled_widens_stop() -> None:
    """High sigma widens stop → stop should NOT fire even with drawdown."""
    ctx = _make_ctx(
        price=49_000.0,
        extra_strategy={
            "stop_loss_bps": 100.0,
            "stop_loss_sigma_scale": 1.0,
            "stop_loss_sigma_ref": 0.001,
        },
    )
    _open_long(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    # sigma=0.02 → mult = 0.02/0.001 = 20 → capped at 4 → effective stop = 400 bps
    # drawdown = 200 bps < 400 bps → no stop
    sig = strategy.on_prediction(_pred(prob_up=0.65, sigma=0.02), ctx)
    assert sig is None or "stop_loss_exit" not in (sig.reason or "")


# ── max_holding_seconds ───────────────────────────────────────────────────────

def test_max_holding_seconds_exit() -> None:
    ctx = _make_ctx(extra_strategy={"max_holding_seconds": 60.0})
    _open_long(ctx, qty=0.2, price=50_000.0)
    strategy = ThresholdStrategy()

    # First call: registers open_ts
    strategy.on_prediction(_pred(), ctx)

    # Advance time past max holding
    ctx.time_source.advance(timedelta(seconds=120))
    sig = strategy.on_prediction(_pred(), ctx)
    assert sig is not None
    assert sig.side == "SELL"
    assert "max_holding_exit" in sig.reason


# ── neutral exit ──────────────────────────────────────────────────────────────

def test_neutral_exit_with_open_long() -> None:
    ctx = _make_ctx()
    _open_long(ctx, qty=0.5, price=50_000.0)
    strategy = ThresholdStrategy()
    # prob_up=0.50 → no directional signal, but position open → neutral exit
    sig = strategy.on_prediction(_pred(prob_up=0.50, mu=0.0, sigma=0.01), ctx)
    assert sig is not None
    assert sig.side == "SELL"


def test_neutral_exit_fraction_partial() -> None:
    ctx = _make_ctx(extra_strategy={"neutral_exit_fraction": 0.5})
    _open_long(ctx, qty=1.0, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.50, mu=0.0, sigma=0.01), ctx)
    assert sig is not None
    assert abs(sig.strength - 0.5) < 1e-9


def test_neutral_exit_disabled() -> None:
    ctx = _make_ctx(extra_strategy={"enable_neutral_exit": False})
    _open_long(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.50, mu=0.0, sigma=0.01), ctx)
    assert sig is None


# ── flip / close ──────────────────────────────────────────────────────────────

def test_flip_long_to_short() -> None:
    """BUY position + SELL signal → close + flip."""
    ctx = _make_ctx(extra_strategy={"flip_extra_entry": 0.1})
    _open_long(ctx, qty=0.2, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.20, mu=-0.03, sigma=0.01), ctx)
    assert sig is not None
    assert sig.side == "SELL"
    # strength = close_qty (0.2) + flip_extra (0.1) capped at max(0, strength-close_qty)
    assert sig.strength >= 0.2


def test_flip_short_to_long() -> None:
    """Short position + BUY signal → close + flip."""
    ctx = _make_ctx(extra_strategy={"flip_extra_entry": 0.1})
    _open_short(ctx, qty=0.2, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig is not None
    assert sig.side == "BUY"
    assert sig.strength >= 0.2


# ── scale-in ──────────────────────────────────────────────────────────────────

def test_scale_in_allowed_long() -> None:
    ctx = _make_ctx(extra_strategy={"allow_scale_in": True})
    _open_long(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.05, sigma=0.01), ctx)
    assert sig is not None
    assert sig.side == "BUY"
    assert "scale_in_long" in sig.reason


def test_scale_in_blocked_when_disabled() -> None:
    ctx = _make_ctx(extra_strategy={"allow_scale_in": False})
    _open_long(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.05, sigma=0.01), ctx)
    assert sig is None


def test_scale_in_allowed_short() -> None:
    ctx = _make_ctx(extra_strategy={"allow_scale_in": True})
    _open_short(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.20, mu=-0.05, sigma=0.01), ctx)
    assert sig is not None
    assert sig.side == "SELL"
    assert "scale_in_short" in sig.reason


def test_same_side_short_blocked_when_scale_in_disabled() -> None:
    ctx = _make_ctx(extra_strategy={"allow_scale_in": False})
    _open_short(ctx, qty=0.1, price=50_000.0)
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.20, mu=-0.05, sigma=0.01), ctx)
    assert sig is None


# ── short-entry disabled ──────────────────────────────────────────────────────

def test_short_entry_disabled_blocks_sell() -> None:
    ctx = _make_ctx(extra_strategy={"allow_short_entries": False})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.20, mu=-0.03, sigma=0.01), ctx)
    assert sig is None


# ── allowed_horizons ──────────────────────────────────────────────────────────

def test_allowed_horizons_blocks_wrong_horizon() -> None:
    ctx = _make_ctx(extra_strategy={"allowed_horizons": [12, 24]})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.05, sigma=0.01, horizon=1), ctx)
    assert sig is None


def test_allowed_horizons_passes_correct_horizon() -> None:
    ctx = _make_ctx(extra_strategy={"allowed_horizons": [12]})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.05, sigma=0.01, horizon=12), ctx)
    assert sig is not None


# ── mu/prob agreement ─────────────────────────────────────────────────────────

def test_enforce_mu_prob_agreement_blocks_disagreement() -> None:
    ctx = _make_ctx(extra_strategy={"enforce_mu_prob_agreement": True})
    strategy = ThresholdStrategy()
    # prob_up=0.8 → BUY, but mu<0 → disagreement
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=-0.03, sigma=0.01), ctx)
    assert sig is None


def test_enforce_mu_prob_agreement_allows_agreement() -> None:
    ctx = _make_ctx(extra_strategy={"enforce_mu_prob_agreement": True})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig is not None


# ── min_confirm_score ─────────────────────────────────────────────────────────

def test_confirm_score_filters_weak_signal() -> None:
    ctx = _make_ctx(extra_strategy={"min_confirm_score": 0.5})
    strategy = ThresholdStrategy()
    # BUY pred, confirm_score=0.2 → directional_confirm=0.2 < 0.5 → blocked
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01, confirm_score=0.2), ctx)
    assert sig is None


def test_confirm_score_passes_strong_signal() -> None:
    ctx = _make_ctx(extra_strategy={"min_confirm_score": 0.5})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01, confirm_score=0.8), ctx)
    assert sig is not None


# ── use_edge_score ────────────────────────────────────────────────────────────

def test_use_edge_score_true_generates_signal() -> None:
    ctx = _make_ctx(extra_strategy={"use_edge_score": True, "min_signal_score": 0.001})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.05, sigma=0.01), ctx)
    assert sig is not None


def test_use_edge_score_false_uses_mu_confidence() -> None:
    ctx = _make_ctx(extra_strategy={"use_edge_score": False, "min_signal_score": 0.001})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.05, sigma=0.01), ctx)
    assert sig is not None


# ── horizon_scale_mode=none ───────────────────────────────────────────────────

def test_horizon_scale_mode_none_ignores_horizon() -> None:
    ctx = _make_ctx(extra_strategy={"horizon_scale_mode": "none"})
    strategy = ThresholdStrategy()
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01, horizon=96), ctx)
    assert sig is not None
    # strength = |mu|/sigma = 3.0 (not divided by horizon)
    assert abs(sig.strength - 3.0) < 1e-9


# ── max_strength cap ──────────────────────────────────────────────────────────

def test_max_strength_caps_extreme_values() -> None:
    ctx = _make_ctx(extra_strategy={"max_strength": 1.0})
    strategy = ThresholdStrategy()
    # |mu|/sigma = 100 → should be capped at 1.0
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=1.0, sigma=0.01), ctx)
    assert sig is not None
    assert sig.strength <= 1.0 + 1e-9


# ── bar cooldown ──────────────────────────────────────────────────────────────

def test_bar_cooldown_blocks_rapid_signals() -> None:
    ctx = _make_ctx(extra_strategy={"min_bars_between_signals": 5}, bar_index=10)
    strategy = ThresholdStrategy()
    sig1 = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig1 is not None
    # Same bar_index → cooldown not expired
    sig2 = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig2 is None


def test_bar_cooldown_passes_after_enough_bars() -> None:
    ctx = _make_ctx(extra_strategy={"min_bars_between_signals": 3}, bar_index=10)
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    ctx.bar_index = 14  # 4 bars later > 3
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig is not None


# ── time cooldown ─────────────────────────────────────────────────────────────

def test_time_cooldown_blocks_within_window() -> None:
    ctx = _make_ctx(extra_strategy={"min_cooldown_seconds": 300.0})
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    ctx.time_source.advance(timedelta(seconds=10))
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig is None


def test_time_cooldown_passes_after_window() -> None:
    ctx = _make_ctx(extra_strategy={"min_cooldown_seconds": 60.0})
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    ctx.time_source.advance(timedelta(seconds=120))
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig is not None


# ── reset / diagnostics ───────────────────────────────────────────────────────

def test_reset_clears_cooldown_state() -> None:
    ctx = _make_ctx(extra_strategy={"min_cooldown_seconds": 300.0})
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    strategy.reset()
    # After reset cooldown is gone → should fire again
    sig = strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    assert sig is not None


def test_reset_clears_diagnostics() -> None:
    ctx = _make_ctx()
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    strategy.reset()
    diag = strategy.diagnostics()
    assert diag["decision_hits"] == {}
    assert diag["block_reason_hits"] == {}


def test_diagnostics_tracks_decisions() -> None:
    ctx = _make_ctx()
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.01), ctx)
    diag = strategy.diagnostics()
    assert diag["decision_hits"].get("enter", 0) >= 1


def test_diagnostics_tracks_block_reasons() -> None:
    ctx = _make_ctx(extra_strategy={"max_sigma": 0.001})
    strategy = ThresholdStrategy()
    strategy.on_prediction(_pred(prob_up=0.80, mu=0.03, sigma=0.5), ctx)
    diag = strategy.diagnostics()
    assert sum(diag["block_reason_hits"].values()) >= 1


# ── id / warmup_required ─────────────────────────────────────────────────────

def test_strategy_id_returns_string() -> None:
    s = ThresholdStrategy()
    assert isinstance(s.id(), str)
    assert len(s.id()) > 0


def test_warmup_required_default() -> None:
    s = ThresholdStrategy()
    assert isinstance(s.warmup_required(), int)
    assert s.warmup_required() >= 0
