from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Protocol, TYPE_CHECKING

from core.domain.models import Signal
from core.ml.services import Prediction

if TYPE_CHECKING:
    from core.application.stores import RunContext

logger = logging.getLogger(__name__)


# Strategy interface: predikcióból signal készítése.
class IStrategy(Protocol):
    def on_prediction(self, pred: Prediction, ctx: RunContext) -> Signal | None:
        ...

    def id(self) -> str:
        ...

    def warmup_required(self) -> int:
        ...


# Legegyszerűbb threshold stratégia MVP-hez.
@dataclass(slots=True)
class ThresholdStrategy(IStrategy):
    threshold: float = 0.55
    min_edge: float = 0.02
    min_confidence: float = 0.55
    min_signal_score: float = 0.0
    max_sigma: float = 1.0
    min_sigma: float = 1e-8  # Avoid division-by-zero, extreme strength
    max_strength: float = 0.0  # 0.0=disabled; >0 caps strength for safety
    use_edge_score: bool = False  # True: edge*|mu|, False: |mu|*confidence
    min_cooldown_seconds: float = 0.0
    min_bars_between_signals: int = 0
    horizon_scale_mode: str = "inverse"  # inverse|none
    flip_extra_entry: float = 0.0
    min_confirm_score: float = -1.0  # -1.0=disabled; 0.0=neutral ok; 0.33=2/3 categories agree
    stop_loss_bps: float = 0.0        # 0=disabled; e.g. 150=1.5% hard stop on open position
    stop_loss_sigma_scale: float = 0.0  # 0=fixed stop; >0 widens/narrows stop proportional to pred.sigma
    stop_loss_sigma_ref: float = 0.001  # reference sigma (≈ typical 5m BTC return vol)
    _last_signal_ts_by_symbol: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)
    _last_signal_bar_by_symbol: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _position_open_ts_by_symbol: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)
    _block_reason_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _decision_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _state_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _config_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _inc_counter(self, counter: dict[str, int], key: str) -> None:
        with self._state_lock:
            counter[key] = counter.get(key, 0) + 1

    def diagnostics(self) -> dict[str, dict[str, int]]:
        with self._state_lock:
            return {
                "block_reason_hits": dict(self._block_reason_hits),
                "decision_hits": dict(self._decision_hits),
            }

    def _log_block(
        self,
        *,
        symbol: str,
        reason: str,
        prob_up: float,
        mu: float,
        sigma: float,
        desired_side: str,
        threshold: float,
    ) -> None:
        reason_bucket = reason.split(":", 1)[0] if ":" in reason else reason
        self._inc_counter(self._block_reason_hits, reason_bucket)
        logger.debug(
            "[STRATEGY] BLOCKED: reason=%s | symbol=%s | prob_up=%.6f | mu=%.6f | sigma=%.6f | desired_side=%s | threshold=%.6f",
            reason,
            symbol,
            prob_up,
            mu,
            sigma,
            desired_side or "NONE",
            threshold,
        )

    @staticmethod
    def _bar_count(ctx: RunContext) -> int:
        # Explicit contract: RunContext MUST have bar_index attribute.
        # Fallback csak backward compatibility-hez; production enginenek
        # explicit bar_index-et kell biztosítania.
        if not hasattr(ctx, "bar_index"):
            # Implicit fallback, de logolható warning production-ben
            return int(ctx.config.get("strategy.bar_count", len(ctx.wallet.history)))
        return int(ctx.bar_index)

    def _in_cooldown(self, symbol: str, now: datetime, bar_count: int, min_seconds: float, min_bars: int) -> bool:
        # Thread-safe read of mutable state dictionaries.
        with self._state_lock:
            last_ts = self._last_signal_ts_by_symbol.get(symbol)
            last_bar = self._last_signal_bar_by_symbol.get(symbol)
        
        if last_ts is not None and min_seconds > 0:
            if (now - last_ts).total_seconds() < min_seconds:
                return True
        if last_bar is not None and min_bars > 0:
            if (bar_count - last_bar) < min_bars:
                return True
        return False

    def reset(self) -> None:
        """Reset strategy internal state for deterministic replay boundaries.
        
        CRITICAL: Engine MUST call this at replay session start to clear cooldown state.
        
        Multi-threading:
        - Thread-safe: uses _state_lock to protect dictionary mutations.
        - Single-threaded engine: lock overhead minimal.
        - Multi-threaded engine: prevents race conditions during concurrent on_prediction calls.
        
        Replay contract:
        - Call reset() before replaying historical bars.
        - Ensures cooldown tracking starts fresh.
        - Does NOT reset config-driven parameters (threshold, min_edge, etc.).
        """
        with self._state_lock:
            self._last_signal_ts_by_symbol = {}
            self._last_signal_bar_by_symbol = {}
            self._position_open_ts_by_symbol = {}
            self._block_reason_hits = {}
            self._decision_hits = {}

    def on_prediction(self, pred: Prediction, ctx: RunContext) -> Signal | None:
        # Determinisztikus replay megjegyzés:
        # ugyanazt a strategy.* és model.signal_threshold configot kell használni replay során.
        threshold = float(
            ctx.config.get(
                "model.signal_threshold",
                ctx.config.get("strategy.threshold", self.threshold),
            )
        )
        min_edge = float(ctx.config.get("strategy.min_edge", self.min_edge))
        min_confidence = float(ctx.config.get("strategy.min_confidence", self.min_confidence))
        min_signal_score = float(
            ctx.config.get(
                "strategy.min_signal_score",
                ctx.config.get("strategy.min_expected_value", self.min_signal_score),
            )
        )
        max_sigma = float(ctx.config.get("strategy.max_sigma", self.max_sigma))
        min_sigma = float(ctx.config.get("strategy.min_sigma", self.min_sigma))
        max_strength = float(ctx.config.get("strategy.max_strength", self.max_strength))
        min_cooldown_seconds = float(ctx.config.get("strategy.min_cooldown_seconds", self.min_cooldown_seconds))
        min_bars_between_signals = int(ctx.config.get("strategy.min_bars_between_signals", self.min_bars_between_signals))
        horizon_scale_mode = str(ctx.config.get("strategy.horizon_scale_mode", self.horizon_scale_mode))
        flip_extra_entry = float(ctx.config.get("strategy.flip_extra_entry", self.flip_extra_entry))
        min_confirm_score = float(ctx.config.get("strategy.min_confirm_score", self.min_confirm_score))
        allow_short_entries = self._config_bool(ctx.config.get("strategy.allow_short_entries", True), default=True)
        allow_scale_in = self._config_bool(ctx.config.get("strategy.allow_scale_in", False), default=False)
        enforce_mu_prob_agreement = self._config_bool(
            ctx.config.get("strategy.enforce_mu_prob_agreement", False),
            default=False,
        )
        enable_neutral_exit = self._config_bool(ctx.config.get("strategy.enable_neutral_exit", True), default=True)
        neutral_exit_fraction = min(
            1.0,
            max(0.0, float(ctx.config.get("strategy.neutral_exit_fraction", 1.0))),
        )
        neutral_exit_respects_cooldown = self._config_bool(
            ctx.config.get("strategy.neutral_exit_respects_cooldown", False),
            default=False,
        )
        max_holding_seconds = max(float(ctx.config.get("strategy.max_holding_seconds", 0.0)), 0.0)
        max_holding_respects_cooldown = self._config_bool(
            ctx.config.get("strategy.max_holding_respects_cooldown", False),
            default=False,
        )
        allowed_horizons = ctx.config.get("strategy.allowed_horizons", [])

        bar_count = self._bar_count(ctx)
        if bar_count < self.warmup_required():
            self._log_block(
                symbol=ctx.instrument.symbol,
                reason=f"warmup-required:{bar_count}<{self.warmup_required()}",
                prob_up=0.5,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side="",
                threshold=threshold,
            )
            return None

        prob_up = self._clamp01(float(pred.prob_up))
        confidence = max(prob_up, 1.0 - prob_up)
        edge = abs(prob_up - 0.5)
        desired_side = "BUY" if prob_up >= threshold else "SELL" if prob_up <= (1.0 - threshold) else ""
        symbol = ctx.instrument.symbol
        now = ctx.now()
        pos = ctx.position_store.get(ctx.instrument)

        with self._state_lock:
            open_ts = self._position_open_ts_by_symbol.get(symbol)
            if abs(float(pos.qty)) <= 0:
                self._position_open_ts_by_symbol.pop(symbol, None)
                open_ts = None
            elif open_ts is None:
                self._position_open_ts_by_symbol[symbol] = now
                open_ts = now

        # ── Price-based stop-loss (optionally sigma-scaled) ───────────────────
        _stop_bps = float(ctx.config.get("strategy.stop_loss_bps", self.stop_loss_bps))
        if _stop_bps > 0.0 and abs(float(pos.qty)) > 0:
            _cur_price = float(ctx.wallet.last_prices.get(symbol, 0.0))
            _avg_price = float(pos.avg_price)
            if _cur_price > 0 and _avg_price > 0:
                _drawdown = (
                    (_avg_price - _cur_price) / _avg_price if pos.qty > 0
                    else (_cur_price - _avg_price) / _avg_price
                )
                # Dynamic stop: scale threshold by pred.sigma / sigma_ref.
                # High uncertainty → wider stop (avoids noise stops in volatile regimes).
                # Low uncertainty  → tighter stop (protect gains in calm regimes).
                _sigma_scale = float(ctx.config.get("strategy.stop_loss_sigma_scale", self.stop_loss_sigma_scale))
                _sigma_ref = max(float(ctx.config.get("strategy.stop_loss_sigma_ref", self.stop_loss_sigma_ref)), 1e-9)
                if _sigma_scale > 0.0 and float(pred.sigma) > 0.0:
                    _sigma_mult = max(0.5, min(4.0, float(pred.sigma) / _sigma_ref))
                    _effective_stop_bps = _stop_bps * _sigma_mult
                else:
                    _effective_stop_bps = _stop_bps
                if _drawdown >= _effective_stop_bps / 10_000.0:
                    _exit_side = "SELL" if float(pos.qty) > 0 else "BUY"
                    with self._state_lock:
                        self._last_signal_ts_by_symbol[symbol] = now
                        self._last_signal_bar_by_symbol[symbol] = bar_count
                    self._inc_counter(self._decision_hits, "stop_loss_exit")
                    return Signal(
                        instrument=ctx.instrument,
                        side=_exit_side,
                        strength=abs(float(pos.qty)),
                        confidence=1.0,
                        horizon=pred.horizon,
                        reason=(
                            f"{self.id()}|intent=stop_loss_exit"
                            f"|drawdown_bps={_drawdown * 10000:.1f}"
                            f"|stop_bps={_effective_stop_bps:.0f}"
                            f"|avg={_avg_price:.2f}|cur={_cur_price:.2f}"
                            f"|sigma={float(pred.sigma):.5f}"
                        ),
                    )

        if max_holding_seconds > 0 and abs(float(pos.qty)) > 0 and open_ts is not None:
            held_seconds = max((now - open_ts).total_seconds(), 0.0)
            if held_seconds >= max_holding_seconds:
                if max_holding_respects_cooldown and self._in_cooldown(
                    symbol,
                    now,
                    bar_count,
                    min_cooldown_seconds,
                    min_bars_between_signals,
                ):
                    self._log_block(
                        symbol=symbol,
                        reason="max-holding-cooldown-active",
                        prob_up=prob_up,
                        mu=float(pred.mu),
                        sigma=float(pred.sigma),
                        desired_side=desired_side,
                        threshold=threshold,
                    )
                    return None

                exit_side = "SELL" if float(pos.qty) > 0 else "BUY"
                exit_strength = abs(float(pos.qty))
                if exit_strength > 0:
                    with self._state_lock:
                        self._last_signal_ts_by_symbol[symbol] = now
                        self._last_signal_bar_by_symbol[symbol] = bar_count
                    self._inc_counter(self._decision_hits, "max_holding_exit")
                    return Signal(
                        instrument=ctx.instrument,
                        side=exit_side,
                        strength=exit_strength,
                        confidence=confidence,
                        horizon=pred.horizon,
                        reason=(
                            f"{self.id()}|threshold={threshold:.4f}|edge={edge:.4f}|"
                            f"intent=max_holding_exit|sigma={float(pred.sigma):.6f}|h={int(pred.horizon)}|"
                            f"bar={bar_count}|scale={horizon_scale_mode}"
                        ),
                    )

        if not desired_side:
            if enable_neutral_exit and abs(float(pos.qty)) > 0:
                if neutral_exit_respects_cooldown and self._in_cooldown(
                    symbol,
                    now,
                    bar_count,
                    min_cooldown_seconds,
                    min_bars_between_signals,
                ):
                    self._log_block(
                        symbol=symbol,
                        reason="neutral-exit-cooldown-active",
                        prob_up=prob_up,
                        mu=float(pred.mu),
                        sigma=float(pred.sigma),
                        desired_side=desired_side,
                        threshold=threshold,
                    )
                    return None

                exit_side = "SELL" if float(pos.qty) > 0 else "BUY"
                exit_strength = abs(float(pos.qty)) * neutral_exit_fraction
                if exit_strength <= 0:
                    self._log_block(
                        symbol=symbol,
                        reason="neutral-exit-zero-size",
                        prob_up=prob_up,
                        mu=float(pred.mu),
                        sigma=float(pred.sigma),
                        desired_side=desired_side,
                        threshold=threshold,
                    )
                    return None

                with self._state_lock:
                    self._last_signal_ts_by_symbol[symbol] = now
                    self._last_signal_bar_by_symbol[symbol] = bar_count

                self._inc_counter(self._decision_hits, "neutral_exit")
                return Signal(
                    instrument=ctx.instrument,
                    side=exit_side,
                    strength=exit_strength,
                    confidence=confidence,
                    horizon=pred.horizon,
                    reason=(
                        f"{self.id()}|threshold={threshold:.4f}|edge={edge:.4f}|"
                        f"intent=neutral_exit|sigma={float(pred.sigma):.6f}|h={int(pred.horizon)}|"
                        f"bar={bar_count}|scale={horizon_scale_mode}"
                    ),
                )

            self._log_block(
                symbol=symbol,
                reason="between-threshold-bands",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None

        if enforce_mu_prob_agreement:
            mu_value = float(pred.mu)
            mu_sign = 1 if mu_value > 0 else -1 if mu_value < 0 else 0
            prob_sign = 1 if prob_up > 0.5 else -1 if prob_up < 0.5 else 0
            if mu_sign != 0 and prob_sign != 0 and mu_sign != prob_sign:
                self._log_block(
                    symbol=symbol,
                    reason="mu-prob-sign-mismatch",
                    prob_up=prob_up,
                    mu=mu_value,
                    sigma=float(pred.sigma),
                    desired_side=desired_side,
                    threshold=threshold,
                )
                return None

        # Confirmation filter: require multi-indicator agreement before entering.
        # confirm_score is in [-1, 1]; directional_confirm flips sign for SELL.
        if min_confirm_score > -1.0:
            _confirm = float(getattr(pred, "confirm_score", 0.0))
            directional_confirm = _confirm if desired_side == "BUY" else -_confirm
            if directional_confirm < min_confirm_score:
                self._log_block(
                    symbol=symbol,
                    reason=f"confirm-score:{directional_confirm:.3f}<{min_confirm_score:.3f}",
                    prob_up=prob_up,
                    mu=float(pred.mu),
                    sigma=float(pred.sigma),
                    desired_side=desired_side,
                    threshold=threshold,
                )
                return None
        
        # Signal quality scoring:
        # use_edge_score=False (default): |mu| * confidence (volatility-neutral quality)
        # use_edge_score=True: edge * |mu| (directional conviction * magnitude)
        use_edge_score = bool(ctx.config.get("strategy.use_edge_score", self.use_edge_score))
        if use_edge_score:
            signal_score = edge * abs(float(pred.mu))
        else:
            signal_score = abs(float(pred.mu)) * confidence

        logger.debug(
            "[STRATEGY] prob_up=%.6f | conf=%.6f | edge=%.6f | mu=%.6f | sigma=%.6f | score=%.6f",
            prob_up, confidence, edge, float(pred.mu), float(pred.sigma), signal_score,
        )
        logger.debug(
            "[STRATEGY] Thresholds: min_conf=%.6f | min_edge=%.6f | min_score=%.6f | max_sigma=%.6f | threshold=%.6f | desired_side=%s",
            min_confidence, min_edge, min_signal_score, max_sigma, threshold, desired_side or "NONE",
        )

        if confidence < min_confidence:
            self._log_block(
                symbol=ctx.instrument.symbol,
                reason=f"confidence:{confidence:.6f}<{min_confidence:.6f}",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None
        if edge < min_edge:
            self._log_block(
                symbol=ctx.instrument.symbol,
                reason=f"edge:{edge:.6f}<{min_edge:.6f}",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None
        if signal_score < min_signal_score:
            self._log_block(
                symbol=ctx.instrument.symbol,
                reason=f"signal-score:{signal_score:.6f}<{min_signal_score:.6f}",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None
        if float(pred.sigma) > max_sigma:
            self._log_block(
                symbol=ctx.instrument.symbol,
                reason=f"sigma:{float(pred.sigma):.6f}>{max_sigma:.6f}",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None

        if isinstance(allowed_horizons, (list, tuple, set)) and len(allowed_horizons) > 0:
            if int(pred.horizon) not in {int(x) for x in allowed_horizons}:
                self._log_block(
                    symbol=ctx.instrument.symbol,
                    reason=f"horizon:{int(pred.horizon)}-not-allowed",
                    prob_up=prob_up,
                    mu=float(pred.mu),
                    sigma=float(pred.sigma),
                    desired_side=desired_side,
                    threshold=threshold,
                )
                return None

        if self._in_cooldown(symbol, now, bar_count, min_cooldown_seconds, min_bars_between_signals):
            self._log_block(
                symbol=symbol,
                reason="cooldown-active",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None

        # Volatility clamping: prevent extreme strength from tiny sigma.
        # min_sigma default 1e-8, but configurable for production tuning.
        vol = max(float(pred.sigma), min_sigma)
        risk_scaled_strength = abs(float(pred.mu)) / vol
        if horizon_scale_mode == "inverse":
            horizon_scaled_strength = risk_scaled_strength / max(1, int(pred.horizon))
        else:
            horizon_scaled_strength = risk_scaled_strength
        
        # Base strength application with optional upper cap.
        strength = max(0.0, horizon_scaled_strength)
        if max_strength > 0.0:
            strength = min(strength, max_strength)

        intent = "enter"
        if desired_side == "SELL" and (pos.qty <= 0) and not allow_short_entries:
            self._log_block(
                symbol=symbol,
                reason="short-entry-disabled",
                prob_up=prob_up,
                mu=float(pred.mu),
                sigma=float(pred.sigma),
                desired_side=desired_side,
                threshold=threshold,
            )
            return None
        if desired_side == "BUY" and pos.qty > 0:
            if not allow_scale_in:
                self._log_block(
                    symbol=symbol,
                    reason="same-side-position-long",
                    prob_up=prob_up,
                    mu=float(pred.mu),
                    sigma=float(pred.sigma),
                    desired_side=desired_side,
                    threshold=threshold,
                )
                return None
            intent = "scale_in_long"
        if desired_side == "SELL" and pos.qty < 0:
            if not allow_scale_in:
                self._log_block(
                    symbol=symbol,
                    reason="same-side-position-short",
                    prob_up=prob_up,
                    mu=float(pred.mu),
                    sigma=float(pred.sigma),
                    desired_side=desired_side,
                    threshold=threshold,
                )
                return None
            intent = "scale_in_short"
        if desired_side == "BUY" and pos.qty < 0:
            intent = "close_or_flip_short_to_long"
            close_qty = abs(float(pos.qty))
            strength = close_qty + min(max(0.0, strength - close_qty), max(0.0, flip_extra_entry))
        if desired_side == "SELL" and pos.qty > 0:
            intent = "close_or_flip_long_to_short"
            close_qty = abs(float(pos.qty))
            strength = close_qty + min(max(0.0, strength - close_qty), max(0.0, flip_extra_entry))

        # Thread-safe write to cooldown tracking dictionaries.
        with self._state_lock:
            self._last_signal_ts_by_symbol[symbol] = now
            self._last_signal_bar_by_symbol[symbol] = bar_count

        self._inc_counter(self._decision_hits, intent)

        return Signal(
            instrument=ctx.instrument,
            side=desired_side,
            strength=strength,
            confidence=confidence,
            horizon=pred.horizon,
            reason=(
                f"{self.id()}|threshold={threshold:.4f}|edge={edge:.4f}|"
                f"intent={intent}|sigma={float(pred.sigma):.6f}|h={int(pred.horizon)}|"
                f"bar={bar_count}|scale={horizon_scale_mode}"
            ),
        )

    def id(self) -> str:
        return "threshold-strategy"

    def warmup_required(self) -> int:
        return 1
