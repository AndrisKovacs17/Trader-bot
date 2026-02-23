from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from threading import Lock
from typing import Optional, Protocol, TYPE_CHECKING

from core.domain.models import Signal
from core.ml.services import Prediction

if TYPE_CHECKING:
    from core.application.stores import RunContext


# Strategy interface: predikcióból signal készítése.
class IStrategy(Protocol):
    def on_prediction(self, pred: Prediction, ctx: RunContext) -> Optional[Signal]:
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
    _last_signal_ts_by_symbol: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)
    _last_signal_bar_by_symbol: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _state_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

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

    def on_prediction(self, pred: Prediction, ctx: RunContext) -> Optional[Signal]:
        # Determinisztikus replay megjegyzés:
        # ugyanazt a strategy.* és model.signal_threshold configot kell használni replay során.
        threshold = float(
            ctx.config.get(
                "strategy.threshold",
                ctx.config.get("model.signal_threshold", self.threshold),
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
        allowed_horizons = ctx.config.get("strategy.allowed_horizons", [])

        bar_count = self._bar_count(ctx)
        if bar_count < self.warmup_required():
            logging.debug(f"[STRATEGY] BLOCKED: warmup required {self.warmup_required()}, have {bar_count}")
            return None

        prob_up = self._clamp01(float(pred.prob_up))
        confidence = max(prob_up, 1.0 - prob_up)
        edge = abs(prob_up - 0.5)
        
        # Signal quality scoring:
        # use_edge_score=False (default): |mu| * confidence (volatility-neutral quality)
        # use_edge_score=True: edge * |mu| (directional conviction * magnitude)
        use_edge_score = bool(ctx.config.get("strategy.use_edge_score", self.use_edge_score))
        if use_edge_score:
            signal_score = edge * abs(float(pred.mu))
        else:
            signal_score = abs(float(pred.mu)) * confidence

        logging.debug(f"[STRATEGY] prob_up={prob_up:.6f} | conf={confidence:.6f} | edge={edge:.6f} | mu={float(pred.mu):.6f} | sigma={float(pred.sigma):.6f} | score={signal_score:.6f}")
        logging.debug(f"[STRATEGY] Thresholds: min_conf={min_confidence:.6f} | min_edge={min_edge:.6f} | min_score={min_signal_score:.6f} | max_sigma={max_sigma:.6f}")

        if confidence < min_confidence:
            logging.debug(f"[STRATEGY] BLOCKED: confidence {confidence:.6f} < min_confidence {min_confidence:.6f}")
            return None
        if edge < min_edge:
            logging.debug(f"[STRATEGY] BLOCKED: edge {edge:.6f} < min_edge {min_edge:.6f}")
            return None
        if signal_score < min_signal_score:
            logging.debug(f"[STRATEGY] BLOCKED: signal_score {signal_score:.6f} < min_signal_score {min_signal_score:.6f}")
            return None
        if float(pred.sigma) > max_sigma:
            logging.debug(f"[STRATEGY] BLOCKED: sigma {float(pred.sigma):.6f} > max_sigma {max_sigma:.6f}")
            return None

        if isinstance(allowed_horizons, (list, tuple, set)) and len(allowed_horizons) > 0:
            if int(pred.horizon) not in {int(x) for x in allowed_horizons}:
                logging.debug(f"[STRATEGY] BLOCKED: horizon {int(pred.horizon)} not in allowed {allowed_horizons}")
                return None

        now = ctx.now()
        symbol = ctx.instrument.symbol
        if self._in_cooldown(symbol, now, bar_count, min_cooldown_seconds, min_bars_between_signals):
            logging.debug(f"[STRATEGY] BLOCKED: cooldown active for {symbol}")
            return None

        desired_side = "BUY" if prob_up >= threshold else "SELL" if prob_up <= (1.0 - threshold) else ""
        if not desired_side:
            logging.debug(f"[STRATEGY] BLOCKED: prob_up {prob_up:.6f} not >= threshold {threshold:.6f} and not <= {1.0-threshold:.6f}")
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

        pos = ctx.position_store.get(ctx.instrument)
        intent = "enter"
        if desired_side == "BUY" and pos.qty > 0:
            return None
        if desired_side == "SELL" and pos.qty < 0:
            return None
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
