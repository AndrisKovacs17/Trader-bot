from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import logging
from threading import Lock
from typing import Protocol, TYPE_CHECKING

from core.domain.events import RiskApprovedEvent, RiskBlockedEvent
from core.domain.models import (
    Order,
    RiskDecision,
    RiskReasonCode,
    RiskResult,
    RiskSeverity,
    RiskStatus,
    Signal,
)

if TYPE_CHECKING:
    from core.application.stores import RunContext


logger = logging.getLogger(__name__)


# Chain-of-responsibility szabaly interface.
class IRiskRule(Protocol):
    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        ...

    def name(self) -> str:
        ...

    def severity(self) -> RiskSeverity:
        ...

    def priority(self) -> int:
        ...


# Egyszeru max mennyiseg szabaly.
@dataclass(slots=True)
class MaxQtyRule(IRiskRule):
    max_qty: float = 1.0
    min_qty: float = 0.0001

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        raw_qty = float(signal.strength)
        if raw_qty <= 0:
            return RiskResult(
                allowed=False,
                reason="strength<=0",
                reason_code=RiskReasonCode.STRENGTH_NON_POSITIVE.value,
                severity=self.severity(),
            )
        qty = min(self.max_qty, raw_qty)
        if qty < self.min_qty:
            return RiskResult(
                allowed=False,
                reason="qty-below-min-qty",
                reason_code=RiskReasonCode.QTY_NON_POSITIVE.value,
                severity=self.severity(),
            )
        return RiskResult(allowed=True, adjusted_qty=qty)

    def name(self) -> str:
        return "max-qty"

    def severity(self) -> RiskSeverity:
        return RiskSeverity.HIGH

    def priority(self) -> int:
        return 20


# Egyszeru confidence kuszob szabaly.
@dataclass(slots=True)
class ConfidenceRule(IRiskRule):
    min_confidence: float = 0.50

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        if signal.confidence < self.min_confidence:
            return RiskResult(
                allowed=False,
                reason="low-confidence",
                reason_code=RiskReasonCode.LOW_CONFIDENCE.value,
                severity=self.severity(),
            )
        return RiskResult(allowed=True)

    def name(self) -> str:
        return "confidence"

    def severity(self) -> RiskSeverity:
        return RiskSeverity.MEDIUM

    def priority(self) -> int:
        return 10


@dataclass(slots=True)
class MaxNotionalRule(IRiskRule):
    max_notional: float = 1000.0

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        price = float(ctx.wallet.last_prices.get(signal.instrument.symbol, 0.0))
        if price <= 0:
            return RiskResult(
                allowed=False,
                reason="price-reference-missing",
                reason_code=RiskReasonCode.PRICE_REFERENCE_MISSING.value,
                severity=self.severity(),
            )
        target_qty = max(float(signal.strength), 0.0)
        notional = target_qty * price
        configured_limit = float(ctx.config.get("risk_limits.max_notional", self.max_notional))
        if notional > configured_limit:
            adjusted_qty = configured_limit / price
            if adjusted_qty <= 0:
                return RiskResult(
                    allowed=False,
                    reason="max-notional-exceeded",
                    reason_code=RiskReasonCode.MAX_NOTIONAL_EXCEEDED.value,
                    severity=self.severity(),
                )
            return RiskResult(allowed=True, adjusted_qty=adjusted_qty)
        return RiskResult(allowed=True)

    def name(self) -> str:
        return "max-notional"

    def severity(self) -> RiskSeverity:
        return RiskSeverity.HIGH

    def priority(self) -> int:
        return 30


@dataclass(slots=True)
class MaxAbsPositionRule(IRiskRule):
    max_abs_position_qty: float = 2.0

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        position = ctx.position_store.get(signal.instrument)
        current_qty = float(position.qty)
        target_qty = max(float(signal.strength), 0.0)
        signed_target = target_qty if signal.side == "BUY" else -target_qty
        projected = current_qty + signed_target
        configured_limit = float(ctx.config.get("risk_limits.max_abs_position_qty", self.max_abs_position_qty))
        if abs(projected) <= configured_limit:
            return RiskResult(allowed=True)

        if signal.side == "BUY":
            allowed_delta = configured_limit - current_qty
        else:
            allowed_delta = configured_limit + current_qty
        adjusted_qty = max(0.0, allowed_delta)
        if adjusted_qty <= 0:
            return RiskResult(
                allowed=False,
                reason="max-abs-position-exceeded",
                reason_code=RiskReasonCode.MAX_ABS_POSITION_EXCEEDED.value,
                severity=self.severity(),
            )
        return RiskResult(allowed=True, adjusted_qty=adjusted_qty)

    def name(self) -> str:
        return "max-abs-position"

    def severity(self) -> RiskSeverity:
        return RiskSeverity.CRITICAL

    def priority(self) -> int:
        return 40


@dataclass(slots=True)
class SlippageGuardRule(IRiskRule):
    max_slippage_bps: float = 50.0

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        price = float(ctx.wallet.last_prices.get(signal.instrument.symbol, 0.0))
        if price <= 0:
            return RiskResult(
                allowed=False,
                reason="price-reference-missing",
                reason_code=RiskReasonCode.PRICE_REFERENCE_MISSING.value,
                severity=self.severity(),
            )
        strength_mode = str(ctx.config.get("risk_limits.strength_mode", "qty_proxy"))
        if strength_mode == "alpha_score":
            # strength interpreted as normalized alpha score in [0,1]
            alpha_scale_bps = float(ctx.config.get("risk_limits.alpha_scale_bps", 100.0))
            implied_bps = abs(float(signal.strength)) * alpha_scale_bps
        else:
            # default: strength interpreted as qty proxy
            expected_move = abs(float(signal.strength))
            implied_bps = (expected_move / price) * 10_000.0 if price > 0 else 0.0
        configured_limit = float(ctx.config.get("risk_limits.max_slippage_bps", self.max_slippage_bps))
        if implied_bps > configured_limit:
            return RiskResult(
                allowed=False,
                reason="slippage-bps-exceeded",
                reason_code=RiskReasonCode.SLIPPAGE_BPS_EXCEEDED.value,
                severity=self.severity(),
            )
        return RiskResult(allowed=True, adjusted_price=price)

    def name(self) -> str:
        return "slippage-guard"

    def severity(self) -> RiskSeverity:
        return RiskSeverity.MEDIUM

    def priority(self) -> int:
        return 25


# Protocol for news sentiment providers (satisfied by NewsStateService in the adapter layer).
# Defined here so core domain has no import dependency on adapters.
class ISentimentProvider(Protocol):
    def current_score(self) -> float:
        """Return current sentiment in [-1, 1]. 0.0 when neutral or stale."""
        ...

    def diagnostics(self) -> dict:
        ...


class NewsSentimentGateRule:
    """Block entry signals that strongly oppose current news sentiment.

    Only activates when |sentiment| >= strong_threshold (default 0.45).
    Neutral/mildly-bullish/bearish news is ignored; the gate is intentionally
    conservative to avoid blocking on noise.

    Exit signals (closing/reducing an existing position) are always allowed.
    """

    def __init__(self, sentiment_provider: ISentimentProvider, strong_threshold: float = 0.45) -> None:
        self._provider = sentiment_provider
        self._strong_threshold = max(0.01, float(strong_threshold))

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        score = self._provider.current_score()
        # Fast path: neutral news → don't interfere
        if abs(score) < self._strong_threshold:
            return RiskResult(allowed=True)

        # Determine if this is an entry (increases directional exposure) or exit
        position = ctx.position_store.get(signal.instrument)
        current_qty = float(position.qty)
        is_entry_long = signal.side == "BUY" and current_qty >= 0.0
        is_entry_short = signal.side == "SELL" and current_qty <= 0.0

        if is_entry_long and score <= -self._strong_threshold:
            return RiskResult(
                allowed=False,
                reason=f"news-strongly-bearish({score:.3f})-blocks-long-entry",
                reason_code=RiskReasonCode.NEWS_SENTIMENT_GATE.value,
                severity=self.severity(),
            )
        if is_entry_short and score >= self._strong_threshold:
            return RiskResult(
                allowed=False,
                reason=f"news-strongly-bullish({score:.3f})-blocks-short-entry",
                reason_code=RiskReasonCode.NEWS_SENTIMENT_GATE.value,
                severity=self.severity(),
            )
        return RiskResult(allowed=True)

    def name(self) -> str:
        return "news-sentiment-gate"

    def severity(self) -> RiskSeverity:
        return RiskSeverity.MEDIUM

    def priority(self) -> int:
        return 12  # runs after ConfidenceRule(10), before MaxQtyRule(20)


# Kockazati policy: vegigfuttatja a szabaly lancot.
@dataclass(slots=True)
class RiskPolicy:
    rules: list[IRiskRule]
    dedup_window: int = 512
    _seen_dedup: set[str] = field(default_factory=set, init=False, repr=False)
    _dedup_order: deque[str] = field(default_factory=deque, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _rule_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _block_reason_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _last_approved_edge_by_side: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _same_side_reentry_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rules = sorted(self.rules, key=lambda rule: rule.priority())

    def _remember(self, dedup_key: str) -> None:
        with self._lock:
            self._seen_dedup.add(dedup_key)
            self._dedup_order.append(dedup_key)
            while len(self._dedup_order) > self.dedup_window:
                expired = self._dedup_order.popleft()
                self._seen_dedup.discard(expired)

    def _already_seen(self, dedup_key: str) -> bool:
        with self._lock:
            return dedup_key in self._seen_dedup

    def _inc_rule_hit(self, rule_name: str) -> None:
        with self._lock:
            self._rule_hits[rule_name] = self._rule_hits.get(rule_name, 0) + 1

    def _inc_block_reason(self, reason_code: str) -> None:
        with self._lock:
            self._block_reason_hits[reason_code] = self._block_reason_hits.get(reason_code, 0) + 1

    def diagnostics(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                "rule_hits": dict(self._rule_hits),
                "block_reason_hits": dict(self._block_reason_hits),
            }

    @staticmethod
    def _model_version(ctx: RunContext) -> str:
        return str(
            ctx.config.get("model.version", ctx.config.get("model.active_version", "unknown"))
        )

    @staticmethod
    def _build_dedup_key(signal: Signal, qty: float, ctx: RunContext) -> str:
        model_version = RiskPolicy._model_version(ctx)
        return (
            f"{signal.instrument.symbol}|{signal.side}|{qty:.8f}|"
            f"{signal.horizon}|{signal.reason}|{model_version}"
        )

    @staticmethod
    def _extract_edge(signal: Signal) -> float | None:
        try:
            chunks = str(signal.reason or "").split("|")
            for chunk in chunks:
                if chunk.startswith("edge="):
                    return float(chunk.split("=", 1)[1])
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _side_key(symbol: str, side: str) -> str:
        return f"{symbol}:{side.upper()}"

    @staticmethod
    def _working_signal(signal: Signal, qty_override: float | None) -> Signal:
        if qty_override is None:
            return signal
        return replace(signal, strength=float(qty_override))

    @staticmethod
    def _reference_price(signal: Signal, ctx: RunContext, price_override: float | None) -> float:
        if price_override is not None:
            return float(price_override)
        return float(ctx.wallet.last_prices.get(signal.instrument.symbol, 0.0))

    @staticmethod
    def _config_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _blocked_decision(
        self,
        *,
        signal: Signal,
        ctx: RunContext,
        reason: str,
        reason_code: str,
        severity: RiskSeverity,
        rule_name: str,
    ) -> RiskDecision:
        self._inc_block_reason(reason_code)
        event = RiskBlockedEvent(
            payload={
                "reason": reason,
                "reason_code": reason_code,
                "severity": severity.value,
                "rule": rule_name,
                "risk_counters": self.diagnostics(),
            },
            source="risk",
            dedup_key=f"risk-blocked:{ctx.correlation_id}:{signal.instrument.symbol}:{signal.side}:{reason_code}",
        )
        logger.debug(
            "[RISK] BLOCKED: reason=%s code=%s rule=%s side=%s symbol=%s",
            reason,
            reason_code,
            rule_name,
            signal.side,
            signal.instrument.symbol,
        )
        return RiskDecision(status=RiskStatus.BLOCKED, order=None, events_to_emit=[event])

    def evaluate(self, signal: Signal, ctx: RunContext) -> RiskDecision:
        shadow_required = int(ctx.config.get("risk_limits.shadow_predictions_required", 0))
        current_bar = int(getattr(ctx, "bar_index", 0))
        if shadow_required > 0 and current_bar < shadow_required:
            return self._blocked_decision(
                signal=signal,
                ctx=ctx,
                reason=f"shadow-mode-active:{current_bar}<{shadow_required}",
                reason_code=RiskReasonCode.SHADOW_MODE_ACTIVE.value,
                severity=RiskSeverity.LOW,
                rule_name="shadow-warmup",
            )

        qty_override: float | None = None
        price_override: float | None = None
        for rule in self.rules:
            working_signal = self._working_signal(signal, qty_override)
            self._inc_rule_hit(rule.name())
            result = rule.check(working_signal, ctx)
            if not result.allowed:
                reason_code = result.reason_code or "UNKNOWN"
                self._inc_block_reason(reason_code)
                logger.debug(
                    "[RISK] BLOCKED: reason=%s code=%s rule=%s side=%s symbol=%s",
                    result.reason,
                    reason_code,
                    rule.name(),
                    signal.side,
                    signal.instrument.symbol,
                )
                event = RiskBlockedEvent(
                    payload={
                        "reason": result.reason,
                        "reason_code": reason_code,
                        "severity": (result.severity.value if result.severity else rule.severity().value),
                        "rule": rule.name(),
                        "risk_counters": self.diagnostics(),
                    },
                    source="risk",
                    dedup_key=f"risk-blocked:{ctx.correlation_id}:{signal.instrument.symbol}:{signal.side}:{reason_code}",
                )
                return RiskDecision(status=RiskStatus.BLOCKED, order=None, events_to_emit=[event])
            if result.adjusted_qty is not None:
                qty_override = float(result.adjusted_qty)
            if result.adjusted_price is not None:
                price_override = float(result.adjusted_price)

        raw_strength = max(float(signal.strength), 0.0)
        if qty_override is not None:
            order_qty = float(qty_override)
        else:
            use_score_sizing = self._config_bool(ctx.config.get("risk_limits.use_score_sizing", False))
            if use_score_sizing:
                score_to_qty_scale = max(float(ctx.config.get("risk_limits.score_to_qty_scale", 0.0)), 0.0)
                min_score_for_size = max(float(ctx.config.get("risk_limits.min_score_for_size", 0.0)), 0.0)
                effective_score = max(0.0, raw_strength - min_score_for_size)
                order_qty = effective_score * score_to_qty_scale
            else:
                order_qty = raw_strength
        min_qty = float(ctx.config.get("risk_limits.min_qty", 0.0001))
        if order_qty < min_qty:
            return self._blocked_decision(
                signal=signal,
                ctx=ctx,
                reason="adjusted-qty-below-min-qty",
                reason_code=RiskReasonCode.QTY_NON_POSITIVE.value,
                severity=RiskSeverity.HIGH,
                rule_name="post-adjustment-min-qty",
            )

        reference_price = self._reference_price(signal, ctx, price_override)
        if reference_price <= 0:
            return self._blocked_decision(
                signal=signal,
                ctx=ctx,
                reason="price-reference-missing",
                reason_code=RiskReasonCode.PRICE_REFERENCE_MISSING.value,
                severity=RiskSeverity.HIGH,
                rule_name="portfolio-constraints",
            )

        allow_short = self._config_bool(ctx.config.get("risk_limits.allow_short_selling", False))
        current_qty = float(ctx.position_store.get(signal.instrument).qty)
        if signal.side == "SELL" and not allow_short:
            max_sell_qty = max(0.0, current_qty)
            if max_sell_qty < min_qty:
                return self._blocked_decision(
                    signal=signal,
                    ctx=ctx,
                    reason="short-selling-disabled",
                    reason_code=RiskReasonCode.SHORT_SELL_NOT_ALLOWED.value,
                    severity=RiskSeverity.CRITICAL,
                    rule_name="portfolio-constraints",
                )
            order_qty = min(order_qty, max_sell_qty)

        estimated_fee_bps = max(float(ctx.config.get("risk_limits.estimated_fee_bps", 10.0)), 0.0)
        fee_multiplier = 1.0 + (estimated_fee_bps / 10_000.0)
        if signal.side == "BUY":
            max_affordable_qty = max(0.0, float(ctx.wallet.cash)) / (reference_price * fee_multiplier)
            if order_qty > max_affordable_qty:
                if max_affordable_qty < min_qty:
                    return self._blocked_decision(
                        signal=signal,
                        ctx=ctx,
                        reason="insufficient-cash",
                        reason_code=RiskReasonCode.INSUFFICIENT_CASH.value,
                        severity=RiskSeverity.HIGH,
                        rule_name="portfolio-constraints",
                    )
                order_qty = max_affordable_qty

        if order_qty < min_qty:
            reason_code = (
                RiskReasonCode.SHORT_SELL_NOT_ALLOWED.value if signal.side == "SELL" and not allow_short
                else RiskReasonCode.INSUFFICIENT_CASH.value if signal.side == "BUY"
                else RiskReasonCode.QTY_NON_POSITIVE.value
            )
            reason = (
                "sell-qty-below-min-after-position-clamp" if signal.side == "SELL" and not allow_short
                else "buy-qty-below-min-after-cash-clamp" if signal.side == "BUY"
                else "adjusted-qty-below-min-qty"
            )
            severity = RiskSeverity.HIGH if signal.side == "BUY" else RiskSeverity.CRITICAL
            return self._blocked_decision(
                signal=signal,
                ctx=ctx,
                reason=reason,
                reason_code=reason_code,
                severity=severity,
                rule_name="portfolio-constraints",
            )

        same_side_reentry = (signal.side == "BUY" and current_qty > 0) or (signal.side == "SELL" and current_qty < 0)
        edge_value = self._extract_edge(signal)
        side_key = self._side_key(signal.instrument.symbol, signal.side)
        opposite_side = "SELL" if signal.side == "BUY" else "BUY"
        opposite_key = self._side_key(signal.instrument.symbol, opposite_side)

        min_edge_improvement = max(float(ctx.config.get("risk_limits.same_side_min_edge_improvement", 0.0)), 0.0)
        max_same_side_scale_in = int(ctx.config.get("risk_limits.max_same_side_scale_in", -1))
        weak_reentry_edge_threshold = max(float(ctx.config.get("risk_limits.weak_reentry_edge_threshold", 0.0)), 0.0)

        if same_side_reentry:
            with self._lock:
                prev_edge = self._last_approved_edge_by_side.get(side_key)
                reentry_count = self._same_side_reentry_count.get(side_key, 0)

            if weak_reentry_edge_threshold > 0 and edge_value is not None and edge_value < weak_reentry_edge_threshold:
                return self._blocked_decision(
                    signal=signal,
                    ctx=ctx,
                    reason="weak-same-side-reentry",
                    reason_code=RiskReasonCode.WEAK_REENTRY_SIGNAL.value,
                    severity=RiskSeverity.MEDIUM,
                    rule_name="same-side-reentry",
                )

            if (
                edge_value is not None
                and prev_edge is not None
                and (edge_value - prev_edge) < min_edge_improvement
            ):
                return self._blocked_decision(
                    signal=signal,
                    ctx=ctx,
                    reason="same-side-edge-not-improved",
                    reason_code=RiskReasonCode.REENTRY_EDGE_NOT_IMPROVED.value,
                    severity=RiskSeverity.MEDIUM,
                    rule_name="same-side-reentry",
                )

            if max_same_side_scale_in >= 0 and reentry_count >= max_same_side_scale_in:
                return self._blocked_decision(
                    signal=signal,
                    ctx=ctx,
                    reason="same-side-scale-in-limit",
                    reason_code=RiskReasonCode.SCALE_IN_LIMIT_EXCEEDED.value,
                    severity=RiskSeverity.HIGH,
                    rule_name="same-side-reentry",
                )

        dedup_key = self._build_dedup_key(signal, order_qty, ctx)
        if self._already_seen(dedup_key):
            return self._blocked_decision(
                signal=signal,
                ctx=ctx,
                reason="duplicate-signal",
                reason_code=RiskReasonCode.DUPLICATE_SIGNAL.value,
                severity=RiskSeverity.LOW,
                rule_name="dedup",
            )
        self._remember(dedup_key)

        order = Order(
            instrument=signal.instrument,
            side=signal.side,
            qty=order_qty,
            limit_price=reference_price,
            client_tag="risk-approved",
        )
        event = RiskApprovedEvent(
            payload={
                "side": signal.side,
                "qty": order.qty,
                "limit_price": order.limit_price,
                "dedup_key": dedup_key,
                "risk_counters": self.diagnostics(),
            },
            source="risk",
            dedup_key=f"risk-approved:{dedup_key}",
        )
        logger.debug(
            "[RISK] APPROVED: side=%s qty=%.8f price=%.4f symbol=%s",
            signal.side,
            float(order.qty),
            float(order.limit_price or 0.0),
            signal.instrument.symbol,
        )

        with self._lock:
            if same_side_reentry:
                self._same_side_reentry_count[side_key] = self._same_side_reentry_count.get(side_key, 0) + 1
            else:
                self._same_side_reentry_count[side_key] = 0
                self._same_side_reentry_count[opposite_key] = 0

            if edge_value is not None:
                self._last_approved_edge_by_side[side_key] = edge_value
            elif not same_side_reentry:
                self._last_approved_edge_by_side.pop(side_key, None)

            if not same_side_reentry:
                self._last_approved_edge_by_side.pop(opposite_key, None)

        return RiskDecision(status=RiskStatus.APPROVED, order=order, events_to_emit=[event])
