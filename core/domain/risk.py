from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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


# Chain-of-responsibility szabály interface.
class IRiskRule(Protocol):
    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        ...

    def name(self) -> str:
        ...

    def severity(self) -> RiskSeverity:
        ...

    def priority(self) -> int:
        ...


# Egyszerű max mennyiség szabály.
@dataclass(slots=True)
class MaxQtyRule(IRiskRule):
    max_qty: float = 1.0
    min_qty: float = 0.0001

    def check(self, signal: Signal, ctx: RunContext) -> RiskResult:
        if signal.strength <= 0:
            return RiskResult(
                allowed=False,
                reason="strength<=0",
                reason_code=RiskReasonCode.STRENGTH_NON_POSITIVE.value,
                severity=self.severity(),
            )
        qty = min(self.max_qty, max(self.min_qty, signal.strength))
        if qty <= 0:
            return RiskResult(
                allowed=False,
                reason="qty<=0",
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


# Egyszerű confidence küszöb szabály.
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


# Kockázati policy: végigfuttatja a szabályláncot.
@dataclass(slots=True)
class RiskPolicy:
    rules: list[IRiskRule]
    dedup_window: int = 512
    _seen_dedup: set[str] = field(default_factory=set, init=False, repr=False)
    _dedup_order: deque[str] = field(default_factory=deque, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _rule_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _block_reason_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)

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
        base = (
            f"{ctx.correlation_id}|{signal.instrument.symbol}|{signal.side}|{qty:.8f}|"
            f"{signal.horizon}|{signal.reason}|{model_version}"
        )
        return base

    def evaluate(self, signal: Signal, ctx: RunContext) -> RiskDecision:
        qty_override = None
        price_override = None
        for rule in self.rules:
            self._inc_rule_hit(rule.name())
            result = rule.check(signal, ctx)
            if not result.allowed:
                reason_code = result.reason_code or "UNKNOWN"
                self._inc_block_reason(reason_code)
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
                qty_override = result.adjusted_qty
            if result.adjusted_price is not None:
                price_override = result.adjusted_price

        fallback_qty = max(float(signal.strength), 0.0)
        order_qty = qty_override if qty_override is not None else fallback_qty
        min_qty = float(ctx.config.get("risk_limits.min_qty", 0.0001))
        if order_qty < min_qty:
            self._inc_block_reason(RiskReasonCode.QTY_NON_POSITIVE.value)
            event = RiskBlockedEvent(
                payload={
                    "reason": "adjusted-qty-below-min-qty",
                    "reason_code": RiskReasonCode.QTY_NON_POSITIVE.value,
                    "severity": RiskSeverity.HIGH.value,
                    "rule": "post-adjustment-min-qty",
                    "risk_counters": self.diagnostics(),
                },
                source="risk",
                dedup_key=f"risk-blocked:{ctx.correlation_id}:{signal.instrument.symbol}:{signal.side}:{RiskReasonCode.QTY_NON_POSITIVE.value}",
            )
            return RiskDecision(status=RiskStatus.BLOCKED, order=None, events_to_emit=[event])

        dedup_key = self._build_dedup_key(signal, order_qty, ctx)
        if self._already_seen(dedup_key):
            self._inc_block_reason(RiskReasonCode.DUPLICATE_SIGNAL.value)
            event = RiskBlockedEvent(
                payload={
                    "reason": "duplicate-signal",
                    "reason_code": RiskReasonCode.DUPLICATE_SIGNAL.value,
                    "severity": RiskSeverity.LOW.value,
                    "rule": "dedup",
                    "risk_counters": self.diagnostics(),
                },
                source="risk",
                dedup_key=f"risk-dup:{dedup_key}",
            )
            return RiskDecision(status=RiskStatus.BLOCKED, order=None, events_to_emit=[event])
        self._remember(dedup_key)

        order = Order(
            instrument=signal.instrument,
            side=signal.side,
            qty=order_qty,
            limit_price=price_override,
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
        return RiskDecision(status=RiskStatus.APPROVED, order=order, events_to_emit=[event])
