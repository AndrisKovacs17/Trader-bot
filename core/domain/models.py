from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from core.domain.events import Event


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# Instrument value object: kereskedett termek alapadatai.
@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    venue: str = "SIM"
    tick_size: float = 0.01
    lot_size: float = 1.0


# Strategy output value object.
@dataclass(slots=True, frozen=True)
class Signal:
    instrument: Instrument
    side: str
    strength: float
    confidence: float
    horizon: int
    reason: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", str(self.side).upper())
        object.__setattr__(self, "ts", _ensure_utc(self.ts))


class OrderState(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


# Order entitas.
@dataclass(slots=True)
class Order:
    instrument: Instrument
    side: str
    qty: float
    order_type: str = "MARKET"
    limit_price: float | None = None
    time_in_force: str = "IOC"
    client_tag: str = "mvp"
    state: OrderState = OrderState.NEW
    filled_qty: float = 0.0
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        self.side = str(self.side).upper()
        self.order_type = str(self.order_type).upper()
        self.time_in_force = str(self.time_in_force).upper()
        self.ts = _ensure_utc(self.ts)
        self.qty = float(self.qty)
        self.filled_qty = float(self.filled_qty)
        if self.limit_price is not None:
            self.limit_price = float(self.limit_price)
        if self.qty <= 0:
            raise ValueError("order qty must be > 0")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("order side must be BUY or SELL")
        if self.filled_qty < 0:
            raise ValueError("filled_qty must be >= 0")
        if self.filled_qty > self.qty:
            raise ValueError("filled_qty must be <= qty")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be > 0 when provided")
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("limit orders require a positive limit_price")

    def apply_fill(self, fill: Fill) -> None:
        if self.state == OrderState.CANCELED:
            return
        if fill.order_id != self.order_id:
            return
        self.filled_qty = min(self.qty, self.filled_qty + float(fill.qty))
        if self.filled_qty >= self.qty:
            self.state = OrderState.FILLED
        elif self.filled_qty > 0:
            self.state = OrderState.PARTIALLY_FILLED

    def cancel(self) -> None:
        if self.state != OrderState.FILLED:
            self.state = OrderState.CANCELED

    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    def is_active(self) -> bool:
        return self.state in {OrderState.NEW, OrderState.PARTIALLY_FILLED}


# Fill entitas.
@dataclass(slots=True)
class Fill:
    order_id: str
    qty: float
    price: float
    fee: float
    side: str = "BUY"
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.qty = float(self.qty)
        self.price = float(self.price)
        self.fee = float(self.fee)
        self.side = str(self.side).upper()
        self.ts = _ensure_utc(self.ts)
        if self.qty <= 0:
            raise ValueError("fill qty must be > 0")
        if self.price <= 0:
            raise ValueError("fill price must be > 0")
        if self.fee < 0:
            raise ValueError("fill fee must be >= 0")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("fill side must be BUY or SELL")


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Risk checker eredmenye.
@dataclass(slots=True)
class RiskResult:
    allowed: bool
    reason: str = ""
    reason_code: str = ""
    severity: RiskSeverity | None = None
    adjusted_qty: float | None = None
    adjusted_price: float | None = None


class RiskReasonCode(str, Enum):
    STRENGTH_NON_POSITIVE = "STRENGTH_NON_POSITIVE"
    QTY_NON_POSITIVE = "QTY_NON_POSITIVE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MAX_NOTIONAL_EXCEEDED = "MAX_NOTIONAL_EXCEEDED"
    MAX_ABS_POSITION_EXCEEDED = "MAX_ABS_POSITION_EXCEEDED"
    PRICE_REFERENCE_MISSING = "PRICE_REFERENCE_MISSING"
    SLIPPAGE_BPS_EXCEEDED = "SLIPPAGE_BPS_EXCEEDED"
    DUPLICATE_SIGNAL = "DUPLICATE_SIGNAL"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    SHORT_SELL_NOT_ALLOWED = "SHORT_SELL_NOT_ALLOWED"
    REENTRY_EDGE_NOT_IMPROVED = "REENTRY_EDGE_NOT_IMPROVED"
    SCALE_IN_LIMIT_EXCEEDED = "SCALE_IN_LIMIT_EXCEEDED"
    WEAK_REENTRY_SIGNAL = "WEAK_REENTRY_SIGNAL"
    SHADOW_MODE_ACTIVE = "SHADOW_MODE_ACTIVE"
    NEWS_SENTIMENT_GATE = "NEWS_SENTIMENT_GATE"


class RiskStatus(str, Enum):
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"


# Risk policy vegso dontese.
@dataclass(slots=True)
class RiskDecision:
    status: RiskStatus
    order: Order | None
    events_to_emit: list[Event]


# Pozicio entitas.
@dataclass(slots=True)
class Position:
    qty: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def apply_fill(self, fill: Fill) -> None:
        prev_qty = self.qty
        prev_avg = self.avg_price

        if fill.side == "BUY":
            if prev_qty >= 0:
                new_qty = prev_qty + fill.qty
                self.avg_price = ((prev_avg * prev_qty) + (fill.price * fill.qty)) / new_qty if new_qty > 0 else 0.0
                self.qty = new_qty
                return

            cover_qty = min(abs(prev_qty), fill.qty)
            self.realized_pnl += (prev_avg - fill.price) * cover_qty
            new_qty = prev_qty + fill.qty
            self.qty = new_qty
            if new_qty > 0:
                self.avg_price = fill.price
            elif new_qty == 0:
                self.avg_price = 0.0
            return

        if prev_qty <= 0:
            new_abs = abs(prev_qty) + fill.qty
            self.avg_price = ((prev_avg * abs(prev_qty)) + (fill.price * fill.qty)) / new_abs if new_abs > 0 else 0.0
            self.qty = prev_qty - fill.qty
            return

        close_qty = min(prev_qty, fill.qty)
        self.realized_pnl += (fill.price - prev_avg) * close_qty
        new_qty = prev_qty - fill.qty
        self.qty = new_qty
        if new_qty < 0:
            self.avg_price = fill.price
        elif new_qty == 0:
            self.avg_price = 0.0

    def mark_to_market(self, price: float) -> None:
        self.unrealized_pnl = (float(price) - self.avg_price) * self.qty

    def reset(self) -> None:
        self.qty = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
