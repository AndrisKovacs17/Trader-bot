from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4


# Event registry for polymorphic deserialization
_EVENT_REGISTRY: dict[str, type["Event"]] = {}


def register_event(*aliases: str):
    """Decorator to auto-register event class with type aliases."""
    def decorator(cls: type["Event"]) -> type["Event"]:
        for alias in aliases:
            _EVENT_REGISTRY[alias] = cls
        _EVENT_REGISTRY[cls.__name__] = cls
        return cls
    return decorator


# Alap domain esemény entitás.
# Minden további esemény ebből öröklődik.
@dataclass(slots=True, frozen=True)
class Event:
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    correlation_id: str = ""
    schema_version: int = 1
    dedup_key: Optional[str] = None  # NOTE: Currently only a marker; not enforced by engine/store
    event_id: str = field(default_factory=lambda: str(uuid4()))
    ts_event: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ts_ingest: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        # Workaround for frozen dataclass: use object.__setattr__ for initialization
        if not self.event_type:
            object.__setattr__(self, "event_type", self.default_event_type())
        object.__setattr__(self, "ts_event", _ensure_utc(self.ts_event))
        object.__setattr__(self, "ts_ingest", _ensure_utc(self.ts_ingest))
        self.validate()

    @classmethod
    def default_event_type(cls) -> str:
        return getattr(cls, "EVENT_TYPE", cls.__name__)

    # Minimális validáció MVP célra.
    def validate(self) -> None:
        if not self.event_type:
            raise ValueError("event_type kötelező")
        if not isinstance(self.payload, dict):
            raise ValueError("payload csak dict lehet")

    # Egyszerű serializálás loggoláshoz/tároláshoz.
    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "ts_event": self.ts_event.isoformat(),
            "ts_ingest": self.ts_ingest.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "schema_version": self.schema_version,
            "dedup_key": self.dedup_key,
        }

    # Egyszerű deserializálás dictionary-ből.
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        raw_type = str(data.get("event_type", ""))
        event_cls = _event_cls_for_type(raw_type)
        return event_cls(
            event_id=str(data.get("event_id", str(uuid4()))),
            event_type=raw_type,
            ts_event=_parse_dt(data.get("ts_event")),
            ts_ingest=_parse_dt(data.get("ts_ingest")),
            source=str(data.get("source", "unknown")),
            correlation_id=str(data.get("correlation_id", "")),
            payload=dict(data.get("payload", {})),
            schema_version=int(data.get("schema_version", 1)),
            dedup_key=data.get("dedup_key"),
        )


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str) and value:
        try:
            return _ensure_utc(datetime.fromisoformat(value))
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# Piaci adat esemény.
@register_event("MarketData")
@dataclass(slots=True, frozen=True)
class MarketDataEvent(Event):
    EVENT_TYPE = "MarketData"


# Megrendelés/fill státusz esemény.
@register_event("Order", "OrderSubmitted", "OrderFilled", "OrderRejected", "OrderCancelled")
@dataclass(slots=True, frozen=True)
class OrderEvent(Event):
    EVENT_TYPE = "Order"


@register_event("OrderFilled")
@dataclass(slots=True, frozen=True)
class OrderFilledEvent(OrderEvent):
    EVENT_TYPE = "OrderFilled"


@register_event("OrderSubmitted")
@dataclass(slots=True, frozen=True)
class OrderSubmittedEvent(OrderEvent):
    EVENT_TYPE = "OrderSubmitted"


@register_event("OrderRejected")
@dataclass(slots=True, frozen=True)
class OrderRejectedEvent(OrderEvent):
    EVENT_TYPE = "OrderRejected"


# Signal esemény.
@register_event("Signal", "SignalGenerated")
@dataclass(slots=True, frozen=True)
class SignalEvent(Event):
    EVENT_TYPE = "Signal"


# Kockázati esemény.
@register_event("Risk", "RiskApproved", "RiskBlocked")
@dataclass(slots=True, frozen=True)
class RiskEvent(Event):
    EVENT_TYPE = "Risk"


@register_event("RiskBlocked")
@dataclass(slots=True, frozen=True)
class RiskBlockedEvent(RiskEvent):
    EVENT_TYPE = "RiskBlocked"


@register_event("RiskApproved")
@dataclass(slots=True, frozen=True)
class RiskApprovedEvent(RiskEvent):
    EVENT_TYPE = "RiskApproved"


# Modell diagnosztika esemény.
@register_event("Model", "Prediction", "ModelUpdated")
@dataclass(slots=True, frozen=True)
class ModelEvent(Event):
    EVENT_TYPE = "Model"


@register_event("ModelUpdated")
@dataclass(slots=True, frozen=True)
class ModelUpdatedEvent(ModelEvent):
    EVENT_TYPE = "ModelUpdated"


@register_event("Prediction")
@dataclass(slots=True, frozen=True)
class PredictionEvent(ModelEvent):
    EVENT_TYPE = "Prediction"


# Teljesítmény/pozíció frissítési esemény
@register_event("PerformanceUpdate")
@dataclass(slots=True, frozen=True)
class PerformanceUpdateEvent(Event):
    EVENT_TYPE = "PerformanceUpdate"


# News sentiment event
@register_event("NewsSentiment")
@dataclass(slots=True, frozen=True)
class NewsSentimentEvent(Event):
    EVENT_TYPE = "NewsSentiment"


# Execution completion event
@register_event("ExecutionComplete")
@dataclass(slots=True, frozen=True)
class ExecutionCompletedEvent(Event):
    EVENT_TYPE = "ExecutionComplete"


# Modell életciklus esemény (dedikált tick)
@register_event("ModelLifecycleApplied")
@dataclass(slots=True, frozen=True)
class ModelLifecycleAppliedEvent(ModelEvent):
    EVENT_TYPE = "ModelLifecycleApplied"


# Engine error event
@register_event("EngineError")
@dataclass(slots=True, frozen=True)
class EngineErrorEvent(Event):
    EVENT_TYPE = "EngineError"


def _event_cls_for_type(event_type: str) -> type[Event]:
    if not event_type:
        return Event
    return _EVENT_REGISTRY.get(event_type, Event)
