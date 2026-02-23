from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# Egészség státusz, amit portok adnak vissza.
@dataclass(slots=True)
class HealthStatus:
    ok: bool
    details: dict = field(default_factory=dict)


# Metrics port (hexagonális port, adapter valósítja meg).
class IMetrics(Protocol):
    def counter(self, name: str, labels: dict) -> object:
        ...

    def gauge(self, name: str, labels: dict) -> object:
        ...

    def histogram(self, name: str, labels: dict) -> object:
        ...


# Tracing port.
class ITracer(Protocol):
    def start_span(self, name: str, correlation_id: str) -> object:
        ...

    def annotate(self, span: object, key: str, value: object) -> None:
        ...


# Egyszerű in-memory metrics adapter-szerű implementáció MVP demóhoz.
class SimpleMetrics:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def counter(self, name: str, labels: dict) -> object:
        key = f"counter:{name}:{labels}"
        self.values[key] = self.values.get(key, 0.0) + 1.0
        return self.values[key]

    def gauge(self, name: str, labels: dict) -> object:
        key = f"gauge:{name}:{labels}"
        self.values.setdefault(key, 0.0)
        return self.values[key]

    def histogram(self, name: str, labels: dict) -> object:
        key = f"hist:{name}:{labels}"
        self.values.setdefault(key, 0.0)
        return self.values[key]
