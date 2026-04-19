from __future__ import annotations

from datetime import datetime, timedelta

from core.application.ports import ITimeSource


class SimulatedTimeSource(ITimeSource):
    def __init__(self, current_time: datetime) -> None:
        self.current_time = current_time

    def now(self) -> datetime:
        return self.current_time

    def advance(self, delta: timedelta) -> None:
        self.current_time = self.current_time + delta

    def set(self, value: datetime) -> None:
        self.current_time = value
