from __future__ import annotations

from datetime import datetime, timezone

from core.application.ports import ITimeSource


class SystemTimeSource(ITimeSource):
    """Infrastructure adapter for system UTC time."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)
