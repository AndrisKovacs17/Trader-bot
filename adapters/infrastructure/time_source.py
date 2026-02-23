from __future__ import annotations

from datetime import datetime, timezone


class SystemTimeSource:
    """Infrastructure adapter for system UTC time."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)
