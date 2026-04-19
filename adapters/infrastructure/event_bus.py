from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
import logging
import time
from typing import Optional

from core.application.ports import IEventBusPort, IEventHandlerPort, IEventStorePort
from core.domain.events import Event
from core.domain.models import Instrument


logger = logging.getLogger(__name__)


# Async in-memory event bus adapter.
class AsyncInMemoryEventBus(IEventBusPort):
    def __init__(self, stop_timeout_seconds: float = 3.0) -> None:
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        self.handlers: dict[str, list[IEventHandlerPort]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stop_timeout_seconds = float(stop_timeout_seconds)

    async def publish(self, event: Event) -> None:
        await self.queue.put(event)

    def subscribe(self, event_type: str, handler: IEventHandlerPort) -> None:
        bucket = self.handlers.setdefault(event_type, [])
        if handler not in bucket:
            bucket.append(handler)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self.dispatch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=self._stop_timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(
                    "Event bus stop timeout (%.2fs); draining remaining queue synchronously.",
                    self._stop_timeout_seconds,
                )
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

                # Best-effort drain of remaining events to reduce shutdown loss.
                while not self.queue.empty():
                    try:
                        event = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self._dispatch_event(event)
            finally:
                self._task = None

    async def dispatch_loop(self) -> None:
        while self._running or not self.queue.empty():
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            await self._dispatch_event(event)

    async def _dispatch_event(self, event: Event) -> None:
        direct_handlers = self.handlers.get(event.event_type, [])
        wildcard_handlers = self.handlers.get("*", [])

        # De-duplicate handler calls while preserving order.
        unique_handlers: list[IEventHandlerPort] = []
        seen_ids: set[int] = set()
        for handler in [*direct_handlers, *wildcard_handlers]:
            hid = id(handler)
            if hid in seen_ids:
                continue
            seen_ids.add(hid)
            unique_handlers.append(handler)

        for handler in unique_handlers:
            try:
                supported = handler.supported_types()
                if "*" not in supported and event.event_type not in supported:
                    continue
                await handler.handle(event)
            except Exception as handler_error:
                logger.exception(
                    "Event handler failed: handler=%s event_type=%s error=%s",
                    handler.__class__.__name__,
                    event.event_type,
                    handler_error,
                )


# Egyszerű in-memory event store adapter.
class InMemoryEventStore(IEventStorePort):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def append(self, event: Event) -> None:
        self.events.append(event)

    async def append_batch(self, events: list[Event]) -> None:
        self.events.extend(events)

    async def read_stream(self, filters: dict) -> list[Event]:
        event_type = filters.get("event_type")
        if not event_type:
            return list(self.events)
        return [item for item in self.events if item.event_type == event_type]

    async def replay(self, from_ts: datetime, to_ts: datetime, instrument: Optional[Instrument]) -> list[Event]:
        output = [item for item in self.events if from_ts <= item.ts_event <= to_ts]
        if instrument:
            output = [item for item in output if item.payload.get("symbol") == instrument.symbol]
        return output


# Diagram-aligned alias name.
class EventStoreImpl(InMemoryEventStore):
    pass


# Data recorder: bus handlerként ment eventeket store-ba.
class DataRecorder(IEventHandlerPort):
    def __init__(self, store: IEventStorePort, flush_interval_ms: int = 1000) -> None:
        self.store = store
        self.flush_interval_ms = flush_interval_ms
        self.buffer: list[Event] = []
        self._last_flush_monotonic = time.monotonic()

    async def handle(self, event: Event) -> None:
        self.buffer.append(event)
        if len(self.buffer) >= 10:
            await self.flush()
            return

        # Idő alapú flush: kis terhelés mellett se ragadjon bent esemény.
        elapsed_ms = (time.monotonic() - self._last_flush_monotonic) * 1000.0
        if self.flush_interval_ms <= 0 or elapsed_ms >= float(self.flush_interval_ms):
            await self.flush()

    def supported_types(self) -> set[str]:
        return {"*"}

    async def flush(self) -> None:
        if not self.buffer:
            return
        await self.store.append_batch(self.buffer)
        self.buffer = []
        self._last_flush_monotonic = time.monotonic()
