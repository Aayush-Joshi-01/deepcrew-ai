from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from .types import EventType, StreamEvent

# A consumer must always be able to tell when a stream ends, so presets never
# filter these out. A fully custom StreamPolicy(include={...}) built by hand
# can still exclude them -- that's on the caller.
_TERMINAL_EVENTS: frozenset[EventType] = frozenset({EventType.DONE, EventType.ERROR})

_CHAT_EVENTS: frozenset[EventType] = frozenset({EventType.TEXT_DELTA}) | _TERMINAL_EVENTS

_STANDARD_EVENTS: frozenset[EventType] = _CHAT_EVENTS | frozenset(
    {
        EventType.AGENT_START,
        EventType.AGENT_DONE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.TOOL_DENIED,
        EventType.STEP_START,
        EventType.STEP_DONE,
        EventType.SPAWN_AGENT,
    }
)


@dataclass(frozen=True)
class StreamPolicy:
    """
    Controls which :class:`EventType`\\ s are visible on a stream.

    ``include=None`` (the default) means "no include filter" -- every event
    type passes, subject only to ``exclude``. Use the presets for common
    cases, or build a fully custom policy directly::

        StreamPolicy.chat()       # response text only, for simple chatbots
        StreamPolicy.standard()   # + tool calls and agent/step lifecycle
        StreamPolicy.verbose()    # everything, for technical/debug UIs
        StreamPolicy(include={EventType.TEXT_DELTA, EventType.TOOL_CALL})

    Filtering via :func:`filter_stream` is purely a view over the event
    stream -- it never affects execution, logging, or OpenTelemetry spans.
    """

    include: frozenset[EventType] | None = None
    exclude: frozenset[EventType] = field(default_factory=frozenset)

    def allows(self, event_type: EventType) -> bool:
        if self.include is not None and event_type not in self.include:
            return False
        return event_type not in self.exclude

    @classmethod
    def chat(cls) -> StreamPolicy:
        """Response text deltas plus the terminal done/error events only."""
        return cls(include=_CHAT_EVENTS)

    @classmethod
    def standard(cls) -> StreamPolicy:
        """Chat events plus tool calls/results and agent/step lifecycle."""
        return cls(include=_STANDARD_EVENTS)

    @classmethod
    def verbose(cls) -> StreamPolicy:
        """Every event type -- no include filter."""
        return cls(include=None)


async def filter_stream(
    stream: AsyncGenerator[StreamEvent, None],
    policy: StreamPolicy,
) -> AsyncGenerator[StreamEvent, None]:
    """Yield only the events a :class:`StreamPolicy` allows."""
    async for event in stream:
        if policy.allows(event.event):
            yield event


async def queue_to_stream(
    queue: asyncio.Queue[StreamEvent | None],
    task: asyncio.Task[Any],
) -> AsyncGenerator[StreamEvent, None]:
    """
    Drain a StreamEvent queue until a sentinel ``None`` is received.

    Awaits the background task after the queue is exhausted so that any
    exception raised inside it propagates to the caller.
    """
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event
    await task


def make_error_event(agent_id: str, message: str) -> StreamEvent:
    return StreamEvent(EventType.ERROR, {"message": message}, agent_id)


def make_done_event(agent_id: str = "") -> StreamEvent:
    return StreamEvent(EventType.DONE, {}, agent_id)
