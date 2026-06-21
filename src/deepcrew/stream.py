from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from .types import EventType, StreamEvent


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
