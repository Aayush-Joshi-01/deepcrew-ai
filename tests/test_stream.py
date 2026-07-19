from __future__ import annotations

import asyncio

import pytest

from deepcrew.stream import make_done_event, make_error_event, queue_to_stream
from deepcrew.types import EventType, StreamEvent


@pytest.mark.asyncio
async def test_queue_to_stream_drains_to_none_sentinel():
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(StreamEvent(EventType.TEXT_DELTA, {"chunk": "a"}, "x"))
    await queue.put(StreamEvent(EventType.TEXT_DELTA, {"chunk": "b"}, "x"))
    await queue.put(None)

    async def noop() -> None:
        return None

    task = asyncio.create_task(noop())
    events = [e async for e in queue_to_stream(queue, task)]

    assert [e.data["chunk"] for e in events] == ["a", "b"]


@pytest.mark.asyncio
async def test_queue_to_stream_propagates_task_exception_after_drain():
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(StreamEvent(EventType.TEXT_DELTA, {"chunk": "a"}, "x"))
    await queue.put(None)

    async def failing() -> None:
        raise RuntimeError("boom")

    task = asyncio.create_task(failing())

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in queue_to_stream(queue, task):
            pass


@pytest.mark.asyncio
async def test_queue_to_stream_yields_nothing_when_immediately_terminated():
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(None)

    async def noop() -> None:
        return None

    task = asyncio.create_task(noop())
    events = [e async for e in queue_to_stream(queue, task)]
    assert events == []


def test_stream_event_to_sse_shape():
    event = StreamEvent(EventType.TEXT_DELTA, {"chunk": "hi"}, "agent1")
    sse = event.to_sse()

    assert sse.startswith("event: text_delta\n")
    assert sse.endswith("\n\n")
    assert '"chunk": "hi"' in sse
    assert '"agent_id": "agent1"' in sse


def test_stream_event_to_dict_shape():
    event = StreamEvent(EventType.DONE, {"final_text": "x"}, "agent1")
    d = event.to_dict()
    assert d == {"event": "done", "agent_id": "agent1", "final_text": "x"}


def test_stream_event_default_agent_id_is_empty_string():
    event = StreamEvent(EventType.ERROR, {"message": "oops"})
    assert event.agent_id == ""


def test_make_error_event():
    event = make_error_event("agent1", "oops")
    assert event.event == EventType.ERROR
    assert event.agent_id == "agent1"
    assert event.data == {"message": "oops"}


def test_make_done_event_default_agent_id():
    event = make_done_event()
    assert event.event == EventType.DONE
    assert event.agent_id == ""
    assert event.data == {}


def test_make_done_event_with_agent_id():
    event = make_done_event("agent1")
    assert event.agent_id == "agent1"
