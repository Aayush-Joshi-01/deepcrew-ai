from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.memory.inmemory import InMemoryProvider
from deepcrew.orchestrator import Orchestrator
from deepcrew.stream import StreamPolicy, filter_stream
from deepcrew.types import EventType, StreamEvent


def _make_litellm_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_stream_chunk(text: str | None = None):
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = []
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


async def _fake_stream(*chunks):
    for c in chunks:
        yield c


# --- StreamPolicy presets -----------------------------------------------


def test_chat_policy_allows_text_and_terminal_only():
    policy = StreamPolicy.chat()
    assert policy.allows(EventType.TEXT_DELTA)
    assert policy.allows(EventType.DONE)
    assert policy.allows(EventType.ERROR)
    assert not policy.allows(EventType.TOOL_CALL)
    assert not policy.allows(EventType.SPAWN_AGENT)
    assert not policy.allows(EventType.AGENT_START)


def test_standard_policy_allows_tool_and_lifecycle_events():
    policy = StreamPolicy.standard()
    assert policy.allows(EventType.TEXT_DELTA)
    assert policy.allows(EventType.TOOL_CALL)
    assert policy.allows(EventType.TOOL_RESULT)
    assert policy.allows(EventType.AGENT_START)
    assert policy.allows(EventType.AGENT_DONE)
    assert policy.allows(EventType.DONE)
    assert policy.allows(EventType.ERROR)
    assert not policy.allows(EventType.VERIFIER_SCORED)
    assert not policy.allows(EventType.MEMORY_STORE)


def test_verbose_policy_allows_everything():
    policy = StreamPolicy.verbose()
    for event_type in EventType:
        assert policy.allows(event_type)


def test_custom_include_and_exclude():
    policy = StreamPolicy(include=frozenset({EventType.TEXT_DELTA, EventType.TOOL_CALL}))
    assert policy.allows(EventType.TEXT_DELTA)
    assert policy.allows(EventType.TOOL_CALL)
    assert not policy.allows(EventType.DONE)

    policy2 = StreamPolicy(exclude=frozenset({EventType.TOOL_CALL}))
    assert policy2.allows(EventType.TEXT_DELTA)
    assert not policy2.allows(EventType.TOOL_CALL)


def test_default_policy_include_none_allows_all_unless_excluded():
    policy = StreamPolicy()
    assert policy.allows(EventType.SPAWN_AGENT)
    assert policy.allows(EventType.DONE)


# --- filter_stream ---------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_stream_yields_only_allowed_events():
    async def source():
        yield StreamEvent(EventType.AGENT_START, {}, "a")
        yield StreamEvent(EventType.TEXT_DELTA, {"chunk": "hi"}, "a")
        yield StreamEvent(EventType.TOOL_CALL, {"tool": "x"}, "a")
        yield StreamEvent(EventType.DONE, {}, "a")

    events = [e async for e in filter_stream(source(), StreamPolicy.chat())]
    assert [e.event for e in events] == [EventType.TEXT_DELTA, EventType.DONE]


@pytest.mark.asyncio
async def test_filter_stream_verbose_yields_everything():
    async def source():
        yield StreamEvent(EventType.AGENT_START, {}, "a")
        yield StreamEvent(EventType.TOOL_CALL, {}, "a")
        yield StreamEvent(EventType.DONE, {}, "a")

    events = [e async for e in filter_stream(source(), StreamPolicy.verbose())]
    assert len(events) == 3


# --- Orchestrator.stream(policy=...) end-to-end ----------------------------


@pytest.mark.asyncio
async def test_orchestrator_stream_with_chat_policy_hides_tool_events():
    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    get_weather._is_tool = True
    get_weather._tool_name = "get_weather"
    get_weather._tool_description = "Get weather"

    agents = [Agent("weather_agent", model="openai/gpt-4o", tools=[get_weather])]
    orch = Orchestrator(agents, router_model="openai/gpt-4o-mini")

    router_response = _make_litellm_response(
        json.dumps({"route": "single", "agent": "weather_agent", "task": "weather in Paris"})
    )

    tc = MagicMock()
    tc.index = 0
    tc.id = "call_1"
    tc.function = MagicMock()
    tc.function.name = "get_weather"
    tc.function.arguments = '{"city": "Paris"}'
    tc_chunk = _make_stream_chunk()
    tc_chunk.choices[0].delta.tool_calls = [tc]

    final_chunk = _make_stream_chunk("Sunny.")
    end_chunk = _make_stream_chunk()

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("stream") is False:
            return router_response
        if call_count == 2:
            return _fake_stream(tc_chunk, _make_stream_chunk())
        return _fake_stream(final_chunk, end_chunk)

    events = []
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        async for event in orch.stream("What's the weather in Paris?", policy=StreamPolicy.chat()):
            events.append(event)

    event_types = {e.event for e in events}
    assert EventType.TOOL_CALL not in event_types
    assert EventType.TOOL_RESULT not in event_types
    assert EventType.AGENT_START not in event_types
    assert EventType.DONE in event_types


# --- memory events emitted when a memory provider is attached ---------------


@pytest.mark.asyncio
async def test_memory_events_emitted_with_memory_provider():
    from deepcrew.runner import run_agent

    memory = InMemoryProvider()
    await memory.store("sky_fact", "the sky is blue")
    agent = Agent(name="test", model="openai/gpt-4o", memory=memory)

    chunks = [_make_stream_chunk("Hi."), _make_stream_chunk()]

    import asyncio

    queue: asyncio.Queue = asyncio.Queue()
    with patch("litellm.acompletion", new=AsyncMock(return_value=_fake_stream(*chunks))):
        await run_agent(agent, [{"role": "user", "content": "sky_fact"}], queue=queue)

    events = []
    while not queue.empty():
        events.append(await queue.get())

    types = [e.event for e in events]
    assert EventType.MEMORY_RETRIEVE in types
