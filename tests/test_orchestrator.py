from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.exceptions import RouterError
from deepcrew.orchestrator import Orchestrator
from deepcrew.types import EventType, ToolDef


def _make_litellm_response(text: str):
    """Build a non-streaming litellm response with the given content."""
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


@pytest.mark.asyncio
async def test_orchestrator_single_route():
    agents = [
        Agent("researcher", model="openai/gpt-4o", system_prompt="Research specialist."),
    ]
    orch = Orchestrator(agents, router_model="openai/gpt-4o-mini")

    router_response = _make_litellm_response(
        json.dumps({"route": "single", "agent": "researcher", "task": "Research AI trends"})
    )

    agent_chunks = [_make_stream_chunk("AI is advancing rapidly."), _make_stream_chunk()]

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("stream") is False:
            return router_response
        return _fake_stream(*agent_chunks)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await orch.run("What are the latest AI trends?")

    assert "AI" in result.final_text or len(result.agent_results) > 0


@pytest.mark.asyncio
async def test_orchestrator_stream_yields_events():
    agents = [
        Agent("writer", model="openai/gpt-4o", system_prompt="Writer."),
    ]
    orch = Orchestrator(agents, router_model="openai/gpt-4o-mini")

    router_response = _make_litellm_response(
        json.dumps({"route": "single", "agent": "writer", "task": "Write a poem"})
    )
    agent_chunks = [_make_stream_chunk("Roses are red..."), _make_stream_chunk()]

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("stream") is False:
            return router_response
        return _fake_stream(*agent_chunks)

    events = []
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        async for event in orch.stream("Write me a poem"):
            events.append(event)

    event_types = {e.event for e in events}
    assert EventType.AGENT_START in event_types
    assert EventType.DONE in event_types


@pytest.mark.asyncio
async def test_orchestrator_invalid_router_json_raises():
    agents = [Agent("a", model="openai/gpt-4o")]
    orch = Orchestrator(agents, router_model="openai/gpt-4o-mini")

    bad_response = _make_litellm_response("not json at all @@##")

    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=bad_response)),
        pytest.raises(RouterError),
    ):
        await orch._route("some query")


@pytest.mark.asyncio
async def test_orchestrator_router_sees_attachment_note_not_base64():
    from deepcrew.content import image

    agents = [Agent("researcher", model="openai/gpt-4o", system_prompt="Research specialist.")]
    orch = Orchestrator(agents, router_model="openai/gpt-4o-mini")

    router_response = _make_litellm_response(
        json.dumps({"route": "single", "agent": "researcher", "task": "Describe the photo"})
    )
    agent_chunks = [_make_stream_chunk("It's a cat."), _make_stream_chunk()]

    captured_router_calls = []

    async def fake_completion(**kwargs):
        if kwargs.get("stream") is False:
            captured_router_calls.append(kwargs)
            return router_response
        return _fake_stream(*agent_chunks)

    attachments = [image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        await orch.run("What's in this photo?", attachments=attachments)

    assert len(captured_router_calls) == 1
    router_user_message = captured_router_calls[0]["messages"][-1]["content"]
    assert isinstance(router_user_message, str)
    assert "[attachments: 1 image]" in router_user_message
    assert "base64" not in router_user_message
    assert attachments[0].url.split(",", 1)[1] not in router_user_message


@pytest.mark.asyncio
async def test_orchestrator_executing_agent_receives_attachments():
    from deepcrew.content import image

    agents = [Agent("researcher", model="openai/gpt-4o", system_prompt="Research specialist.")]
    orch = Orchestrator(agents, router_model="openai/gpt-4o-mini")

    router_response = _make_litellm_response(
        json.dumps({"route": "single", "agent": "researcher", "task": "Describe the photo"})
    )
    agent_chunks = [_make_stream_chunk("It's a cat."), _make_stream_chunk()]

    captured_agent_calls = []

    async def fake_completion(**kwargs):
        if kwargs.get("stream") is False:
            return router_response
        captured_agent_calls.append(kwargs)
        return _fake_stream(*agent_chunks)

    img = image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        await orch.run("What's in this photo?", attachments=[img])

    assert len(captured_agent_calls) == 1
    agent_user_message = captured_agent_calls[0]["messages"][-1]["content"]
    assert isinstance(agent_user_message, list)
    assert agent_user_message[-1] == {"type": "image_url", "image_url": {"url": img.url}}


@pytest.mark.asyncio
async def test_orchestrator_default_spawn_still_flat_single_level():
    """Regression: default Orchestrator(enable_spawn=True) with no max_spawn_depth
    override behaves like the pre-Phase-6 flat single-level spawn when the
    spawn_agent tool is never actually invoked by the agent."""
    search_tool = ToolDef(
        name="search", description="Search the web", parameters={"type": "object", "properties": {}}
    )
    agents = [Agent("researcher", model="openai/gpt-4o", system_prompt="Research specialist.")]
    orch = Orchestrator(
        agents,
        router_model="openai/gpt-4o-mini",
        global_tools=[search_tool],
        enable_spawn=True,
    )
    assert orch._max_spawn_depth == 2
    assert orch._spawn_complexity_check is None

    router_response = _make_litellm_response(
        json.dumps({"route": "single", "agent": "researcher", "task": "Research AI trends"})
    )
    agent_chunks = [_make_stream_chunk("AI is advancing rapidly."), _make_stream_chunk()]

    async def fake_completion(**kwargs):
        if kwargs.get("stream") is False:
            return router_response
        return _fake_stream(*agent_chunks)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await orch.run("What are the latest AI trends?")

    assert "AI" in result.final_text or len(result.agent_results) > 0
