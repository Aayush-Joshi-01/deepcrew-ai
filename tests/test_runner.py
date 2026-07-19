from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.exceptions import ToolError
from deepcrew.runner import _execute_tool, _tool_def_to_litellm, run_agent
from deepcrew.tools import fn_to_tool_def, tool
from deepcrew.types import EventType, ToolDef


def _make_chunk(
    content: str | None = None, tool_calls=None, usage=None, finish_reason=None, reasoning=None
):
    """Build a minimal litellm-style chunk object."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
    delta.reasoning_content = reasoning
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _make_stream(*chunks):
    """Return an async generator that yields the given chunks."""

    async def _gen():
        for c in chunks:
            yield c

    return _gen()


@pytest.mark.asyncio
async def test_execute_tool_callable_sync():
    @tool
    def double(x: int) -> int:
        return x * 2

    td = fn_to_tool_def(double)
    result = await _execute_tool({"name": "double", "args": {"x": 3}}, {"double": td})
    assert result == "6"


@pytest.mark.asyncio
async def test_execute_tool_callable_async():
    async def async_tool(x: int) -> str:
        return f"got {x}"

    async_tool._is_tool = True
    async_tool._tool_name = "async_tool"
    async_tool._tool_description = ""
    td = fn_to_tool_def(async_tool)
    result = await _execute_tool({"name": "async_tool", "args": {"x": 5}}, {"async_tool": td})
    assert result == "got 5"


@pytest.mark.asyncio
async def test_execute_tool_unknown_raises():
    with pytest.raises(ToolError, match="Unknown tool"):
        await _execute_tool({"name": "ghost", "args": {}}, {})


@pytest.mark.asyncio
async def test_execute_tool_mcp_client():
    fake_mcp = AsyncMock()
    fake_mcp.call_tool = AsyncMock(return_value={"answer": 42})
    td = ToolDef(name="my_tool", description="", parameters={}, _mcp_client=fake_mcp)
    result = await _execute_tool({"name": "my_tool", "args": {}}, {"my_tool": td})
    assert "42" in result
    fake_mcp.call_tool.assert_awaited_once_with("my_tool", {})


def test_tool_def_to_litellm():
    td = ToolDef(
        name="search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    schema = _tool_def_to_litellm(td)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "Search the web"


@pytest.mark.asyncio
async def test_run_agent_no_tools():
    """Agent with no tool calls returns immediately after first stream."""
    agent = Agent(name="test", model="openai/gpt-4o", system_prompt="You are helpful.")

    chunks = [_make_chunk("Hello "), _make_chunk("world!"), _make_chunk()]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        result = await run_agent(agent, [{"role": "user", "content": "Hi"}])

    assert result.text == "Hello world!"
    assert result.agent_id == "test"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_run_agent_emits_events():
    agent = Agent(name="bot", model="openai/gpt-4o")
    chunks = [_make_chunk("Hi"), _make_chunk()]

    queue: asyncio.Queue = asyncio.Queue()
    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        await run_agent(agent, [{"role": "user", "content": "ping"}], queue=queue)

    events = []
    while not queue.empty():
        events.append(await queue.get())

    types = [e.event for e in events]
    assert EventType.AGENT_START in types
    assert EventType.TEXT_DELTA in types
    assert EventType.AGENT_DONE in types


@pytest.mark.asyncio
async def test_run_agent_emits_thinking_delta_when_reasoning_present():
    agent = Agent(name="bot", model="openai/gpt-4o")
    chunks = [
        _make_chunk(reasoning="Let me think..."),
        _make_chunk("The answer is 4."),
        _make_chunk(),
    ]

    queue: asyncio.Queue = asyncio.Queue()
    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        await run_agent(agent, [{"role": "user", "content": "2+2?"}], queue=queue)

    events = []
    while not queue.empty():
        events.append(await queue.get())

    thinking_events = [e for e in events if e.event == EventType.THINKING_DELTA]
    assert len(thinking_events) == 1
    assert thinking_events[0].data["chunk"] == "Let me think..."


@pytest.mark.asyncio
async def test_run_agent_no_thinking_delta_when_reasoning_absent():
    agent = Agent(name="bot", model="openai/gpt-4o")
    chunks = [_make_chunk("Hi"), _make_chunk()]

    queue: asyncio.Queue = asyncio.Queue()
    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        await run_agent(agent, [{"role": "user", "content": "ping"}], queue=queue)

    events = []
    while not queue.empty():
        events.append(await queue.get())

    assert not any(e.event == EventType.THINKING_DELTA for e in events)


@pytest.mark.asyncio
async def test_run_agent_with_tool_call():
    @tool
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    agent = Agent(name="test", model="openai/gpt-4o", tools=[get_weather])

    # Build mock tool call chunk
    tc = MagicMock()
    tc.index = 0
    tc.id = "call_abc"
    tc.function = MagicMock()
    tc.function.name = "get_weather"
    tc.function.arguments = '{"city": "Paris"}'

    chunk_with_tc = _make_chunk(tool_calls=[tc])
    # Second call returns text (no more tool calls)
    chunk_text = _make_chunk("It is sunny in Paris.")
    chunk_end = _make_chunk()

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_stream(chunk_with_tc, _make_chunk())
        return _make_stream(chunk_text, chunk_end)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await run_agent(
            agent,
            [{"role": "user", "content": "What's the weather in Paris?"}],
        )

    assert "sunny" in result.text.lower() or len(result.tool_calls) > 0


@pytest.mark.asyncio
async def test_run_agent_multimodal_message_memory_search_gets_text_only():
    """Memory search must be given plain text, even when content is a block list."""
    memory = AsyncMock()
    memory.search = AsyncMock(return_value=[])
    agent = Agent(name="test", model="openai/gpt-4o", memory=memory)

    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    }

    chunks = [_make_chunk("A cat."), _make_chunk()]
    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        await run_agent(agent, [multimodal_message])

    memory.search.assert_awaited_once()
    query_arg = memory.search.await_args.args[0]
    assert query_arg == "What is in this image?"


@pytest.mark.asyncio
async def test_run_agent_multimodal_message_sent_untouched_to_litellm():
    """The block-list content sent to litellm.acompletion must be unmodified."""
    agent = Agent(name="test", model="openai/gpt-4o")

    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    }

    chunks = [_make_chunk("A cat."), _make_chunk()]
    mock_completion = AsyncMock(return_value=_make_stream(*chunks))
    with patch("litellm.acompletion", new=mock_completion):
        await run_agent(agent, [multimodal_message])

    sent_messages = mock_completion.call_args.kwargs["messages"]
    sent_user_message = next(m for m in sent_messages if m["role"] == "user")
    assert sent_user_message["content"] == multimodal_message["content"]
