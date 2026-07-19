from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.hooks import AgentHooks
from deepcrew.runner import run_agent
from deepcrew.tools import tool
from deepcrew.types import EventType


def _make_chunk(content: str | None = None, tool_calls=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_stream(*chunks):
    async def _gen():
        for c in chunks:
            yield c

    return _gen()


def _make_tool_call_chunk(name: str, args: dict):
    import json

    tc = MagicMock()
    tc.index = 0
    tc.id = "call_1"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return _make_chunk(tool_calls=[tc])


@pytest.mark.asyncio
async def test_approve_tool_false_denies_call_without_executing():
    calls: list[str] = []

    @tool
    def dangerous_tool(x: int) -> str:
        calls.append("executed")
        return "should not run"

    async def deny(name: str, args: dict) -> bool:
        return False

    agent = Agent(
        name="t",
        model="openai/gpt-4o",
        tools=[dangerous_tool],
        hooks=AgentHooks(approve_tool=deny),
    )

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_stream(_make_tool_call_chunk("dangerous_tool", {"x": 1}), _make_chunk())
        return _make_stream(_make_chunk("done"), _make_chunk())

    import asyncio

    queue: asyncio.Queue = asyncio.Queue()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        await run_agent(agent, [{"role": "user", "content": "do it"}], queue=queue)

    assert calls == []  # tool callable was never invoked

    events = []
    while not queue.empty():
        events.append(await queue.get())
    denied = [e for e in events if e.event == EventType.TOOL_DENIED]
    assert len(denied) == 1
    assert denied[0].data["tool"] == "dangerous_tool"


@pytest.mark.asyncio
async def test_denied_tool_result_content_in_history():
    @tool
    def some_tool(x: int) -> str:
        return "ran"

    async def deny(name: str, args: dict) -> bool:
        return False

    agent = Agent(
        name="t", model="openai/gpt-4o", tools=[some_tool], hooks=AgentHooks(approve_tool=deny)
    )

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_stream(_make_tool_call_chunk("some_tool", {"x": 1}), _make_chunk())
        assert "denied by user" in kwargs["messages"][-1]["content"]
        return _make_stream(_make_chunk("acknowledged"), _make_chunk())

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await run_agent(agent, [{"role": "user", "content": "go"}])

    assert result.text == "acknowledged"


@pytest.mark.asyncio
async def test_hooks_called_in_order_with_expected_args():
    @tool
    def echo(x: int) -> str:
        return f"echo {x}"

    events_seen: list[tuple[str, ...]] = []

    async def on_start():
        events_seen.append(("agent_start",))

    async def on_tool_start(name: str, args: dict):
        events_seen.append(("tool_start", name, str(args)))

    async def on_tool_end(name: str, result: str):
        events_seen.append(("tool_end", name, result))

    async def approve(name: str, args: dict) -> bool:
        events_seen.append(("approve", name))
        return True

    agent = Agent(
        name="t",
        model="openai/gpt-4o",
        tools=[echo],
        hooks=AgentHooks(
            on_agent_start=on_start,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            approve_tool=approve,
        ),
    )

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_stream(_make_tool_call_chunk("echo", {"x": 5}), _make_chunk())
        return _make_stream(_make_chunk("done"), _make_chunk())

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        await run_agent(agent, [{"role": "user", "content": "go"}])

    kinds = [e[0] for e in events_seen]
    assert kinds == ["agent_start", "approve", "tool_start", "tool_end"]
    assert events_seen[1][1] == "echo"
    assert events_seen[3][2] == "echo 5"


@pytest.mark.asyncio
async def test_raising_hook_does_not_crash_the_run():
    @tool
    def echo(x: int) -> str:
        return f"echo {x}"

    async def on_start():
        raise RuntimeError("boom")

    async def on_tool_start(name: str, args: dict):
        raise RuntimeError("boom too")

    agent = Agent(
        name="t",
        model="openai/gpt-4o",
        tools=[echo],
        hooks=AgentHooks(on_agent_start=on_start, on_tool_start=on_tool_start),
    )

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_stream(_make_tool_call_chunk("echo", {"x": 1}), _make_chunk())
        return _make_stream(_make_chunk("done"), _make_chunk())

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await run_agent(agent, [{"role": "user", "content": "go"}])

    assert result.text == "done"


@pytest.mark.asyncio
async def test_no_hooks_configured_runs_normally():
    @tool
    def echo(x: int) -> str:
        return f"echo {x}"

    agent = Agent(name="t", model="openai/gpt-4o", tools=[echo])

    call_count = 0

    async def fake_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_stream(_make_tool_call_chunk("echo", {"x": 3}), _make_chunk())
        return _make_stream(_make_chunk("done"), _make_chunk())

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await run_agent(agent, [{"role": "user", "content": "go"}])

    assert result.text == "done"
