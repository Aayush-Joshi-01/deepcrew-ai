from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deepcrew.spawner import SpawnRequest, make_spawn_tool, spawn_agent
from deepcrew.types import AgentResult, ToolDef
from deepcrew.verifier import Verifier


def _search_tool() -> ToolDef:
    return ToolDef(name="search", description="Search the web", parameters={"type": "object", "properties": {}})


async def _fake_run_agent_factory(sink: list):
    async def fake_run_agent(agent, messages, *, tool_defs=None, queue=None, agent_id=None):
        sink.append(tool_defs or [])
        return AgentResult(agent_id=agent_id or agent.name, text="done")
    return fake_run_agent


@pytest.mark.asyncio
async def test_spawn_agent_max_depth_1_no_nested_tool():
    received: list[list[ToolDef]] = []
    fake_run_agent = await _fake_run_agent_factory(received)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("no allocator needed"))):
        req = SpawnRequest(task="do X")
        await spawn_agent(req, [_search_tool()], max_depth=1)

    tool_names = [td.name for td in received[0]]
    assert "spawn_agent" not in tool_names


@pytest.mark.asyncio
async def test_spawn_agent_max_depth_2_first_level_gets_nested_tool_which_terminates_at_cap():
    received: list[list[ToolDef]] = []
    fake_run_agent = await _fake_run_agent_factory(received)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("no allocator needed"))):
        req = SpawnRequest(task="do X")
        await spawn_agent(req, [_search_tool()], max_depth=2)

        first_level_tools = received[0]
        tool_names = [td.name for td in first_level_tools]
        assert "spawn_agent" in tool_names

        nested_tool = next(td for td in first_level_tools if td.name == "spawn_agent")
        result_text = await nested_tool._callable(task="do Y")

    assert result_text == "done"
    second_level_tools = received[-1]
    assert "spawn_agent" not in [td.name for td in second_level_tools]


@pytest.mark.asyncio
async def test_spawn_tool_backstop_when_current_depth_at_cap():
    tool = make_spawn_tool([], None, "openai/gpt-4o-mini", "parent", current_depth=2, max_depth=2)
    result = await tool._callable(task="anything")
    assert "Maximum nesting depth reached" in result


@pytest.mark.asyncio
async def test_spawn_agent_complexity_check_gates_nested_tool():
    received: list[list[ToolDef]] = []
    fake_run_agent = await _fake_run_agent_factory(received)
    gate = Verifier()

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("no allocator needed"))), \
         patch.object(Verifier, "assess_complexity", new=AsyncMock(return_value=False)):
        req = SpawnRequest(task="a trivially simple task")
        await spawn_agent(req, [_search_tool()], max_depth=3, complexity_check=gate)

    tool_names = [td.name for td in received[0]]
    assert "spawn_agent" not in tool_names


@pytest.mark.asyncio
async def test_spawn_agent_complexity_check_allows_nested_tool_when_true():
    received: list[list[ToolDef]] = []
    fake_run_agent = await _fake_run_agent_factory(received)
    gate = Verifier()

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("no allocator needed"))), \
         patch.object(Verifier, "assess_complexity", new=AsyncMock(return_value=True)):
        req = SpawnRequest(task="a genuinely complex multi-part task")
        await spawn_agent(req, [_search_tool()], max_depth=3, complexity_check=gate)

    tool_names = [td.name for td in received[0]]
    assert "spawn_agent" in tool_names


@pytest.mark.asyncio
async def test_spawn_agent_default_max_depth_matches_flat_regression():
    """Default max_depth=2 still allows exactly one nested level -- flat single-level
    spawning (as used before this phase) keeps working when nesting isn't invoked."""
    received: list[list[ToolDef]] = []
    fake_run_agent = await _fake_run_agent_factory(received)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("no allocator needed"))):
        req = SpawnRequest(task="do X")
        result = await spawn_agent(req, [_search_tool()])

    assert result.text == "done"
