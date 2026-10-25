"""Smallest possible real call: one agent, one tool, one turn."""

from __future__ import annotations

import pytest

from deepcrew import Agent, run_agent, tool

pytestmark = pytest.mark.live(
    feature="core_agent",
    watch=[
        "src/deepcrew/runner.py",
        "src/deepcrew/agent.py",
        "src/deepcrew/tools.py",
    ],
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.mark.asyncio
async def test_single_agent_calls_tool(live_model: str):
    agent = Agent(
        name="calc",
        model=live_model,
        system_prompt="Use the add tool for any arithmetic. Then state the number.",
        tools=[add],
        max_turns=4,
    )
    result = await run_agent(agent, [{"role": "user", "content": "What is 21 + 21?"}])

    assert result.text.strip()
    assert any(tc["tool"] == "add" for tc in result.tool_calls)
    assert "42" in result.text
