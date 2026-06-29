from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import litellm

from .types import AgentResult, EventType, StreamEvent, ToolDef


@dataclass
class SpawnRequest:
    """Parameters for dynamically spawning a sub-agent."""

    task: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    system_prompt: str | None = None
    max_turns: int = 5


class ToolAllocator:
    """
    Given a task description and a pool of ToolDefs, selects the most relevant
    subset using the router LLM.
    """

    def __init__(self, router_model: str) -> None:
        self._model = router_model

    async def allocate(
        self,
        task: str,
        all_tools: list[ToolDef],
        max_tools: int = 10,
    ) -> list[ToolDef]:
        """Return at most *max_tools* ToolDefs relevant to *task*."""
        if not all_tools:
            return []

        tool_descriptions = "\n".join(
            f"- {td.name}: {td.description}" for td in all_tools
        )
        prompt = (
            f"Task: {task}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            "Select the most relevant tools for this task. "
            "Return ONLY a JSON array of tool names, e.g. [\"tool_a\", \"tool_b\"]."
        )

        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "[]"
            # The model may return {"tools": [...]} or a bare array
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                names = next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
            else:
                names = parsed if isinstance(parsed, list) else []
        except Exception:
            # Fallback: return all tools up to max_tools
            return all_tools[:max_tools]

        name_set = set(str(n) for n in names)
        selected = [td for td in all_tools if td.name in name_set]
        return selected[:max_tools]


async def spawn_agent(
    request: SpawnRequest,
    all_tool_defs: list[ToolDef],
    parent_queue: asyncio.Queue[StreamEvent | None] | None = None,
    router_model: str = "openai/gpt-4o-mini",
    parent_agent_id: str = "parent",
) -> AgentResult:
    """
    Dynamically create and run a sub-agent with intelligently allocated tools.

    Emits a SPAWN_AGENT event before the sub-agent starts.
    """
    from .agent import Agent
    from .runner import run_agent

    if parent_queue:
        await parent_queue.put(StreamEvent(
            EventType.SPAWN_AGENT,
            {"task": request.task[:200], "requested_tools": request.tools},
            parent_agent_id,
        ))

    allocator = ToolAllocator(router_model)
    allocated = await allocator.allocate(request.task, all_tool_defs)

    # If the request specified explicit tool names, filter to those
    if request.tools:
        requested_set = set(request.tools)
        allocated = [td for td in allocated if td.name in requested_set] or allocated

    sub_agent = Agent(
        name=f"spawned_{parent_agent_id}",
        model=request.model or router_model,
        system_prompt=request.system_prompt or "You are a helpful sub-agent. Complete the given task.",
        max_turns=request.max_turns,
    )

    return await run_agent(
        sub_agent,
        [{"role": "user", "content": request.task}],
        tool_defs=allocated,
        queue=parent_queue,
        agent_id=f"spawned_{parent_agent_id}",
    )


def make_spawn_tool(
    all_tool_defs: list[ToolDef],
    parent_queue: asyncio.Queue[StreamEvent | None] | None,
    router_model: str,
    parent_agent_id: str,
) -> ToolDef:
    """
    Return a ToolDef named 'spawn_agent' that the LLM can call to create
    a sub-agent dynamically. The tool is backed by a closure over spawn_agent().
    """
    async def _spawn(
        task: str,
        model: str = "",
        system_prompt: str = "",
    ) -> str:
        req = SpawnRequest(
            task=task,
            model=model or None,
            system_prompt=system_prompt or None,
        )
        result = await spawn_agent(
            req,
            all_tool_defs,
            parent_queue=parent_queue,
            router_model=router_model,
            parent_agent_id=parent_agent_id,
        )
        return result.text

    return ToolDef(
        name="spawn_agent",
        description=(
            "Spawn a sub-agent to handle a specific sub-task. "
            "The sub-agent will be automatically equipped with relevant tools. "
            "Use when the current task requires a specialized independent worker."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific task for the sub-agent to complete",
                },
                "model": {
                    "type": "string",
                    "description": "Optional LiteLLM model string override",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt for the sub-agent",
                },
            },
            "required": ["task"],
        },
        _callable=_spawn,
    )
