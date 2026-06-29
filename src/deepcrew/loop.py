from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .exceptions import LoopConvergedError
from .types import AgentResult, EventType, StreamEvent, ToolDef

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class LoopConfig:
    """Controls the outer iteration loop over full agent runs."""

    max_iterations: int = 5
    convergence_fn: Callable[[AgentResult], bool] | None = None
    stop_condition: Callable[[AgentResult], bool] | None = None
    refine_prompt: str = (
        "Your previous answer was not yet complete or sufficiently accurate. "
        "Please refine and improve it."
    )


@dataclass
class LoopState:
    """Tracks state across loop iterations."""

    iteration: int
    results: list[AgentResult] = field(default_factory=list)


async def run_agent_loop(
    agent: "Agent",
    messages: list[dict[str, Any]],
    *,
    tool_defs: list[ToolDef] | None = None,
    queue: asyncio.Queue[StreamEvent | None] | None = None,
    agent_id: str | None = None,
) -> AgentResult:
    """
    Outer iteration loop that repeatedly calls run_agent until convergence.

    Distinct from max_turns (which governs tool-call cycles within one run),
    this loop re-prompts the agent with a refinement message when the result
    does not satisfy convergence_fn.
    """
    # Import here to break the circular import: loop → runner → agent → loop
    from .runner import run_agent

    cfg = agent.loop_config
    if cfg is None:
        return await run_agent(agent, messages, tool_defs=tool_defs, queue=queue, agent_id=agent_id)

    aid = agent_id or agent.name
    state = LoopState(iteration=0)
    current_messages = list(messages)
    fetched_tool_defs = tool_defs

    for i in range(cfg.max_iterations):
        state.iteration = i

        if queue:
            await queue.put(StreamEvent(
                EventType.LOOP_ITERATION,
                {"iteration": i, "converged": False},
                aid,
            ))

        result = await run_agent(
            agent,
            current_messages,
            tool_defs=fetched_tool_defs,
            queue=queue,
            agent_id=aid,
        )
        result.loop_iterations = i + 1
        state.results.append(result)

        if cfg.stop_condition and cfg.stop_condition(result):
            raise LoopConvergedError(
                f"Agent {aid!r} loop stopped at iteration {i} by stop_condition",
                result=result,
            )

        if cfg.convergence_fn and cfg.convergence_fn(result):
            if queue:
                await queue.put(StreamEvent(
                    EventType.LOOP_ITERATION,
                    {"iteration": i, "converged": True},
                    aid,
                ))
            return result

        current_messages = current_messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": cfg.refine_prompt},
        ]

    return state.results[-1]


async def search_loop(
    query: str,
    search_tool: Callable[..., Any],
    *,
    agent: "Agent",
    max_iterations: int = 3,
    confidence_threshold: float = 0.8,
    queue: asyncio.Queue[StreamEvent | None] | None = None,
) -> AgentResult:
    """
    Convenience wrapper: run an agent in a loop until its confidence score
    meets *confidence_threshold* or *max_iterations* is exhausted.
    """
    from .agent import Agent as AgentClass

    loop_agent = AgentClass(
        name=agent.name,
        model=agent.model,
        system_prompt=agent.system_prompt,
        tools=[search_tool],
        max_turns=agent.max_turns,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens,
        extra_params=agent.extra_params,
        loop_config=LoopConfig(
            max_iterations=max_iterations,
            convergence_fn=lambda r: (r.confidence or 0.0) >= confidence_threshold,
        ),
    )

    return await run_agent_loop(
        loop_agent,
        [{"role": "user", "content": query}],
        queue=queue,
    )
