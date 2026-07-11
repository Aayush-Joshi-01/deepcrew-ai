from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .exceptions import LoopConvergedError
from .procedural_memory import ProceduralMemory
from .types import AgentResult, EventType, StreamEvent, ToolDef
from .verifier import Verifier, VerifierFeedback

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
    verifier: Verifier | None = None
    """
    Optional structured critic. When set, its ``VerifierFeedback`` (score +
    specific issues + a concrete suggestion) drives both convergence and the
    refinement prompt sent for the next iteration, instead of the static
    ``refine_prompt`` string. Either ``convergence_fn`` or the verifier
    reporting ``converged=True`` is sufficient to stop the loop.
    """
    procedural_memory: ProceduralMemory | None = None
    """
    Optional durable, evolving playbook (see :class:`ProceduralMemory`). When
    set, accumulated strategies for ``task_tag`` are read and injected before
    the first iteration, and curated (merged with new insights) when the loop
    exits -- but only if ``verifier`` is also configured, since curation needs
    a ``VerifierFeedback`` to grade the run. With no verifier, this is a no-op.
    """
    task_tag: str | None = None
    """Playbook namespace for ``procedural_memory``. Defaults to ``agent.name``."""
    adaptive: bool = False
    """
    When True, track the verifier score across iterations and stop early once
    improvement plateaus (see ``min_improvement``/``plateau_patience``),
    instead of always running ``max_iterations``. Requires ``verifier`` to be
    set -- with no verifier there is no score to track, so this is a no-op.
    Can only shorten the loop; ``max_iterations`` remains a hard ceiling.
    """
    min_improvement: float = 0.02
    """Minimum verifier-score delta between iterations to count as still improving."""
    plateau_patience: int = 2
    """Consecutive non-improving iterations tolerated before an adaptive early stop."""


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
    original_query = _extract_query(messages)
    tag = cfg.task_tag or agent.name
    last_feedback: VerifierFeedback | None = None
    scores: list[float] = []
    plateau_count = 0

    if cfg.procedural_memory is not None:
        playbook_entries = await cfg.procedural_memory.load(tag)
        playbook_block = cfg.procedural_memory.render(playbook_entries)
        if playbook_block:
            current_messages = [{"role": "system", "content": playbook_block}] + current_messages

    async def _curate_if_configured() -> None:
        if cfg.procedural_memory is not None and last_feedback is not None:
            count = await cfg.procedural_memory.curate(tag, last_feedback, state.results)
            if queue:
                await queue.put(StreamEvent(
                    EventType.PLAYBOOK_UPDATED,
                    {"task_tag": tag, "entry_count": count},
                    aid,
                ))

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
            await _curate_if_configured()
            raise LoopConvergedError(
                f"Agent {aid!r} loop stopped at iteration {i} by stop_condition",
                result=result,
            )

        feedback: VerifierFeedback | None = None
        if cfg.verifier is not None:
            feedback = await cfg.verifier.evaluate(
                original_query, result, default_model=agent.model
            )
            last_feedback = feedback
            scores.append(feedback.score)
            if queue:
                await queue.put(StreamEvent(
                    EventType.VERIFIER_SCORED,
                    {"iteration": i, "score": feedback.score, "issues": feedback.issues},
                    aid,
                ))

        converged = (cfg.convergence_fn and cfg.convergence_fn(result)) or (
            feedback is not None and feedback.converged
        )
        if converged:
            if queue:
                await queue.put(StreamEvent(
                    EventType.LOOP_ITERATION,
                    {"iteration": i, "converged": True},
                    aid,
                ))
            await _curate_if_configured()
            return result

        if cfg.adaptive and len(scores) >= 2:
            delta = scores[-1] - scores[-2]
            plateau_count = 0 if delta >= cfg.min_improvement else plateau_count + 1
            if plateau_count >= cfg.plateau_patience:
                best_result = max(zip(scores, state.results), key=lambda p: p[0])[1]
                if queue:
                    await queue.put(StreamEvent(
                        EventType.LOOP_ITERATION,
                        {"iteration": i, "converged": False, "early_stop": "plateau"},
                        aid,
                    ))
                await _curate_if_configured()
                return best_result

        refine_message = _build_refine_message(cfg.refine_prompt, feedback)
        current_messages = current_messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": refine_message},
        ]

    await _curate_if_configured()
    return state.results[-1]


def _extract_query(messages: list[dict[str, Any]]) -> str:
    """Return the first user-role message content, used by the verifier."""
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _build_refine_message(default_prompt: str, feedback: VerifierFeedback | None) -> str:
    """Build the next refinement prompt from verifier feedback when available."""
    if feedback is None or (not feedback.issues and not feedback.suggestion):
        return default_prompt

    parts = ["Your previous answer needs improvement."]
    if feedback.issues:
        bullets = "\n".join(f"- {issue}" for issue in feedback.issues)
        parts.append(f"Specific issues:\n{bullets}")
    if feedback.suggestion:
        parts.append(f"Suggestion: {feedback.suggestion}")
    return "\n\n".join(parts)


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
