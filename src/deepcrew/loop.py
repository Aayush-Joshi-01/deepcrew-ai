from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .content import extract_text
from .exceptions import LoopConvergedError
from .procedural_memory import ProceduralMemory
from .skills import FunctionSkill, SkillRegistry
from .types import AgentResult, EventType, StreamEvent, ToolDef
from .verifier import Verifier, VerifierFeedback

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


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
    branches: int = 1
    """
    When > 1, run this many parallel candidate continuations per iteration
    (self-consistency) instead of a single linear path. The best branch is
    picked by ``verifier`` score when configured, or merged via
    ``APEXSynthesizer`` otherwise. Multiplies LLM call volume by ``branches``
    per iteration -- use deliberately.
    """
    auto_extract_skill: bool = False
    """
    When True, a loop run that genuinely converges (via ``convergence_fn`` or
    ``verifier``) with a quality signal >= ``skill_confidence_threshold`` is
    distilled into a reusable, replayable ``Skill`` and registered in
    ``SkillRegistry`` (Voyager-inspired). Never triggers on plain
    ``max_iterations`` exhaustion without convergence.
    """
    skill_confidence_threshold: float = 0.85
    """Minimum quality signal (verifier score, or AgentResult.confidence) to distill a skill."""


@dataclass
class LoopState:
    """Tracks state across loop iterations."""

    iteration: int
    results: list[AgentResult] = field(default_factory=list)


async def run_agent_loop(
    agent: Agent,
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
    # Import here to break the circular import: loop → runner → agent → loop.
    # Use the internal _run_agent_turns, never the public run_agent: run_agent
    # re-checks agent.loop_config and would delegate straight back into this
    # function on every call, recursing indefinitely for a looped agent.
    from .runner import _run_agent_turns

    cfg = agent.loop_config
    if cfg is None:
        return await _run_agent_turns(
            agent, messages, tool_defs=tool_defs, queue=queue, agent_id=agent_id
        )

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
            current_messages = [{"role": "system", "content": playbook_block}, *current_messages]

    async def _curate_if_configured() -> None:
        if cfg.procedural_memory is not None and last_feedback is not None:
            count = await cfg.procedural_memory.curate(tag, last_feedback, state.results)
            if queue:
                await queue.put(
                    StreamEvent(
                        EventType.PLAYBOOK_UPDATED,
                        {"task_tag": tag, "entry_count": count},
                        aid,
                    )
                )

    async def _extract_skill_if_configured(
        converged_result: AgentResult, converged_feedback: VerifierFeedback | None
    ) -> None:
        if not cfg.auto_extract_skill:
            return
        quality = (
            converged_feedback.score
            if converged_feedback is not None
            else converged_result.confidence
        )
        if quality is None or quality < cfg.skill_confidence_threshold:
            return

        skill_name = _distilled_skill_name(tag, original_query)
        skill_description = _distilled_skill_description(original_query)
        distilled = _make_distilled_skill(agent, skill_name, skill_description)
        SkillRegistry.register(distilled)
        if queue:
            await queue.put(
                StreamEvent(
                    EventType.SKILL_EXTRACTED,
                    {"skill_name": skill_name, "score": quality},
                    aid,
                )
            )

    for i in range(cfg.max_iterations):
        state.iteration = i

        if queue:
            await queue.put(
                StreamEvent(
                    EventType.LOOP_ITERATION,
                    {"iteration": i, "converged": False},
                    aid,
                )
            )

        if cfg.branches > 1:
            result = await _run_branches(
                agent, current_messages, fetched_tool_defs, queue, aid, cfg, original_query
            )
        else:
            result = await _run_agent_turns(
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
                await queue.put(
                    StreamEvent(
                        EventType.VERIFIER_SCORED,
                        {"iteration": i, "score": feedback.score, "issues": feedback.issues},
                        aid,
                    )
                )

        converged = (cfg.convergence_fn and cfg.convergence_fn(result)) or (
            feedback is not None and feedback.converged
        )
        if converged:
            if queue:
                await queue.put(
                    StreamEvent(
                        EventType.LOOP_ITERATION,
                        {"iteration": i, "converged": True},
                        aid,
                    )
                )
            await _curate_if_configured()
            await _extract_skill_if_configured(result, feedback)
            return result

        if cfg.adaptive and len(scores) >= 2:
            delta = scores[-1] - scores[-2]
            plateau_count = 0 if delta >= cfg.min_improvement else plateau_count + 1
            if plateau_count >= cfg.plateau_patience:
                best_result = max(zip(scores, state.results, strict=False), key=lambda p: p[0])[1]
                if queue:
                    await queue.put(
                        StreamEvent(
                            EventType.LOOP_ITERATION,
                            {"iteration": i, "converged": False, "early_stop": "plateau"},
                            aid,
                        )
                    )
                await _curate_if_configured()
                return best_result

        refine_message = _build_refine_message(cfg.refine_prompt, feedback)
        current_messages = [
            *current_messages,
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": refine_message},
        ]

    await _curate_if_configured()
    return state.results[-1]


async def _run_branches(
    agent: Agent,
    messages: list[dict[str, Any]],
    tool_defs: list[ToolDef] | None,
    queue: asyncio.Queue[StreamEvent | None] | None,
    aid: str,
    cfg: LoopConfig,
    original_query: str,
) -> AgentResult:
    """
    Run cfg.branches parallel candidate continuations for one iteration and
    pick the best (via verifier score) or merge them (via APEXSynthesizer
    when no verifier is configured).
    """
    from .runner import _run_agent_turns

    branch_results = await asyncio.gather(
        *[
            _run_agent_turns(agent, messages, tool_defs=tool_defs, queue=queue, agent_id=aid)
            for _ in range(cfg.branches)
        ]
    )

    total_in = sum(r.input_tokens for r in branch_results)
    total_out = sum(r.output_tokens for r in branch_results)

    if cfg.verifier is not None:
        feedbacks = await asyncio.gather(
            *[
                cfg.verifier.evaluate(original_query, r, default_model=agent.model)
                for r in branch_results
            ]
        )
        winning_index = max(range(len(feedbacks)), key=lambda idx: feedbacks[idx].score)
        winner = branch_results[winning_index]
        winner.input_tokens = total_in
        winner.output_tokens = total_out
        if queue:
            await queue.put(
                StreamEvent(
                    EventType.BRANCH_SELECTED,
                    {
                        "branch_count": cfg.branches,
                        "winning_index": winning_index,
                        "winning_score": feedbacks[winning_index].score,
                    },
                    aid,
                )
            )
        return winner

    from .apex import APEXSynthesizer

    synthesizer = APEXSynthesizer(agent.model)
    merged = await synthesizer.synthesize(original_query, branch_results, queue=queue)
    merged.agent_id = aid
    merged.input_tokens = total_in
    merged.output_tokens = total_out
    if queue:
        await queue.put(
            StreamEvent(
                EventType.BRANCH_SELECTED,
                {
                    "branch_count": cfg.branches,
                    "winning_index": None,
                    "winning_score": merged.confidence,
                },
                aid,
            )
        )
    return merged


def _distilled_skill_name(tag: str, original_query: str) -> str:
    """Deterministic skill name: sanitized task_tag/agent name + a short query hash."""
    digest = hashlib.sha1(original_query.encode("utf-8")).hexdigest()[:8]
    safe_tag = re.sub(r"[^a-zA-Z0-9_]+", "_", tag).strip("_") or "agent"
    return f"{safe_tag}_{digest}"


def _distilled_skill_description(original_query: str) -> str:
    """Cheap, deterministic description built from the original query (no LLM call)."""
    snippet = original_query.strip().replace("\n", " ")[:150]
    return f"Reuses a proven approach that successfully handled: {snippet}"


def _make_distilled_skill(agent: Agent, name: str, description: str) -> FunctionSkill:
    """
    Build a replayable Skill: it re-runs the original agent (same system_prompt,
    tools, mcps, skills) against a new task, rather than memoizing one frozen
    answer -- so the distilled skill generalizes to similar future tasks.
    """
    from .agent import Agent as AgentClass

    async def _replay(task: str) -> str:
        from .runner import run_agent

        replay_agent = AgentClass(
            name=agent.name,
            model=agent.model,
            system_prompt=agent.system_prompt,
            mcps=agent.mcps,
            tools=agent.tools,
            max_turns=agent.max_turns,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
            extra_params=agent.extra_params,
            skills=agent.skills,
        )
        result = await run_agent(replay_agent, [{"role": "user", "content": task}])
        return result.text

    return FunctionSkill(
        _fn=_replay,
        _name=name,
        _description=description,
        _parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific task to complete using this proven approach.",
                },
            },
            "required": ["task"],
        },
    )


def _extract_query(messages: list[dict[str, Any]]) -> str:
    """Return the text of the first user-role message, used by the verifier."""
    for m in messages:
        if m.get("role") == "user":
            text = extract_text(m.get("content"))
            if text:
                return text
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
    agent: Agent,
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
