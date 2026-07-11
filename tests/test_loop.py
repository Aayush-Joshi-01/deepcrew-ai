from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.loop import LoopConfig, run_agent_loop
from deepcrew.types import AgentResult, EventType
from deepcrew.verifier import Verifier, VerifierFeedback


def _agent_result(agent_id: str, text: str) -> AgentResult:
    return AgentResult(agent_id=agent_id, text=text)


@pytest.mark.asyncio
async def test_loop_convergence_fn_only_regression():
    """Pre-Phase-1 behavior: convergence_fn alone, no verifier, is unchanged."""
    agent = Agent(
        name="test",
        model="openai/gpt-4o",
        loop_config=LoopConfig(
            max_iterations=5,
            convergence_fn=lambda r: "done" in r.text,
        ),
    )

    call_count = 0

    async def fake_run_agent(agent, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        text = "not yet" if call_count < 2 else "done"
        return _agent_result(agent.name, text)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}])

    assert result.text == "done"
    assert result.loop_iterations == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_loop_verifier_drives_convergence_and_refine_message():
    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(
            max_iterations=5,
            verifier=Verifier(),
        ),
    )

    feedback_sequence = [
        VerifierFeedback(score=0.4, issues=["missing citations"], suggestion="add sources", converged=False),
        VerifierFeedback(score=0.9, issues=[], suggestion="", converged=True),
    ]
    eval_call_count = 0
    sent_messages: list[list[dict]] = []

    async def fake_run_agent(agent, messages, **kwargs):
        sent_messages.append(messages)
        return _agent_result(agent.name, f"answer v{len(sent_messages)}")

    async def fake_evaluate(query, result, *, default_model):
        nonlocal eval_call_count
        fb = feedback_sequence[eval_call_count]
        eval_call_count += 1
        return fb

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "Explain CRISPR"}])

    assert result.text == "answer v2"
    assert eval_call_count == 2
    # second call's messages must contain the verifier's issues/suggestion, not the
    # static default refine_prompt
    second_call_messages = sent_messages[1]
    refine_text = second_call_messages[-1]["content"]
    assert "missing citations" in refine_text
    assert "add sources" in refine_text
    assert "Your previous answer was not yet complete" not in refine_text


@pytest.mark.asyncio
async def test_loop_emits_verifier_scored_event():
    import asyncio

    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(max_iterations=1, verifier=Verifier()),
    )

    async def fake_run_agent(agent, messages, **kwargs):
        return _agent_result(agent.name, "an answer")

    async def fake_evaluate(query, result, *, default_model):
        return VerifierFeedback(score=0.95, converged=True)

    queue: asyncio.Queue = asyncio.Queue()
    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate)):
        await run_agent_loop(agent, [{"role": "user", "content": "hi"}], queue=queue)

    events = []
    while not queue.empty():
        events.append(await queue.get())

    scored = [e for e in events if e.event == EventType.VERIFIER_SCORED]
    assert len(scored) == 1
    assert scored[0].data["score"] == 0.95
