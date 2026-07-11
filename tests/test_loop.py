from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.loop import LoopConfig, run_agent_loop
from deepcrew.memory.inmemory import InMemoryProvider
from deepcrew.procedural_memory import ProceduralMemory
from deepcrew.types import AgentResult, EventType
from deepcrew.verifier import Verifier, VerifierConfig, VerifierFeedback


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


@pytest.mark.asyncio
async def test_loop_playbook_injected_on_second_run_with_shared_procedural_memory():
    backend = InMemoryProvider()
    pm = ProceduralMemory(backend)

    def make_agent():
        return Agent(
            name="researcher",
            model="openai/gpt-4o",
            loop_config=LoopConfig(
                max_iterations=1,
                verifier=Verifier(),
                procedural_memory=pm,
            ),
        )

    sent_messages: list[list[dict]] = []

    async def fake_run_agent(agent, messages, **kwargs):
        sent_messages.append(messages)
        return _agent_result(agent.name, "an answer")

    async def fake_evaluate_high(query, result, *, default_model):
        return VerifierFeedback(score=0.95, issues=[], suggestion="", converged=True)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate_high)):
        await run_agent_loop(make_agent(), [{"role": "user", "content": "Explain CRISPR"}])
        await run_agent_loop(make_agent(), [{"role": "user", "content": "Explain CRISPR again"}])

    # First run: no playbook yet (nothing persisted).
    first_run_messages = sent_messages[0]
    assert not any(
        m.get("role") == "system" and "Known strategies" in str(m.get("content", ""))
        for m in first_run_messages
    )

    # Second run: playbook curated from the first run's high-confidence result is injected.
    second_run_messages = sent_messages[1]
    assert any(
        m.get("role") == "system" and "Known strategies" in str(m.get("content", ""))
        for m in second_run_messages
    )


@pytest.mark.asyncio
async def test_loop_procedural_memory_without_verifier_is_noop():
    backend = InMemoryProvider()
    pm = ProceduralMemory(backend)

    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(
            max_iterations=1,
            convergence_fn=lambda r: True,
            procedural_memory=pm,
        ),
    )

    async def fake_run_agent(agent, messages, **kwargs):
        return _agent_result(agent.name, "an answer")

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}])

    assert result.text == "an answer"
    assert await pm.load("researcher") == []


@pytest.mark.asyncio
async def test_loop_adaptive_stops_early_on_plateau():
    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(
            max_iterations=10,
            verifier=Verifier(),
            adaptive=True,
            min_improvement=0.02,
            plateau_patience=2,
        ),
    )

    # deltas: 0.1, 0.01, 0.005 -- last two are below min_improvement=0.02,
    # so plateau_count hits plateau_patience=2 on the 4th score.
    score_sequence = [0.5, 0.6, 0.61, 0.615]
    call_count = 0

    async def fake_run_agent(agent, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return _agent_result(agent.name, f"answer{call_count}")

    async def fake_evaluate(query, result, *, default_model):
        idx = call_count - 1
        return VerifierFeedback(score=score_sequence[idx], converged=False)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}])

    assert call_count == 4  # stopped well short of max_iterations=10
    assert result.text == "answer4"  # highest-scoring (0.615) result returned


@pytest.mark.asyncio
async def test_loop_adaptive_without_verifier_is_noop_regression():
    """adaptive=True with no verifier behaves identically to adaptive=False."""
    agent = Agent(
        name="test",
        model="openai/gpt-4o",
        loop_config=LoopConfig(
            max_iterations=5,
            convergence_fn=lambda r: "done" in r.text,
            adaptive=True,
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
async def test_loop_adaptive_never_exceeds_max_iterations():
    """Monotonically-improving scores that never plateau still stop at the cap."""
    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(
            max_iterations=4,
            verifier=Verifier(VerifierConfig(threshold=0.999)),
            adaptive=True,
            min_improvement=0.02,
            plateau_patience=2,
        ),
    )

    call_count = 0

    async def fake_run_agent(agent, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return _agent_result(agent.name, f"answer{call_count}")

    async def fake_evaluate(query, result, *, default_model):
        # Keeps improving by 0.1 each time -- never plateaus, never converges.
        return VerifierFeedback(score=0.1 * call_count, converged=False)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}])

    assert call_count == 4  # stopped by max_iterations, not by plateau
    assert result.text == "answer4"


@pytest.mark.asyncio
async def test_loop_branching_picks_highest_scored_branch():
    import asyncio

    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(max_iterations=1, verifier=Verifier(), branches=3),
    )

    branch_texts = ["draft A", "draft B", "draft C"]
    branch_scores = {"draft A": 0.5, "draft B": 0.9, "draft C": 0.3}
    call_count = 0

    async def fake_run_agent(agent, messages, **kwargs):
        nonlocal call_count
        text = branch_texts[call_count]
        call_count += 1
        return _agent_result(agent.name, text)

    async def fake_evaluate(query, result, *, default_model):
        return VerifierFeedback(score=branch_scores[result.text], converged=False)

    queue: asyncio.Queue = asyncio.Queue()
    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}], queue=queue)

    assert result.text == "draft B"

    events = []
    while not queue.empty():
        events.append(await queue.get())
    branch_events = [e for e in events if e.event == EventType.BRANCH_SELECTED]
    assert len(branch_events) == 1
    assert branch_events[0].data["winning_index"] == 1
    assert branch_events[0].data["winning_score"] == 0.9


@pytest.mark.asyncio
async def test_loop_branching_no_verifier_uses_apex_synthesis():
    from deepcrew.apex import APEXSynthesizer

    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(max_iterations=1, branches=3, convergence_fn=lambda r: True),
    )

    call_count = 0

    async def fake_run_agent(agent, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        return _agent_result(agent.name, f"draft {call_count}")

    synth_calls: list[list[AgentResult]] = []

    async def fake_synthesize(self, query, results, queue=None, tool_defs=None):
        synth_calls.append(results)
        return _agent_result("apex", "merged answer")

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(APEXSynthesizer, "synthesize", new=fake_synthesize):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}])

    assert result.text == "merged answer"
    assert len(synth_calls) == 1
    assert len(synth_calls[0]) == 3


@pytest.mark.asyncio
async def test_loop_branches_one_is_regression_safe():
    """branches=1 (default) behaves identically to pre-Phase-4 single-path execution."""
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
    assert call_count == 2  # no parallel branch fan-out


@pytest.mark.asyncio
async def test_loop_branching_token_totals_sum_across_branches():
    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        loop_config=LoopConfig(max_iterations=1, verifier=Verifier(), branches=3),
    )

    async def fake_run_agent(agent, messages, **kwargs):
        return AgentResult(agent_id=agent.name, text="draft", input_tokens=10, output_tokens=5)

    async def fake_evaluate(query, result, *, default_model):
        return VerifierFeedback(score=0.9, converged=True)

    with patch("deepcrew.runner.run_agent", new=AsyncMock(side_effect=fake_run_agent)), \
         patch.object(Verifier, "evaluate", new=AsyncMock(side_effect=fake_evaluate)):
        result = await run_agent_loop(agent, [{"role": "user", "content": "go"}])

    assert result.total_tokens == 3 * (10 + 5)
