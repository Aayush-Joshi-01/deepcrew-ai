from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.episodic_memory import Episode, EpisodicMemory
from deepcrew.memory.inmemory import InMemoryProvider
from deepcrew.runner import run_agent
from deepcrew.types import EventType


async def _drain(queue: asyncio.Queue) -> list:
    events = []
    while not queue.empty():
        ev = queue.get_nowait()
        if ev is not None:
            events.append(ev)
    return events


@pytest.mark.asyncio
async def test_run_agent_records_episode_and_sets_count(make_stream, make_stream_chunk):
    em = EpisodicMemory(InMemoryProvider(), namespace="wiring")
    agent = Agent(name="a", model="openai/gpt-4o", episodic_memory=em)

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=make_stream(make_stream_chunk("hello world"))),
    ):
        result = await run_agent(agent, [{"role": "user", "content": "say hi"}])

    stored = await em.load()
    assert len(stored) == 1
    assert stored[0].query == "say hi"
    assert stored[0].response == "hello world"
    assert result.episodes_recalled == 0


@pytest.mark.asyncio
async def test_recall_injects_system_block_and_emits_event(make_stream, make_stream_chunk):
    backend = InMemoryProvider()
    em = EpisodicMemory(backend, namespace="wiring")
    await em.record(Episode(query="how to reverse a linked list", response="iterate and flip next"))

    agent = Agent(name="a", model="openai/gpt-4o", episodic_memory=em)
    captured: dict = {}

    async def fake_completion(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return make_stream(make_stream_chunk("done"))

    queue: asyncio.Queue = asyncio.Queue()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await run_agent(
            agent,
            [{"role": "user", "content": "reversing a linked list in place"}],
            queue=queue,
        )

    system_text = "\n".join(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert "Relevant past experience" in system_text
    assert "reverse a linked list" in system_text
    assert result.episodes_recalled == 1
    types = {e.event for e in await _drain(queue)}
    assert EventType.EPISODE_RECALLED in types
    assert EventType.EPISODE_RECORDED in types


@pytest.mark.asyncio
async def test_procedural_task_tag_read_by_runner(make_stream, make_stream_chunk):
    from deepcrew.procedural_memory import ProceduralMemory
    from deepcrew.types import AgentResult
    from deepcrew.verifier import VerifierFeedback

    backend = InMemoryProvider()
    pm = ProceduralMemory(backend)
    await pm.curate(
        "shared-tag",
        VerifierFeedback(score=0.95, issues=["cite sources"], suggestion="", converged=True),
        [AgentResult(agent_id="x", text="a solid answer")],
    )

    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        procedural_memory=pm,
        procedural_task_tag="shared-tag",
    )
    captured: dict = {}

    async def fake_completion(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        return make_stream(make_stream_chunk("done"))

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        await run_agent(agent, [{"role": "user", "content": "do the task"}])

    system_text = "\n".join(m["content"] for m in captured["messages"] if m["role"] == "system")
    assert "Known strategies for this task" in system_text


@pytest.mark.asyncio
async def test_loop_records_one_episode_per_iteration(make_stream, make_stream_chunk):
    from deepcrew import LoopConfig, Verifier, VerifierConfig

    em = EpisodicMemory(InMemoryProvider(), namespace="loop")
    scores = iter([0.4, 0.95])

    async def fake_completion(*args, **kwargs):
        if kwargs.get("response_format", {}).get("type") == "json_object":
            resp = AsyncMock()
            msg = type("M", (), {"content": f'{{"score": {next(scores)}, "issues": []}}'})()
            resp.choices = [type("C", (), {"message": msg})()]
            return resp
        return make_stream(make_stream_chunk("an answer"))

    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        episodic_memory=em,
        loop_config=LoopConfig(max_iterations=2, verifier=Verifier(VerifierConfig(threshold=0.9))),
    )
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        await run_agent(agent, [{"role": "user", "content": "research topic X"}])

    stored = await em.load()
    assert len(stored) == 2
    assert stored[0].outcome == "partial"
    assert stored[1].outcome == "success"
