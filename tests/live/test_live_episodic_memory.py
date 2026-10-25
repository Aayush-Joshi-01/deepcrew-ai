"""Episodic memory: the record -> recall cycle against a real model."""

from __future__ import annotations

import pytest

from deepcrew import Agent, Episode, EpisodicMemory, LoopConfig, Verifier, VerifierConfig, run_agent
from deepcrew.types import EventType

pytestmark = pytest.mark.live(
    feature="episodic_memory",
    watch=[
        "src/deepcrew/episodic_memory.py",
        "src/deepcrew/runner.py",
        "src/deepcrew/loop.py",
    ],
)


@pytest.mark.asyncio
async def test_record_then_recall_on_next_run(live_model: str, live_backend):
    em = EpisodicMemory(live_backend, namespace="live", recall_k=2)
    agent = Agent(name="assistant", model=live_model, episodic_memory=em, max_turns=2)

    r1 = await run_agent(
        agent, [{"role": "user", "content": "In one sentence, what is a hash map?"}]
    )
    assert r1.text.strip()
    assert len(await em.load()) == 1

    import asyncio

    queue: asyncio.Queue = asyncio.Queue()
    r2 = await run_agent(
        agent,
        [{"role": "user", "content": "How does a hash map handle collisions?"}],
        queue=queue,
    )
    assert r2.text.strip()
    assert r2.episodes_recalled >= 1
    events = []
    while not queue.empty():
        ev = queue.get_nowait()
        if ev is not None:
            events.append(ev.event)
    assert EventType.EPISODE_RECALLED in events
    assert EventType.EPISODE_RECORDED in events
    assert len(await em.load()) == 2


@pytest.mark.asyncio
async def test_loop_records_per_iteration_with_scores(live_model: str, live_backend):
    em = EpisodicMemory(live_backend, namespace="live-loop")
    agent = Agent(
        name="researcher",
        model=live_model,
        system_prompt="Answer briefly.",
        max_turns=2,
        episodic_memory=em,
        loop_config=LoopConfig(
            max_iterations=2,
            verifier=Verifier(
                VerifierConfig(threshold=0.99, rubric="Must name at least three distinct items.")
            ),
        ),
    )
    await run_agent(agent, [{"role": "user", "content": "Name common sorting algorithms."}])

    episodes = await em.load()
    assert len(episodes) == 2
    assert all(e.score is not None for e in episodes)


@pytest.mark.asyncio
async def test_reflection_populated_when_reflect_model_set(live_model: str, live_backend):
    em = EpisodicMemory(live_backend, namespace="live-reflect", reflect_model=live_model)
    ep = await em.record(
        Episode(query="explain TCP handshake", response="SYN, SYN-ACK, ACK.", outcome="success")
    )
    assert ep.reflection and ep.reflection.strip()
