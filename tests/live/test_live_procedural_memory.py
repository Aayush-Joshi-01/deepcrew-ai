"""Procedural memory: the create (curate) -> utilize (inject) cycle, live."""

from __future__ import annotations

import asyncio

import pytest

from deepcrew import (
    Agent,
    LoopConfig,
    ProceduralMemory,
    Verifier,
    VerifierConfig,
    run_agent,
)
from deepcrew.types import EventType

pytestmark = pytest.mark.live(
    feature="procedural_memory",
    watch=[
        "src/deepcrew/procedural_memory.py",
        "src/deepcrew/loop.py",
        "src/deepcrew/runner.py",
    ],
)


@pytest.mark.asyncio
async def test_curate_then_inject_cycle(live_model: str, live_backend):
    pm = ProceduralMemory(live_backend)
    tag = "essay-writing"

    # --- create: a loop that can't one-shot the rubric curates the playbook
    writer = Agent(
        name="writer",
        model=live_model,
        system_prompt="Write a short paragraph.",
        max_turns=2,
        loop_config=LoopConfig(
            max_iterations=2,
            task_tag=tag,
            procedural_memory=pm,
            verifier=Verifier(
                VerifierConfig(
                    threshold=0.99,
                    rubric="Must contain a concrete statistic with a cited source.",
                )
            ),
        ),
    )
    queue: asyncio.Queue = asyncio.Queue()
    await run_agent(
        writer,
        [{"role": "user", "content": "Write about the benefits of cycling to work."}],
        queue=queue,
    )
    events = []
    while not queue.empty():
        ev = queue.get_nowait()
        if ev is not None:
            events.append(ev.event)
    assert EventType.PLAYBOOK_UPDATED in events

    entries = await pm.load(tag)
    assert entries, "curate should have persisted at least one strategy"

    # --- utilize: a fresh single-shot agent reads the same playbook
    rendered: list[str] = []
    real_render = pm.render

    def _spy_render(es):
        block = real_render(es)
        rendered.append(block)
        return block

    pm.render = _spy_render  # type: ignore[method-assign]

    reader = Agent(
        name="writer",
        model=live_model,
        system_prompt="Write a short paragraph.",
        max_turns=2,
        procedural_memory=pm,
        procedural_task_tag=tag,
    )
    result = await run_agent(
        reader, [{"role": "user", "content": "Write about the benefits of walking."}]
    )
    assert result.text.strip()
    assert rendered and "Known strategies for this task" in rendered[-1]


@pytest.mark.asyncio
async def test_reflect_model_curation(live_model: str, live_backend):
    pm = ProceduralMemory(live_backend, reflect_model=live_model)
    agent = Agent(
        name="analyst",
        model=live_model,
        system_prompt="Answer concisely.",
        max_turns=2,
        loop_config=LoopConfig(
            max_iterations=2,
            task_tag="analysis",
            procedural_memory=pm,
            verifier=Verifier(VerifierConfig(threshold=0.99)),
        ),
    )
    await run_agent(agent, [{"role": "user", "content": "Summarize the causes of inflation."}])
    assert await pm.load("analysis")
