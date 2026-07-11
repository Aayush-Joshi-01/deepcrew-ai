from __future__ import annotations

import pytest

from deepcrew.memory.inmemory import InMemoryProvider
from deepcrew.procedural_memory import PlaybookEntry, ProceduralMemory
from deepcrew.types import AgentResult
from deepcrew.verifier import VerifierFeedback


@pytest.mark.asyncio
async def test_curate_persists_entries_retrievable_via_load():
    backend = InMemoryProvider()
    pm = ProceduralMemory(backend)

    feedback = VerifierFeedback(
        score=0.9, issues=["forgot to cite sources"], suggestion="", converged=True
    )
    trajectory = [AgentResult(agent_id="a", text="A great, well-researched answer about CRISPR.")]

    await pm.curate("research", feedback, trajectory)
    entries = await pm.load("research")

    assert len(entries) == 2  # one harmful (issue) + one helpful (high-score summary)
    kinds = {e.kind for e in entries}
    assert kinds == {"helpful", "harmful"}


@pytest.mark.asyncio
async def test_curate_twice_with_overlapping_issue_bumps_existing_entry():
    backend = InMemoryProvider()
    pm = ProceduralMemory(backend)

    feedback = VerifierFeedback(
        score=0.5, issues=["missing citations"], suggestion="", converged=False
    )
    trajectory = [AgentResult(agent_id="a", text="answer")]

    await pm.curate("research", feedback, trajectory)
    await pm.curate("research", feedback, trajectory)

    entries = await pm.load("research")
    harmful = [e for e in entries if e.kind == "harmful"]
    assert len(harmful) == 1
    assert harmful[0].uses == 2


@pytest.mark.asyncio
async def test_curate_prunes_to_max_entries():
    backend = InMemoryProvider()
    pm = ProceduralMemory(backend, max_entries=3)

    for i in range(6):
        feedback = VerifierFeedback(score=0.9, issues=[f"issue {i}"], suggestion="", converged=True)
        trajectory = [AgentResult(agent_id="a", text=f"answer {i}")]
        await pm.curate("task", feedback, trajectory)

    entries = await pm.load("task")
    assert len(entries) <= 3


@pytest.mark.asyncio
async def test_render_empty_list_returns_empty_string():
    pm = ProceduralMemory(InMemoryProvider())
    assert pm.render([]) == ""


@pytest.mark.asyncio
async def test_render_nonempty_includes_helpful_and_avoid_labels():
    pm = ProceduralMemory(InMemoryProvider())
    entries = [
        PlaybookEntry(content="cite sources", kind="helpful", uses=2, last_score=0.9),
        PlaybookEntry(content="don't skip verification", kind="harmful", uses=1, last_score=0.4),
    ]
    block = pm.render(entries)
    assert "helpful" in block
    assert "avoid" in block
    assert "cite sources" in block


@pytest.mark.asyncio
async def test_load_returns_empty_list_when_nothing_persisted():
    pm = ProceduralMemory(InMemoryProvider())
    assert await pm.load("unknown_task") == []
