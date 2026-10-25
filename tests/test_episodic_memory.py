from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.episodic_memory import Episode, EpisodicMemory
from deepcrew.memory.inmemory import InMemoryProvider


@pytest.mark.asyncio
async def test_record_then_load_roundtrip():
    em = EpisodicMemory(InMemoryProvider(), namespace="t")
    await em.record(Episode(query="how to cache in python", response="use functools.lru_cache"))
    episodes = await em.load()
    assert len(episodes) == 1
    assert episodes[0].query == "how to cache in python"
    assert episodes[0].timestamp  # auto-filled


@pytest.mark.asyncio
async def test_ring_buffer_prunes_to_max_and_keeps_newest():
    em = EpisodicMemory(InMemoryProvider(), namespace="t", max_episodes=3)
    for i in range(6):
        await em.record(Episode(query=f"q{i}", response=f"a{i}"))
    episodes = await em.load()
    assert len(episodes) == 3
    assert [e.query for e in episodes] == ["q3", "q4", "q5"]  # oldest-first, newest kept


@pytest.mark.asyncio
async def test_recall_ranks_by_keyword_overlap():
    em = EpisodicMemory(InMemoryProvider(), namespace="t", recall_k=1)
    await em.record(Episode(query="baking sourdough bread at home", response="..."))
    await em.record(Episode(query="python asyncio event loop internals", response="..."))
    hits = await em.recall("how does the asyncio event loop schedule tasks")
    assert len(hits) == 1
    assert "asyncio" in hits[0].query


@pytest.mark.asyncio
async def test_recall_respects_top_k_and_drops_zero_score():
    em = EpisodicMemory(InMemoryProvider(), namespace="t")
    await em.record(Episode(query="completely unrelated topic apples", response="..."))
    hits = await em.recall("quantum chromodynamics lattice", top_k=5)
    assert hits == []


@pytest.mark.asyncio
async def test_render_empty_and_nonempty():
    em = EpisodicMemory(InMemoryProvider())
    assert em.render([]) == ""
    block = em.render([Episode(query="Q", response="an answer", outcome="success", score=0.91)])
    assert "Relevant past experience" in block
    assert "success" in block


@pytest.mark.asyncio
async def test_clear():
    em = EpisodicMemory(InMemoryProvider(), namespace="t")
    await em.record(Episode(query="q", response="a"))
    await em.clear()
    assert await em.load() == []


@pytest.mark.asyncio
async def test_reflect_populates_reflection_when_model_set():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "Always cite a concrete source."
    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        em = EpisodicMemory(InMemoryProvider(), reflect_model="gemini/x")
        ep = await em.record(Episode(query="q", response="a", outcome="success", score=0.9))
    assert ep.reflection == "Always cite a concrete source."


@pytest.mark.asyncio
async def test_reflect_failure_is_swallowed():
    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("boom"))):
        em = EpisodicMemory(InMemoryProvider(), reflect_model="gemini/x")
        ep = await em.record(Episode(query="q", response="a"))
    assert ep.reflection is None
