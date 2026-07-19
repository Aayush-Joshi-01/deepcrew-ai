from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from deepcrew.agent import Agent
from deepcrew.retry import FallbackChain, RetryPolicy, with_retry_and_fallback
from deepcrew.types import EventType


@pytest.mark.asyncio
async def test_retry_then_succeed():
    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0.01),
    )
    call_count = 0

    async def factory(model: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")
        return "ok"

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await with_retry_and_fallback(factory, agent, None, "a")

    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_last_error():
    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0.01),
    )

    async def factory(model: str) -> str:
        raise RuntimeError("always fails")

    with (
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError, match="always fails"),
    ):
        await with_retry_and_fallback(factory, agent, None, "a")


@pytest.mark.asyncio
async def test_retry_only_retries_configured_exception_types():
    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        retry_policy=RetryPolicy(max_retries=3, backoff_seconds=0.01, retry_on=(ValueError,)),
    )

    async def factory(model: str) -> str:
        raise RuntimeError("not retryable")

    with (
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError, match="not retryable"),
    ):
        await with_retry_and_fallback(factory, agent, None, "a")


@pytest.mark.asyncio
async def test_fallback_chain_switches_model():
    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        fallback_chain=FallbackChain(models=["openai/gpt-4o-mini"]),
    )
    seen_models: list[str] = []

    async def factory(model: str) -> str:
        seen_models.append(model)
        if model == "openai/gpt-4o":
            raise RuntimeError("primary down")
        return f"ok from {model}"

    with patch("asyncio.sleep", new=AsyncMock()):
        result = await with_retry_and_fallback(factory, agent, None, "a")

    assert result == "ok from openai/gpt-4o-mini"
    assert seen_models == ["openai/gpt-4o", "openai/gpt-4o-mini"]


@pytest.mark.asyncio
async def test_retry_and_fallback_events_emitted():
    agent = Agent(
        name="a",
        model="openai/gpt-4o",
        retry_policy=RetryPolicy(max_retries=1, backoff_seconds=0.01),
        fallback_chain=FallbackChain(models=["openai/gpt-4o-mini"]),
    )

    async def factory(model: str) -> str:
        if model == "openai/gpt-4o":
            raise RuntimeError("down")
        return "ok"

    queue: asyncio.Queue = asyncio.Queue()
    with patch("asyncio.sleep", new=AsyncMock()):
        result = await with_retry_and_fallback(factory, agent, queue, "a")

    assert result == "ok"

    events = []
    while not queue.empty():
        events.append(await queue.get())

    types = [e.event for e in events]
    assert EventType.RETRY_ATTEMPT in types
    assert EventType.FALLBACK_TRIGGERED in types


@pytest.mark.asyncio
async def test_no_retry_policy_and_no_fallback_raises_immediately():
    agent = Agent(name="a", model="openai/gpt-4o")

    async def factory(model: str) -> str:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await with_retry_and_fallback(factory, agent, None, "a")
