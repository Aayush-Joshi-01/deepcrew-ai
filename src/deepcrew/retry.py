from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from .types import EventType, StreamEvent

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class RetryPolicy:
    """Per-agent retry configuration for transient LLM failures."""

    max_retries: int = 3
    backoff_seconds: float = 1.0
    retry_on: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))
    exponential: bool = True


@dataclass
class FallbackChain:
    """Ordered list of model strings to try when all retries for the current model fail."""

    models: list[str]


async def with_retry_and_fallback(
    coro_factory: Callable[[str], Coroutine[Any, Any, Any]],
    agent: "Agent",
    queue: asyncio.Queue[StreamEvent | None] | None,
    agent_id: str,
) -> Any:
    """
    Wrap an LLM call factory with retry and model-fallback logic.

    ``coro_factory`` receives the model string and must return a fresh coroutine
    each call (coroutines cannot be restarted).
    """
    models_to_try: list[str] = [agent.model]
    if agent.fallback_chain:
        models_to_try.extend(agent.fallback_chain.models)

    last_exc: Exception | None = None

    for model_idx, model in enumerate(models_to_try):
        if model_idx > 0 and queue:
            await queue.put(StreamEvent(
                EventType.FALLBACK_TRIGGERED,
                {"from_model": models_to_try[model_idx - 1], "to_model": model},
                agent_id,
            ))

        policy = agent.retry_policy
        max_attempts = (policy.max_retries + 1) if policy else 1

        for attempt in range(max_attempts):
            if attempt > 0:
                retry_on = policy.retry_on if policy else (Exception,)
                if not isinstance(last_exc, tuple(retry_on)):
                    break
                delay = policy.backoff_seconds * (2 ** (attempt - 1) if policy.exponential else 1)
                if queue:
                    await queue.put(StreamEvent(
                        EventType.RETRY_ATTEMPT,
                        {"attempt": attempt, "model": model, "delay": delay},
                        agent_id,
                    ))
                await asyncio.sleep(delay)

            try:
                return await coro_factory(model)
            except Exception as exc:
                last_exc = exc
                retry_on = policy.retry_on if policy else (Exception,)
                if not isinstance(exc, tuple(retry_on)):
                    raise
                continue

    assert last_exc is not None
    raise last_exc
