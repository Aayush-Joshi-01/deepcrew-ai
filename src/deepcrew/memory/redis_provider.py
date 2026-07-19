"""
Redis-backed :class:`MemoryProvider`.

Requires the ``redis`` extra::

    pip install deepcrew-ai[redis]

The ``redis`` package is imported lazily inside :class:`RedisMemoryProvider`,
so a bare ``pip install deepcrew-ai`` never pulls it in.
"""

from __future__ import annotations

from typing import Any

from ..exceptions import DeepCrewMemoryError
from .base import MemoryProvider


class RedisMemoryProvider(MemoryProvider):
    """
    Persistent memory store backed by Redis.

    Mirrors :class:`~deepcrew.memory.inmemory.InMemoryProvider`'s ``search``
    semantics exactly: case-insensitive substring match on key or value,
    sorted by key, truncated to ``top_k``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "deepcrew:",
        client: Any | None = None,
    ) -> None:
        self._prefix = prefix
        if client is not None:
            self._client = client
            return

        try:
            import redis.asyncio as redis_asyncio
        except ImportError as exc:
            raise DeepCrewMemoryError(
                "The redis package is not installed. "
                "Install it with: pip install deepcrew-ai[redis]"
            ) from exc

        self._client = redis_asyncio.from_url(url, decode_responses=True)

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def store(self, key: str, value: str) -> None:
        try:
            await self._client.set(self._full_key(key), value)
        except Exception as exc:
            raise DeepCrewMemoryError(f"Redis SET failed for key {key!r}: {exc}") from exc

    async def retrieve(self, key: str) -> str | None:
        try:
            return await self._client.get(self._full_key(key))
        except Exception as exc:
            raise DeepCrewMemoryError(f"Redis GET failed for key {key!r}: {exc}") from exc

    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        try:
            keys: list[str] = []
            async for full_key in self._client.scan_iter(match=f"{self._prefix}*"):
                keys.append(full_key)
            if not keys:
                return []
            values = await self._client.mget(keys)
        except Exception as exc:
            raise DeepCrewMemoryError(f"Redis SCAN/MGET failed: {exc}") from exc

        q = query.lower()
        matches: list[tuple[str, str]] = []
        for full_key, value in zip(keys, values, strict=False):
            if value is None:
                continue
            short_key = full_key[len(self._prefix) :]
            if q in short_key.lower() or q in value.lower():
                matches.append((short_key, value))
        return sorted(matches, key=lambda kv: kv[0])[:top_k]

    async def clear(self) -> None:
        try:
            keys = [full_key async for full_key in self._client.scan_iter(match=f"{self._prefix}*")]
            if keys:
                await self._client.unlink(*keys)
        except Exception as exc:
            raise DeepCrewMemoryError(f"Redis UNLINK failed: {exc}") from exc

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception as exc:
            raise DeepCrewMemoryError(f"Redis connection close failed: {exc}") from exc
