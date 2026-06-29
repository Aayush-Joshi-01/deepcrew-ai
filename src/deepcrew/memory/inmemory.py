from __future__ import annotations

from .base import MemoryProvider


class InMemoryProvider(MemoryProvider):
    """Simple in-process dict-based memory store (short-term, non-persistent)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def store(self, key: str, value: str) -> None:
        self._store[key] = value

    async def retrieve(self, key: str) -> str | None:
        return self._store.get(key)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        q = query.lower()
        matches = [
            (k, v) for k, v in self._store.items()
            if q in k.lower() or q in v.lower()
        ]
        return sorted(matches, key=lambda kv: kv[0])[:top_k]

    async def clear(self) -> None:
        self._store.clear()
