from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryProvider(ABC):
    """Abstract base for pluggable short-term and long-term memory stores."""

    @abstractmethod
    async def store(self, key: str, value: str) -> None:
        """Store a value under the given key."""

    @abstractmethod
    async def retrieve(self, key: str) -> str | None:
        """Return the value for *key*, or None if not found."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """Return up to *top_k* (key, value) pairs relevant to *query*."""

    @abstractmethod
    async def clear(self) -> None:
        """Remove all stored entries."""
