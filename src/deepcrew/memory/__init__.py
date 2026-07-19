from typing import Any

from .base import MemoryProvider
from .file import FileMemoryProvider
from .inmemory import InMemoryProvider

__all__ = ["FileMemoryProvider", "InMemoryProvider", "MemoryProvider", "RedisMemoryProvider"]


def __getattr__(name: str) -> Any:
    """Lazily expose RedisMemoryProvider so a bare install never imports redis."""
    if name == "RedisMemoryProvider":
        from .redis_provider import RedisMemoryProvider

        return RedisMemoryProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
