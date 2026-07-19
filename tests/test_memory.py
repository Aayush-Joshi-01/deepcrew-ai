from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

from deepcrew.exceptions import DeepCrewMemoryError
from deepcrew.memory.file import FileMemoryProvider
from deepcrew.memory.inmemory import InMemoryProvider
from deepcrew.memory.redis_provider import RedisMemoryProvider


class _FakeRedisClient:
    """Minimal in-process stand-in for redis.asyncio.Redis, dict-backed."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def set(self, key: str, value: str) -> None:
        self.data[key] = value

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def scan_iter(self, match: str = "*"):
        prefix = match[:-1] if match.endswith("*") else match
        for k in list(self.data):
            if k.startswith(prefix):
                yield k

    async def mget(self, keys: list[str]) -> list[str | None]:
        return [self.data.get(k) for k in keys]

    async def unlink(self, *keys: str) -> None:
        for k in keys:
            self.data.pop(k, None)

    async def aclose(self) -> None:
        pass


# --- InMemoryProvider full contract -----------------------------------------


@pytest.mark.asyncio
async def test_inmemory_store_and_retrieve():
    mem = InMemoryProvider()
    await mem.store("a", "apple")
    assert await mem.retrieve("a") == "apple"
    assert await mem.retrieve("missing") is None


@pytest.mark.asyncio
async def test_inmemory_search_substring_and_top_k():
    mem = InMemoryProvider()
    await mem.store("fruit_a", "apple")
    await mem.store("fruit_b", "banana")
    await mem.store("veg_c", "carrot")

    results = await mem.search("fruit")
    assert [k for k, _ in results] == ["fruit_a", "fruit_b"]

    results_top1 = await mem.search("fruit", top_k=1)
    assert len(results_top1) == 1


@pytest.mark.asyncio
async def test_inmemory_search_matches_value_too():
    mem = InMemoryProvider()
    await mem.store("k1", "contains needle here")
    results = await mem.search("needle")
    assert results == [("k1", "contains needle here")]


@pytest.mark.asyncio
async def test_inmemory_clear():
    mem = InMemoryProvider()
    await mem.store("a", "1")
    await mem.clear()
    assert await mem.retrieve("a") is None


# --- FileMemoryProvider round-trip -------------------------------------------


@pytest.mark.asyncio
async def test_file_memory_round_trip(tmp_path):
    path = tmp_path / "mem.json"
    mem = FileMemoryProvider(path)
    await mem.store("a", "apple")
    assert path.exists()

    mem2 = FileMemoryProvider(path)
    assert await mem2.retrieve("a") == "apple"


@pytest.mark.asyncio
async def test_file_memory_search_and_clear(tmp_path):
    path = tmp_path / "mem.json"
    mem = FileMemoryProvider(path)
    await mem.store("fruit_a", "apple")
    await mem.store("fruit_b", "banana")

    results = await mem.search("fruit")
    assert [k for k, _ in results] == ["fruit_a", "fruit_b"]

    await mem.clear()
    assert await mem.retrieve("fruit_a") is None


# --- RedisMemoryProvider with an injected fake client ------------------------


@pytest.mark.asyncio
async def test_redis_provider_store_and_retrieve_with_prefix():
    client = _FakeRedisClient()
    mem = RedisMemoryProvider(prefix="deepcrew:", client=client)

    await mem.store("a", "apple")
    assert client.data == {"deepcrew:a": "apple"}
    assert await mem.retrieve("a") == "apple"


@pytest.mark.asyncio
async def test_redis_provider_search_strips_prefix_and_sorts():
    client = _FakeRedisClient()
    mem = RedisMemoryProvider(prefix="deepcrew:", client=client)

    await mem.store("fruit_b", "banana")
    await mem.store("fruit_a", "apple")
    await mem.store("veg_c", "carrot")

    results = await mem.search("fruit")
    assert results == [("fruit_a", "apple"), ("fruit_b", "banana")]


@pytest.mark.asyncio
async def test_redis_provider_search_top_k():
    client = _FakeRedisClient()
    mem = RedisMemoryProvider(prefix="deepcrew:", client=client)
    for i in range(5):
        await mem.store(f"k{i}", f"value {i}")

    results = await mem.search("value", top_k=2)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_redis_provider_clear():
    client = _FakeRedisClient()
    mem = RedisMemoryProvider(prefix="deepcrew:", client=client)
    await mem.store("a", "1")
    await mem.clear()
    assert client.data == {}


@pytest.mark.asyncio
async def test_redis_provider_does_not_touch_keys_outside_prefix():
    client = _FakeRedisClient()
    client.data["other:key"] = "untouched"
    mem = RedisMemoryProvider(prefix="deepcrew:", client=client)

    await mem.store("a", "1")
    results = await mem.search("untouched")
    assert results == []


@pytest.mark.asyncio
async def test_redis_provider_wraps_client_errors_in_deepcrew_memory_error():
    client = AsyncMock()
    client.set = AsyncMock(side_effect=RuntimeError("connection refused"))

    mem = RedisMemoryProvider(client=client)
    with pytest.raises(DeepCrewMemoryError, match="connection refused"):
        await mem.store("a", "1")


@pytest.mark.asyncio
async def test_redis_provider_wraps_get_errors():
    client = AsyncMock()
    client.get = AsyncMock(side_effect=RuntimeError("boom"))

    mem = RedisMemoryProvider(client=client)
    with pytest.raises(DeepCrewMemoryError):
        await mem.retrieve("a")


def test_redis_provider_missing_package_raises_deepcrew_memory_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setitem(sys.modules, "redis.asyncio", None)
    with pytest.raises(DeepCrewMemoryError, match=r"pip install deepcrew-ai\[redis\]"):
        RedisMemoryProvider()
