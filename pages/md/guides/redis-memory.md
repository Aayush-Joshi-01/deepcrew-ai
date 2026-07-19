# Redis Memory Provider
`RedisMemoryProvider` implements the same `MemoryProvider` interface as `InMemoryProvider`/`FileMemoryProvider`, backed by Redis for persistence across processes. Requires the `redis` extra: `pip install deepcrew-ai[redis]`.

```python
from deepcrew import Agent, RedisMemoryProvider

memory = RedisMemoryProvider(url="redis://localhost:6379/0", prefix="myapp:")
agent = Agent(name="assistant", model="openai/gpt-4o", memory=memory)

# ... run agents ...

await memory.aclose()   # close the underlying redis connection when you're done
```

`redis` is imported lazily, inside `RedisMemoryProvider.__init__` — not at module import time — so a bare `pip install deepcrew-ai` never pulls it in, and instantiating this class without the extra installed raises a clear `DeepCrewMemoryError` instead of a raw `ModuleNotFoundError`:

```python
from deepcrew import RedisMemoryProvider, DeepCrewMemoryError

try:
    memory = RedisMemoryProvider(url="redis://localhost:6379/0")
except DeepCrewMemoryError as exc:
    print(exc)  # "The redis package is not installed. Install it with: pip install deepcrew-ai[redis]"
```

### Constructor reference

- **url** (str = "redis://localhost:6379/0"): Passed straight to `redis.asyncio.from_url(url, decode_responses=True)`. Any URL `redis-py` accepts works, including auth (`redis://:password@host:port/db`) and TLS (`rediss://...`) schemes.

- **prefix** (str = "deepcrew:"): Every key is namespaced under this prefix in Redis, so multiple applications (or multiple agents with different logical stores) can safely share one Redis instance without key collisions. Stripped back off automatically before keys are returned from `search()`.

- **client** (Any | None = None): Pass an already-constructed `redis.asyncio` client to reuse an existing connection pool instead of creating a new one — useful when your application already manages a shared Redis client elsewhere. When set, `url` is ignored entirely and the lazy `redis` import never happens (this is also how the test suite injects an `AsyncMock()` in place of a real connection).

### How search works

Search semantics mirror `InMemoryProvider` exactly: `SCAN` for every key under the configured prefix, `MGET` their values in one round trip, then a case-insensitive substring match on key or value, sorted by key, truncated to `top_k`. This means a search over a very large keyspace still has to scan and fetch everything under the prefix — there's no Redis-native indexing or scoring involved, so performance scales with total entry count, not with how selective your query is.

### Error handling

Every method wraps its Redis call in a broad exception handler and re-raises as `DeepCrewMemoryError` with a descriptive message — a dropped connection, a timeout, or an auth failure all surface the same way, distinguishable from a normal `DeepCrewMemoryError` only by the message text, not a different exception subtype.

```python
from deepcrew import DeepCrewMemoryError

try:
    await memory.store("last_query", "...")
except DeepCrewMemoryError as exc:
    print(f"Redis memory write failed: {exc}")
    # fall back to running without memory for this turn, log, alert, etc.
```

### See also

    - [Memory Providers](memory.html) — the shared `MemoryProvider` interface and the built-in in-process/file alternatives.
    - [Procedural Memory](procedural-memory.html) — can be layered on top of a `RedisMemoryProvider` for a shared, persistent playbook across an agent pool.
