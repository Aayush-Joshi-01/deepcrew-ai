# Memory Providers
Memory providers let agents maintain context across turns, runs, and even restarts. They auto-inject into the LLM call (relevant memories added as a system message), and auto-store tool results for future retrieval.

### InMemoryProvider — short-term context

```python
from deepcrew import Agent, run_agent, InMemoryProvider

memory = InMemoryProvider()

agent = Agent(
    name="assistant",
    model="openai/gpt-4o-mini",
    system_prompt="You are a helpful assistant with memory.",
    memory=memory,
)

# First interaction
result1 = await run_agent(agent, [
    {"role": "user", "content": "My name is Alice and I'm building a Python library."}
])

# Memory automatically stores tool results and LLM context
# Second interaction — agent will remember Alice's project
result2 = await run_agent(agent, [
    {"role": "user", "content": "What was my project about again?"}
])
```

### FileMemoryProvider — persistent context

```python
from pathlib import Path
from deepcrew import Agent, run_agent, FileMemoryProvider

# Persists across process restarts
memory = FileMemoryProvider(Path.home() / ".deepcrew" / "my_agent_memory.json")

agent = Agent(
    name="persistent_bot",
    model="openai/gpt-4o-mini",
    memory=memory,
)

# All tool results are atomically written to the JSON file
# On next startup, memories are loaded and injected into context
```

### Custom MemoryProvider

Need something InMemory/File/Redis don't cover — SQLite, a vector database, an external key-value service? Subclass `MemoryProvider` directly. The one detail every implementer gets wrong at least once: `search()` must return `(key, value)` tuples, not just values — every built-in provider's `render`/injection logic depends on having the key available too.

```python
import sqlite3
from deepcrew.memory.base import MemoryProvider

class SQLiteMemoryProvider(MemoryProvider):
    """Simple SQLite-backed memory — one row per key, LIKE-based search."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    async def store(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    async def retrieve(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT key, value FROM memory WHERE key LIKE ? OR value LIKE ? "
            "ORDER BY key LIMIT ?",
            (f"%{query}%", f"%{query}%", top_k),
        ).fetchall()
        return [(k, v) for k, v in rows]

    async def clear(self) -> None:
        self._conn.execute("DELETE FROM memory")
        self._conn.commit()

agent = Agent("persistent", model="openai/gpt-4o", memory=SQLiteMemoryProvider("agent.db"))
```

This example is deliberately synchronous under the hood (plain `sqlite3`, no `await` inside the query calls) — fine for a single-process CLI tool, but it will block the event loop under real concurrent load. For a multi-process or high-concurrency deployment, use [RedisMemoryProvider](redis-memory.html) instead, or wrap blocking calls in `asyncio.to_thread()`.

### Memory events

`run_agent()` is not an async generator — you cannot `async for` over it directly. If you pass a `queue`, events land in it as they occur, but a bare `run_agent()` call does not put a terminating `None` sentinel on that queue when it finishes (only `Orchestrator`/`WorkflowBuilder`'s own internal queues self-terminate that way). The straightforward pattern is to drain whatever's already in the queue once the call returns:

```python
import asyncio
from deepcrew import Agent, run_agent
from deepcrew.types import EventType

queue: asyncio.Queue = asyncio.Queue()
result = await run_agent(agent, messages, queue=queue)

while not queue.empty():
    event = await queue.get()
    if event.event == EventType.MEMORY_RETRIEVE:
        print(f"Injected {event.data['count']} memories into context")
    elif event.event == EventType.MEMORY_STORE:
        print(f"Stored tool result to memory: {event.data['key']}")
```

### MemoryProvider ABC

```python
class MemoryProvider(ABC):
    @abstractmethod
    async def store(self, key: str, value: str) -> None: ...
    @abstractmethod
    async def retrieve(self, key: str) -> str | None: ...
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]: ...
    @abstractmethod
    async def clear(self) -> None: ...
```

All three built-in providers (`InMemoryProvider`, `FileMemoryProvider`, `RedisMemoryProvider`) implement `search()` identically: a case-insensitive substring match on either the key or the value, sorted by key, truncated to `top_k`. This is not semantic search — a query like `"CRISPR gene editing"` will not match a stored value about `"Cas9 mechanism"` unless the substring itself literally overlaps. If you need embedding-based retrieval, that's a natural fit for a custom `MemoryProvider` like the SQLite example above, backed by a vector index instead of a LIKE query.

### Procedural memory (evolving playbook)

`ProceduralMemory` is an opt-in, durable "the system learns from its own past runs" store, built on top of any `MemoryProvider`. See the full [Procedural Memory feature guide](procedural-memory.html) for a showcase of usage patterns, from a single agent that gets smarter over time to sharing one playbook across a whole agent pool.

### Common pitfalls

    - `run_agent()` is not an async generator. Pass a `queue` and drain it manually — see "Memory events" above. Only `Orchestrator.stream()` and `WorkflowBuilder.stream()` return true async generators.
    - Search is substring matching, not semantic search. All three built-in providers share the same naive case-insensitive substring algorithm — don't expect it to find conceptually related but textually different content.
    - `FileMemoryProvider` reads the file lazily and caches it in memory. If another process modifies the same JSON file concurrently, your provider instance won't see those changes until it's recreated — there's no file-watching or cross-process invalidation.
    - Memory injection only looks at the last 3 messages. `run_agent()` builds its search query from the text of the final 3 messages in the conversation you pass it, not the full history — very early context won't influence what gets retrieved.

### See also

    - [Redis Memory Provider](redis-memory.html) — a persistent, shared, multi-process-safe option built into the library.
    - [Procedural Memory](procedural-memory.html) — a structured playbook layered on top of any provider here.
