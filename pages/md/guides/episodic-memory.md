# Episodic Memory

`EpisodicMemory` is a durable log of specific past runs — what was asked, what the agent
answered, which tools it used, and how well it went — with similarity + recency recall of
relevant prior experience before a new run.

It is the third memory concept in deepcrew-ai, and the specifics-keeping one:

| Type | API | Keeps |
|---|---|---|
| Semantic | `Agent.memory` (a `MemoryProvider`) | raw key/value facts (tool results) |
| Procedural | `ProceduralMemory` | distilled *how-to* rules (helpful / harmful bullets) |
| **Episodic** | **`EpisodicMemory`** | **the actual past runs** (query, answer, outcome, score) |

Like `ProceduralMemory`, it layers on any `MemoryProvider` as a ring buffer of JSON episodes
(`episodic:{namespace}:{i}` + a `__meta__` key).

### The automatic cycle

Set `Agent.episodic_memory` and both halves happen on their own:

```python
from deepcrew import Agent, EpisodicMemory, run_agent
from deepcrew.memory import FileMemoryProvider

episodic = EpisodicMemory(FileMemoryProvider("episodes.json"), namespace="support-bot")

agent = Agent(
    name="assistant",
    model="gemini/gemini-3.5-flash-lite",
    episodic_memory=episodic,
)

# run 1 — nothing to recall; the run is recorded afterwards
await run_agent(agent, [{"role": "user", "content": "How do I rotate an API key?"}])

# run 2 — the earlier episode is recalled into context, then this run is recorded too
r = await run_agent(agent, [{"role": "user", "content": "Is key rotation zero-downtime?"}])
print(r.episodes_recalled)   # >= 1
```

- **Recall** happens in the runner before the LLM sees anything: relevant episodes are
  ranked and injected as a `## Relevant past experience` system block. `AgentResult.episodes_recalled` reports how many.
- **Record** happens after the run. In a refinement loop, one episode is recorded **per
  iteration**, with its `outcome` / `score` taken from that iteration's verifier feedback.

### `Episode`

```python
@dataclass
class Episode:
    query: str
    response: str
    outcome: Literal["success", "failure", "partial", "unknown"] = "unknown"
    score: float | None = None
    tools_used: list[str] = field(default_factory=list)
    reflection: str | None = None      # set when reflect_model is configured
    tags: list[str] = field(default_factory=list)
    timestamp: str = ""                # ISO-8601 UTC, auto-filled
    agent_id: str = ""
```

### `EpisodicMemory`

```python
EpisodicMemory(
    backend,                       # any MemoryProvider
    namespace="default",           # isolates one log from another
    max_episodes=100,              # ring buffer: oldest overwritten once full
    recall_k=3,                    # default number returned by recall()
    reflect_model=None,            # optional: LiteLLM model for a one-line lesson per episode
)
```

- `await record(episode) -> Episode` — persist (fills `timestamp`, and `reflection` if
  `reflect_model` is set).
- `await recall(query, *, top_k=None) -> list[Episode]` — ranked by keyword (Jaccard) overlap
  between the query and each episode's query/tags, plus a small recency tie-breaker. Episodes
  with no topical overlap are not returned. Dependency-free — no embeddings.
- `render(episodes) -> str` — the system-prompt block.
- `await reflect(episode) -> str | None` — one LLM call distilling a reusable lesson;
  `None` without `reflect_model`.
- `await clear()` — drop this namespace.

### Optional LLM reflection

```python
episodic = EpisodicMemory(backend, reflect_model="gemini/gemini-3.5-flash-lite")
```

Each recorded episode gets a one-sentence `reflection` ("Always confirm the customer's plan
tier before quoting limits."). `render()` prefers the reflection over the raw response, so
recall stays compact. Off by default — no extra call unless you opt in. Reflection failures
are swallowed (the episode is still recorded, `reflection=None`).

### Events

`EventType.EPISODE_RECALLED` (`{count}`) and `EPISODE_RECORDED` (`{outcome, score}`) are
emitted on the stream. Like `MEMORY_STORE` / `MEMORY_RETRIEVE`, they are **not** in the
`chat()` or `standard()` `StreamPolicy` presets — use `verbose()` to see them.

### Not OpenTelemetry, not the event stream

Episodic memory is plain state in your `MemoryProvider`. It is independent of
`ObservabilityConfig` (spans) and of `StreamPolicy` filtering.

### Common pitfalls

- **Recall needs topical overlap.** A totally unrelated new query recalls nothing — that is
  intentional. Widen recall with `tags` on your episodes.
- **The ring buffer overwrites.** Past `max_episodes`, the oldest episode is gone. Raise
  `max_episodes` or use a persistent backend and prune deliberately.
- **`reflect_model` costs one call per record.** In a 5-iteration loop that is 5 calls. Leave
  it off unless the lesson quality is worth it.
- **Namespaces are manual.** Two agents sharing a backend but not a namespace keep separate
  logs; give them the same `namespace` to share experience.
