# Skills
Skills are higher-level capability bundles. They look identical to tools from the LLM's perspective (both become `ToolDef`), but can wrap multi-step logic, sub-agents, or external APIs internally. Three built-ins are included; you can also create custom skills with the `@skill` decorator.

### Built-in skills

```python
from deepcrew import Agent, run_agent
from deepcrew import WebSearchSkill, SummarizeSkill, CodeExecutionSkill

agent = Agent(
    name="assistant",
    model="openai/gpt-4o",
    system_prompt="You are a versatile AI assistant.",
    skills=[
        WebSearchSkill(),                            # DuckDuckGo Instant Answer API
        SummarizeSkill(model="openai/gpt-4o-mini"),  # LLM-backed summarization
        CodeExecutionSkill(timeout=15.0),            # sandboxed Python subprocess
    ],
)

result = await run_agent(agent, [
    {"role": "user", "content": "Search for Python async best practices, summarize them, then write a demo script and run it."}
])
print(result.text)
```

### Skill ABC — what every skill needs

Every skill, however you build it, must provide four things: a `name` (the tool name the LLM sees), a `description`, a `parameters` JSON Schema object, and an async `execute(self, **kwargs) -> str` method. `to_tool_def()` is provided by the base class and wraps `execute()` into a `ToolDef` — you never need to override it.

- **name** (str): Must be unique among an agent's tools/skills — if a skill and a plain `@tool` function share a name, whichever ends up later in the merged tool list from `Agent.get_tool_defs()` wins.

- **description** (str): Shown to the LLM verbatim. Also what `ToolAllocator` reads when deciding whether a spawned sub-agent should get this skill — see [Agent Spawning](spawning.html).

- **parameters** (dict): A JSON Schema object (`{"type": "object", "properties": {...}, "required": [...]}`) — exactly the shape OpenAI-style function-calling expects.

- **execute(**kwargs)** (async -> str): Receives the LLM's parsed tool-call arguments as keyword arguments. Must return a string — if your logic naturally produces a dict/list, serialize it yourself (e.g. `str(result)` or `json.dumps(result)`).

### @skill decorator — custom skills

The decorator inspects your function's signature to auto-generate the JSON Schema for you, so you don't hand-write `parameters`. The mapping is intentionally simple: parameters annotated `int` become JSON `"integer"`, `float` becomes `"number"`, `bool` becomes `"boolean"`, and everything else (including `str`, no annotation, or a complex type) becomes `"string"` — there is no support for nested objects, lists, unions, or enums via the decorator's auto-generated schema. A parameter is marked `"required"` whenever it has no default value in the function signature; anything with a default is treated as optional.

```python
from deepcrew import skill, Agent

@skill(name="translate", description="Translate text to another language")
async def translate(text: str, target_language: str) -> str:
    """
    Args:
        text (str): The text to translate.
        target_language (str): Target language code, e.g. 'es', 'fr', 'ja'.
    """
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://libretranslate.de/translate",
            json={"q": text, "source": "auto", "target": target_language},
        )
        return r.json()["translatedText"]

@skill(name="send_slack", description="Send a message to a Slack channel")
async def send_slack(channel: str, message: str) -> str:
    """
    Args:
        channel (str): Slack channel name (without #).
        message (str): Message text to send.
    """
    import os
    import httpx
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {os.environ['SLACK_BOT_TOKEN']}"},
            json={"channel": channel, "text": message},
        )
    return f"Message sent to #{channel}"

agent = Agent(
    "comms",
    model="openai/gpt-4o",
    system_prompt="Help with communications and translations.",
    skills=[translate, send_slack],
)
```

### Skill class — full custom implementation

```python
from deepcrew.skills.base import Skill

class DatabaseQuerySkill(Skill):
    name = "database_query"
    description = "Execute a read-only SQL query against the production database"
    parameters = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "SQL SELECT query to execute"},
            "limit": {"type": "integer", "description": "Max rows to return", "default": 100},
        },
        "required": ["sql"],
    }

    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    async def execute(self, sql: str, limit: int = 100) -> str:
        import asyncpg
        conn = await asyncpg.connect(self._conn_str)
        try:
            rows = await conn.fetch(f"{sql} LIMIT {limit}")
            return str([dict(r) for r in rows])
        finally:
            await conn.close()

# Use it:
agent = Agent("db_agent", model="openai/gpt-4o",
              skills=[DatabaseQuerySkill("postgresql://...")])
```

### SkillRegistry

```python
from deepcrew import SkillRegistry
from deepcrew import WebSearchSkill, SummarizeSkill

# Register globally
SkillRegistry.register(WebSearchSkill())
SkillRegistry.register(SummarizeSkill())

# Retrieve by name
search = SkillRegistry.get("web_search")
summarize = SkillRegistry.get("summarize")

# List all registered skills
for skill in SkillRegistry.list_all():
    print(f"{skill.name}: {skill.description}")
```

### Built-in skill reference

      | Class | Tool name | Description | Config |

        | `WebSearchSkill` | `web_search` | DuckDuckGo Instant Answer API. Returns top results. | None |

        | `SummarizeSkill` | `summarize` | LLM-backed text summarization. | `model="openai/gpt-4o-mini"` |

        | `CodeExecutionSkill` | `code_exec` | Runs Python in an isolated subprocess. | `timeout=10.0` |

### How-to: self-evolving skills (auto_extract_skill) v0.2.5

With `LoopConfig.auto_extract_skill=True`, a loop run that genuinely converges (via `convergence_fn` or `verifier`) with a quality signal at or above `skill_confidence_threshold` is distilled into a reusable, replayable `Skill` and registered in `SkillRegistry` — Voyager-style. The distilled skill doesn't just memoize the one answer; it re-runs the original agent's `system_prompt`/`tools`/`mcps` against whatever new task text it's called with, so it generalizes to similar future tasks. This never triggers on plain `max_iterations` exhaustion without real convergence, and is off by default.

```python
from deepcrew import Agent, run_agent, LoopConfig, Verifier, VerifierConfig, SkillRegistry

researcher = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=4,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
        auto_extract_skill=True,
        skill_confidence_threshold=0.85,
    ),
)

result = await run_agent(researcher, [{"role": "user", "content": "Explain CRISPR"}])

# Later, a completely different agent can reuse the distilled skill by name.
distilled = [s for s in SkillRegistry.list_all() if s.name.startswith("researcher_")][0]
writer = Agent(name="writer", model="openai/gpt-4o-mini", skills=[distilled])
```

Listen for `EventType.SKILL_EXTRACTED` (`{"skill_name": ..., "score": ...}`) to know when a new skill was registered.

### Common pitfalls

    - The `@skill` decorator's auto-schema has no support for complex types. Lists, dicts, unions, and enums all silently become plain `"string"` parameters — write a full `Skill` subclass with a hand-authored `parameters` schema if you need anything beyond int/float/bool/string.
    - `FunctionSkill.execute()` stringifies whatever your function returns. A function returning a dict or list gets `str(result)` applied to it (Python's default repr), which is rarely what you want an LLM to parse back — return a string yourself, or use `json.dumps(...)` explicitly.
    - Distilled skills only reuse a fixed subset of the original agent's config. The replay agent copies `system_prompt`, `tools`, `mcps`, `skills`, `max_turns`, `temperature`, `max_tokens`, and `extra_params` — it does not carry over `memory`, `procedural_memory`, `retry_policy`, `fallback_chain`, or `hooks` from the original agent.
    - `SkillRegistry` is process-global, class-level state. It is not automatically cleared between runs — tests that rely on a clean registry call `SkillRegistry.clear()` themselves in setup/teardown.

### See also

    - [Looping → Skill distillation](looping.html#skill-distillation) — the mechanics of how a converged run becomes a `Skill`.
    - [Verifier](verifier.html) — the quality signal that gates distillation.
    - [Agent Spawning](spawning.html) — `ToolAllocator` reads skill descriptions the same way it reads tool descriptions.
