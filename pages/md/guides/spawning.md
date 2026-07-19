# Agent Spawning
deepcrew-ai v0.2.0 introduces Claude Code-style dynamic agent spawning. Any running agent can call a built-in `spawn_agent` tool to create a sub-agent mid-loop, with tools automatically selected from a global pool by the `ToolAllocator`.

### How it works

    - You provide a `global_tools` pool to `Orchestrator`
    - With `enable_spawn=True`, every agent gets a `spawn_agent(task, tools, model)` tool injected
    - When an agent calls `spawn_agent`, `ToolAllocator` uses the router LLM to pick the most relevant tools from the global pool for that specific task
    - A fresh sub-agent is created and runs to completion, its result returned to the parent
    - A `SPAWN_AGENT` stream event is emitted for observability

### Enable spawning via Orchestrator

```python
from deepcrew import Agent, Orchestrator, tool

@tool
def search_web(query: str) -> str:
    "Search the web."
    ...

@tool
def read_file(path: str) -> str:
    "Read a local file."
    ...

@tool
def run_sql(query: str) -> list[dict]:
    "Execute a SQL query."
    ...

@tool
def call_api(url: str, method: str = "GET") -> dict:
    "Make an HTTP API call."
    ...

master = Agent(
    name="coordinator",
    model="openai/gpt-4o",
    system_prompt="""You are a coordinator agent. For complex subtasks,
    use the spawn_agent tool to delegate to a specialized sub-agent.""",
)

orch = Orchestrator(
    agents=[master],
    router_model="openai/gpt-4o-mini",
    apex_model="openai/gpt-4o",
    global_tools=[search_web, read_file, run_sql, call_api],  # pool
    enable_spawn=True,   # injects spawn_agent tool into master
)

result = await orch.run(
    "Research the top 3 AI papers from last month and summarize key findings."
)
print(result.final_text)
```

### Standalone spawning

Calling `spawn_agent()` directly bypasses `Orchestrator` entirely — useful if you're building your own custom control flow. It needs a pool of `ToolDef` objects (not raw `@tool`-decorated functions), so convert with `fn_to_tool_def()` first:

```python
import asyncio
from deepcrew import spawn_agent, SpawnRequest, fn_to_tool_def

all_tool_defs = [fn_to_tool_def(search_web), fn_to_tool_def(read_file)]

request = SpawnRequest(
    task="Find all Python files that import pandas and list their names.",
    tools=["read_file", "search_web"],   # hint: names from the pool above
    model="openai/gpt-4o-mini",
    system_prompt="You are a code analysis assistant.",
    max_turns=5,
)

queue: asyncio.Queue = asyncio.Queue()

result = await spawn_agent(
    request=request,
    all_tool_defs=all_tool_defs,
    parent_queue=queue,             # None is fine if you don't need SPAWN_AGENT events
    router_model="openai/gpt-4o-mini",
    parent_agent_id="coordinator",
    max_depth=2,                    # default; see "Bounded nested spawning" below
)

print(result.text)
```

Note that `request.tools` is only a hint: `spawn_agent()` always runs the full `ToolAllocator` pass first (see below), then intersects the allocator's picks with your hinted names if you gave any — it never trusts the hint blindly, and falls back to the allocator's full selection if the intersection is empty.

### ToolAllocator

This is the piece that decides which tools a spawned sub-agent actually gets — an LLM call, not a keyword match, so tool descriptions matter as much as names.

```python
from deepcrew import ToolAllocator

allocator = ToolAllocator(router_model="openai/gpt-4o-mini")

# Given a task description and a large pool of tools,
# returns only the most relevant subset (up to max_tools)
relevant_tools = await allocator.allocate(
    task="Analyze sentiment in customer reviews and generate a report",
    all_tools=my_tool_defs,   # list[ToolDef] — could be dozens of tools
    max_tools=5,              # default is 10 if you omit this
)

print([t.name for t in relevant_tools])
```

> The allocator prompts the router model with each tool's `name` and `description` and asks for a JSON array of names back. It accepts either a bare array or an object wrapping one array value (some models prefer `{"tools": [...]}` over a bare array). If the response fails to parse as JSON at all, or the router call itself raises, `allocate()` degrades gracefully to returning the first `max_tools` tools from the pool in whatever order you passed them in — not a random or relevance-based subset. A good tool description dramatically improves allocation accuracy in the common case, and also matters for what you get in that fallback case, since pool order is your responsibility.

### SpawnRequest reference

- **task*** (str): Natural language description of the sub-task. Used both for tool allocation and as the sub-agent's first user message — the sub-agent never sees the parent's conversation history, only this string.

- **tools** (list[str] = []): Optional hint: tool names the parent agent thinks the sub-agent should have. `ToolAllocator` still runs first and makes the real decision; this hint only filters its output (see "Standalone spawning" above).

- **model** (str | None = None): Model string for the sub-agent. Falls back to `router_model` if omitted — not the parent agent's model, despite that being the more intuitive default.

- **system_prompt** (str | None = None): Optional system prompt override for the sub-agent. Defaults to a generic `"You are a helpful sub-agent. Complete the given task."` when omitted.

- **max_turns** (int = 5): Max tool-call cycles for the sub-agent — independent of and typically smaller than the parent's own `max_turns`.

- **depth** (int = 0): Nesting depth this spawn happens at. Set automatically by the `spawn_agent` tool wrapper (`make_spawn_tool`) when the LLM calls it; you only need to set this yourself when calling `spawn_agent()` from inside your own already-nested custom logic.

### How-to: bounded nested spawning

A spawned sub-agent can itself spawn further sub-agents — useful when a delegated sub-task is still too large to handle directly. This is strictly depth-bounded, never sibling/fan-out-bounded: each level can still spawn as many sub-agents as it wants, but nesting depth is capped by `max_spawn_depth`. Below that hard cap, an optional `spawn_complexity_check` gate can skip attaching a nested spawn tool when the sub-task doesn't look worth decomposing further.

```python
from deepcrew import Agent, Orchestrator, Verifier

orch = Orchestrator(
    agents=[master_agent],
    global_tools=[search_web, read_file],
    enable_spawn=True,
    max_spawn_depth=3,                    # up to 3 levels of nested delegation
    spawn_complexity_check=Verifier(),     # optional: skip nesting for simple sub-tasks
)
```

When a sub-agent tries to spawn beyond `max_spawn_depth`, it simply has no `spawn_agent` tool available — nothing to invoke, so it completes the task directly instead. A defense-in-depth check inside the tool itself also returns `"Maximum nesting depth reached; complete this task directly without further delegation."` as a plain string result for any caller that bypasses the normal attach logic — never an exception.

`spawn_complexity_check` takes any `Verifier` instance and calls its `assess_complexity(task, default_model=...)` method — a lightweight, separate LLM judgment ("does this task genuinely need decomposing, or can one agent handle it directly?") distinct from the answer-grading `evaluate()` method the same class exposes for the [Verifier](verifier.html) feature. If you pass a `Verifier` constructed with `evaluate_fn` (a custom grading callback), `assess_complexity` has no notion of pre-execution complexity and always stays permissive — it returns `True` unconditionally rather than trying to call your custom function for a purpose it wasn't written for.

### Manual wiring with `make_spawn_tool`

`Orchestrator(enable_spawn=True)` is a convenience wrapper — internally, it calls `make_spawn_tool()` once per orchestration to build the actual `ToolDef` that gets injected into each agent's tool list. Call it yourself if you're wiring spawning into an `Agent` you're running with `run_agent()` directly, outside of `Orchestrator` altogether:

```python
from deepcrew import Agent, run_agent, make_spawn_tool, fn_to_tool_def

pool = [fn_to_tool_def(search_web), fn_to_tool_def(read_file)]

spawn_tool = make_spawn_tool(
    all_tool_defs=pool,
    parent_queue=None,
    router_model="openai/gpt-4o-mini",
    parent_agent_id="solo-agent",
    current_depth=0,
    max_depth=2,
)

agent = Agent(
    name="solo-agent",
    model="openai/gpt-4o",
    tools=[],
    # ToolDef instances aren't plain @tool functions, so they don't go in `tools=` —
    # merge them into tool_defs when you call run_agent() instead:
)
result = await run_agent(agent, [{"role": "user", "content": "..."}], tool_defs=[spawn_tool])
```

### Common pitfalls

    - Sub-agents never see multimodal attachments. If the parent agent received images/PDFs via [multimodal input](multimodal.html), a spawned sub-agent gets none of them automatically — describe what's relevant directly in the `task` string.
    - `SpawnRequest.model` defaults to `router_model`, not the parent's model. If you want the sub-agent on the same model as its parent, pass it explicitly.
    - Depth is a hard ceiling, fan-out is not. `max_spawn_depth` only bounds how many levels deep delegation can go — it does nothing to limit how many sibling sub-agents one level spawns, which is a separate cost/runaway concern you may need to bound yourself (e.g. via your own tool-call counting or a stricter `max_turns`).
    - `ToolAllocator`'s failure mode is silent. A malformed JSON response or a router-call exception both fall back to "first `max_tools` tools in pool order," not an error — if allocation looks wrong, check whether the router is actually returning parseable JSON before assuming your tool descriptions are the problem.

### See also

    - [APEX Synthesizer](apex.html) — merges results when the router fans out to multiple top-level agents; spawning is delegation within one agent's turn.
    - [Verifier](verifier.html) — the same class that grades loop iterations also powers `spawn_complexity_check`.
    - [Multimodal Input](multimodal.html) — attachments and spawning interact; see the pitfall above.
