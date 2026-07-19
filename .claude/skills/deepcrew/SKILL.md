---
name: deepcrew
description: Build multi-agent AI systems with deepcrew-ai — single agents, orchestrated multi-agent pipelines with APEX synthesis, self-improving verifier loops, bounded agent spawning, DAG workflows, multimodal (image/PDF) input, structured output, human-in-the-loop tool approval, and streaming (including a FastAPI SSE endpoint). Use this whenever the user asks about "multi-agent", "agent orchestration", "self-improving agent", "AI agent framework", or mentions deepcrew/deepcrew-ai directly.
---

# deepcrew-ai integration skill

`deepcrew-ai` is an async, LiteLLM-backed multi-agent library. This skill teaches you how to wire
it into any Python project correctly on the first try.

## Install

```bash
pip install deepcrew-ai
# optional extras:
pip install deepcrew-ai[fastapi]   # SSE streaming endpoint
pip install deepcrew-ai[redis]     # Redis-backed memory
pip install deepcrew-ai[otel]      # OpenTelemetry tracing
```

API keys are read by LiteLLM from standard env vars — set whichever providers you use:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc. The `model` string on every `Agent`
determines the provider, e.g. `"openai/gpt-4o"`, `"anthropic/claude-opus-4-8"`,
`"gemini/gemini-2.0-flash"`, `"ollama/llama3.2"` (local, no key needed).

## Decision table

| You need... | Use |
|---|---|
| One agent, no coordination | `Agent` + `run_agent()` |
| Router picks one agent or fans out to several in parallel | `Orchestrator` |
| An explicit, fixed pipeline of steps with dependencies | `WorkflowBuilder` |
| An agent that critiques and improves its own answer | `Agent(loop_config=LoopConfig(verifier=...))` |
| An agent that can delegate sub-tasks to fresh agents mid-run | `Orchestrator(enable_spawn=True)` |
| A simple chatbot UI that should only show the reply text | `StreamPolicy.chat()` |
| A technical/debug UI that should show everything | `StreamPolicy.verbose()` |
| Approve or block individual tool calls before they run | `AgentHooks(approve_tool=...)` |

## Recipes

### Single agent + tool

```python
from deepcrew import Agent, run_agent, tool

@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"Sunny in {city}"

agent = Agent(name="assistant", model="openai/gpt-4o", tools=[get_weather])
result = await run_agent(agent, [{"role": "user", "content": "Weather in Tokyo?"}])
print(result.text)
```

### Multimodal query (image + PDF)

```python
from deepcrew import Agent, run_agent, image, pdf, user_message

agent = Agent(name="analyst", model="anthropic/claude-opus-4-8")
msg = user_message("Summarize this chart and check it against the report.",
                    image("chart.png"), pdf("report.pdf"))
result = await run_agent(agent, [msg])
```

`image()`/`pdf()` accept a URL, a local file path, or raw bytes. Local/byte sources are
size-checked and base64-encoded automatically; raises `ContentError` on bad input.

### Structured output

```python
from pydantic import BaseModel
from deepcrew import Agent, run_agent

class Verdict(BaseModel):
    approved: bool
    reason: str

agent = Agent(name="reviewer", model="openai/gpt-4o", response_model=Verdict)
result = await run_agent(agent, [{"role": "user", "content": "Review this PR diff: ..."}])
result.parsed  # a validated Verdict instance (one automatic repair attempt on bad JSON)
```

### Orchestrator with APEX synthesis

```python
from deepcrew import Agent, Orchestrator, ApexConfig

orch = Orchestrator(
    agents=[researcher, analyst, writer],
    router_model="openai/gpt-4o-mini",
    apex_model="openai/gpt-4o",
    apex_config=ApexConfig(cite_sources=True),
)
result = await orch.run("Explain quantum entanglement")
print(result.final_text)
```

### WorkflowBuilder DAG

```python
from deepcrew import WorkflowBuilder

workflow = (
    WorkflowBuilder()
    .add_agent("research", researcher, task="{input}")
    .add_agent("write", writer, task="Write about:\n{research}")
    .then("research", "write")
)
result = await workflow.run("Quantum computing trends")
```

### Self-improving loop

```python
from deepcrew import Agent, LoopConfig, Verifier, VerifierConfig, run_agent

agent = Agent(
    name="researcher",
    model="openai/gpt-4o",
    loop_config=LoopConfig(
        max_iterations=5,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
        adaptive=True,           # early-stop on score plateau
    ),
)
result = await run_agent(agent, [{"role": "user", "content": "Research topic X thoroughly."}])
```

### Bounded agent spawning

```python
orch = Orchestrator(agents=[planner], router_model="openai/gpt-4o-mini",
                     enable_spawn=True, max_spawn_depth=2)
result = await orch.run("Plan and execute a multi-step research task")
```

### Human-in-the-loop tool approval

```python
from deepcrew import Agent, AgentHooks, run_agent

async def approve(tool_name: str, args: dict) -> bool:
    return tool_name != "delete_file"  # block this one specific tool

agent = Agent(name="assistant", model="openai/gpt-4o", tools=[delete_file, read_file],
              hooks=AgentHooks(approve_tool=approve))
```

Hooks *intercept and can change* execution — different from the observe-only event stream.

### MCP tools

```python
from deepcrew import Agent
from deepcrew.mcp import StdioMCP

fs_mcp = StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])
agent = Agent(name="assistant", model="openai/gpt-4o", mcps=[fs_mcp])
```

### FastAPI SSE endpoint

```python
from fastapi import FastAPI
from deepcrew import Agent, StreamPolicy
from deepcrew.integrations.fastapi import create_stream_router

agent = Agent(name="assistant", model="openai/gpt-4o")
app = FastAPI()
app.include_router(create_stream_router(agent, policy=StreamPolicy.chat()))
# POST /chat streams SSE; POST /chat/complete returns the final result as JSON.
```

## Pitfalls

- **`max_turns` exhaustion**: `Agent.max_turns` (default 10) caps LLM↔tool cycles; a tool-heavy
  agent that never converges raises `MaxTurnsError`. Raise `max_turns` or fix the tool loop.
- **The Orchestrator's router is text-only**: it runs in JSON mode to decide routing, so it never
  sees `attachments` directly — only a text summary of how many images/documents exist. The
  executing agent(s) get the actual attachments.
- **File/PDF content blocks aren't universal**: `pdf()` produces a `file` content block supported
  by OpenAI, Anthropic, and Gemini via LiteLLM. Sending it to an unsupported provider raises an
  API error — `litellm.drop_params` strips unsupported top-level params, not content blocks.
- **Hooks vs. events**: `AgentHooks` can *deny* a tool call (`approve_tool` returning `False`).
  The `StreamEvent` queue and `StreamPolicy` only ever *observe* — they never change execution.
- **`StreamPolicy` never hides `done`/`error`**: presets always let terminal events through so a
  consumer can tell the stream ended, even under `StreamPolicy.chat()`.
- **Spawned sub-agents don't inherit multimodal attachments** — the parent agent has already seen
  the images/documents and is expected to describe what's relevant in the spawn task text.

## Further reading

- `pages/md/migration.md` — mapping from CrewAI / Google ADK concepts to deepcrew equivalents.
- `pages/guides/streampolicy.html` — full `StreamPolicy` reference.
- `pages/guides/fastapi.html` — the FastAPI SSE integration.
- `examples/` — runnable end-to-end scripts for every recipe above.
