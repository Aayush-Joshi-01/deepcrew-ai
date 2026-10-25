# deepcrew-ai

Multi-agent AI library for Python, built on LiteLLM (100+ providers via one interface).

## Install

```bash
pip install deepcrew-ai
# optional extras
pip install deepcrew-ai[fastapi]   # SSE streaming endpoint
pip install deepcrew-ai[redis]     # Redis-backed memory
pip install deepcrew-ai[otel]      # OpenTelemetry tracing
pip install deepcrew-ai[log]       # coloured logging via rich (plain ANSI fallback without it)
```

Set the API key env var for whichever provider(s) you use: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc. The `model` string on each `Agent` determines the
provider, e.g. `"openai/gpt-4o"`, `"anthropic/claude-opus-4-8"`, `"ollama/llama3.2"` (local).

## Quick start

```python
from deepcrew import Agent, run_agent

agent = Agent(name="assistant", model="openai/gpt-4o", system_prompt="You are helpful.")
result = await run_agent(agent, [{"role": "user", "content": "Hello!"}])
print(result.text)
```

## Core building blocks

- **`Agent`** — model, system prompt, tools, memory, hooks, optional `response_model`.
- **`run_agent()`** — the agentic loop: stream → buffer tool calls → execute in parallel → repeat.
- **`Orchestrator`** — NIA (Navigator & Intelligence Allocator), an LLM router, picks a single
  agent or fans out to several in parallel, then `APEXSynthesizer` (APEX, Assembly & Precision
  Executor) merges parallel results with confidence scoring and optional citations.
- **`WorkflowBuilder`** — an explicit DAG of agents; independent nodes at the same level run in
  parallel automatically.
- **`LoopConfig`** — an outer refinement loop: a `Verifier` scores each iteration and drives
  targeted refinement, with optional adaptive early-stop and self-consistency branching.
- **Bounded agent spawning** — `Orchestrator(enable_spawn=True)` lets agents dynamically spawn
  sub-agents mid-run via a `spawn_agent` meta-tool, hard-capped by `max_spawn_depth`.
- **`MemoryProvider`** — pluggable context store (`InMemoryProvider`, `FileMemoryProvider`,
  `RedisMemoryProvider`), auto-injected into each LLM call.
- **`ProceduralMemory`** — a durable, evolving playbook of strategies curated by a verifier loop and
  read back into later runs (optional LLM `reflect_model`).
- **`EpisodicMemory`** — a durable log of past runs; relevant episodes are recalled before each new
  run and the outcome recorded after (`Agent(episodic_memory=...)`).
- **`configure_logging()` / `get_logger()`** — opt-in structured logging (dev/prod/verbose/quiet
  profiles, JSON output); `get_logger` is public for your own components.
- **MCP tools** — `StdioMCP`, `SSEMCP`, `HTTPMCP`, `MCPManager` for attaching MCP servers.
- **`StreamEvent`/`StreamPolicy`** — every consequential action streams as an event; `StreamPolicy`
  controls which event types a given consumer sees (`chat()`, `standard()`, `verbose()`, custom).

## Latest additions

Multimodal input (`image()`, `pdf()`, `user_message()`), selectable streaming visibility
(`StreamPolicy`), an optional FastAPI SSE integration, structured output (`response_model`),
human-in-the-loop tool approval (`AgentHooks`), and a Redis memory provider (v0.4); opt-in
structured logging (v0.5); episodic memory, an LLM reflector for procedural memory, and an
opt-in live-test harness (v0.6). See the [features index](features.html) and
[migration guide](guides/migration.html).

## Documentation map

- [Features](features.html) — every feature has its own page: APEX, agent spawning, looping,
  verifier, procedural memory, episodic memory, skills, memory providers, retry/fallback,
  observability, logging, CLI, multimodal input, StreamPolicy, FastAPI integration,
  structured output, human-in-the-loop hooks,
  and the Redis memory provider. Each page has a "Copy prompt" button for AI-assisted implementation.
- [Examples](examples.html) — runnable end-to-end scripts and domain walkthroughs.
- [Migration Guide](guides/migration.html) — coming from CrewAI or Google ADK.
- [llms.txt](llms.txt) / [llms-full.txt](llms-full.txt) — machine-readable docs index for LLM tools.
