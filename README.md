<p align="center">
  <img src="pages/assets/logo-readme.svg" width="72" height="72" alt="deepcrew-ai logo" />
</p>

<h1 align="center">deepcrew-ai</h1>

Multi-agent AI library for Python. Build parallel workflows, spawn agents dynamically, attach tools via MCP, and stream events in real time — using any of 100+ LLM providers, with minimal boilerplate.

```bash
pip install deepcrew-ai
```

[![PyPI](https://img.shields.io/pypi/v/deepcrew-ai)](https://pypi.org/project/deepcrew-ai)
[![Python](https://img.shields.io/pypi/pyversions/deepcrew-ai)](https://pypi.org/project/deepcrew-ai)
[![CI](https://github.com/Aayush-Joshi-01/deepcrew-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Aayush-Joshi-01/deepcrew-ai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Full docs:** [deepcrew-ai.aayushjoshi.dev](https://deepcrew-ai.aayushjoshi.dev) · [Features guide](https://deepcrew-ai.aayushjoshi.dev/features.html) · [Examples](https://deepcrew-ai.aayushjoshi.dev/examples.html) · [Migrating from CrewAI/ADK](https://deepcrew-ai.aayushjoshi.dev/guides/migration.html)

---

<details>
<summary><strong>Table of contents</strong></summary>

- [What's new in v0.4.0 — Production Readiness & Streaming-First](#whats-new-in-v040--production-readiness--streaming-first)
- [What's new in v0.3.0 — Self-Improving Loop](#whats-new-in-v030--self-improving-loop)
- [Features](#features)
- [Quick Start](#quick-start)
- [Orchestration Modes](#orchestration-modes)
  - [Workflow Builder — explicit DAG](#workflow-builder--explicit-dag)
  - [Orchestrator — automated AI routing with APEX](#orchestrator--automated-ai-routing-with-apex)
- [Feature Guide (v0.2.0 – v0.3.0)](#feature-guide-v020--v030)
  - [APEX Synthesizer](#apex-synthesizer)
  - [Agent Spawning](#agent-spawning) — incl. [nested spawning](#nested-spawning-bounded)
  - [Looping](#looping) — incl. [verifier](#verifier-driven-refinement), [adaptive budget](#adaptive-iteration-budget), [branching](#branching-self-consistency)
  - [Skills](#skills) — incl. [self-evolving skills](#self-evolving-skills-voyager-inspired)
  - [Memory Providers](#memory-providers) — incl. [procedural memory](#procedural-memory-evolving-playbook)
  - [Retry & Fallback](#retry--fallback)
  - [Observability (OpenTelemetry)](#observability-opentelemetry)
  - [CLI](#cli)
- [Composing v0.3.0 Features](#composing-v030-features)
- [MCP Tools](#mcp-tools)
- [Provider Examples](#provider-examples)
- [Streaming with FastAPI](#streaming-with-fastapi)
- [Stream Events](#stream-events)
- [Installation](#installation)
- [Documentation](#documentation)
- [License](#license)

</details>

---

## What's new in v0.4.0 — Production Readiness & Streaming-First

| Feature | Description |
|---|---|
| **Multimodal Input** | `image()`, `pdf()`, `user_message()` — attach images and PDFs to any message as standard content blocks |
| **StreamPolicy** | `chat()` / `standard()` / `verbose()` presets (or a custom set) control exactly which event types a consumer sees |
| **FastAPI Integration** | `create_stream_router()` turns an `Agent`, `Orchestrator`, or `WorkflowBuilder` into an SSE endpoint in one call (`fastapi` extra) |
| **Structured Output** | `Agent(response_model=...)` validates the final answer against a pydantic schema; result on `AgentResult.parsed` |
| **Human-in-the-Loop Hooks** | `AgentHooks.approve_tool` can deny an individual tool call before it runs |
| **Redis Memory Provider** | `RedisMemoryProvider` — a persistent, shared `MemoryProvider` backed by Redis (`redis` extra) |
| **Production hardening** | `py.typed`, ruff + mypy in CI, structured logging, a version-consistency check, and a tagged release workflow that publishes to PyPI with twine and attaches the wheel/sdist to a GitHub Release |

```python
from deepcrew import Agent, run_agent, image, pdf, user_message

agent = Agent(name="analyst", model="anthropic/claude-opus-4-8")
msg = user_message("Summarize this chart against the report.", image("chart.png"), pdf("report.pdf"))
result = await run_agent(agent, [msg])
```

See the [features guide](https://deepcrew-ai.aayushjoshi.dev/features.html) for full examples of
every feature above, and the [migration guide](https://deepcrew-ai.aayushjoshi.dev/guides/migration.html)
if you're coming from CrewAI or Google ADK.

---

## What's new in v0.3.0 — Self-Improving Loop

v0.3.0 turns the outer refinement loop (`LoopConfig`) from a simple "re-run until a
boolean says stop" mechanism into a genuinely self-improving one, and gives agent
spawning bounded recursion. Every addition is opt-in and fully backward compatible.

| Feature | Description |
|---|---|
| **APEX Synthesizer** | Intelligent synthesis with confidence scoring and source citation |
| **Agent Spawning** | Claude Code-style — agents can spawn sub-agents mid-loop with intelligent tool allocation |
| **Nested Spawning** | A spawned sub-agent can itself spawn further sub-agents up to a hard, bounded depth cap |
| **Looping** | Outer iteration loop for search-refine patterns with convergence control |
| **Verifier** | Structured, LLM-graded critique (score + issues + suggestion) driving targeted refinement |
| **Adaptive Budget** | Plateau-detection early exit — stop once refinement stops paying off, bounded by `max_iterations` |
| **Branching** | Parallel self-consistency candidates per iteration, scored by `Verifier` or merged via APEX |
| **Skills** | Higher-level capability bundles: `WebSearchSkill`, `SummarizeSkill`, `CodeExecutionSkill`, `@skill` decorator |
| **Self-Evolving Skills** | A converged, high-confidence loop run is distilled into a reusable, replayable `Skill` |
| **Memory** | Pluggable `InMemoryProvider` and `FileMemoryProvider` — auto-injected into agent context |
| **Procedural Memory** | An opt-in, durable evolving playbook of strategies accumulated across runs (ACE-inspired) |
| **Retry & Fallback** | Per-agent `RetryPolicy` + `FallbackChain` for model resilience |
| **Observability** | OpenTelemetry spans for every LLM call, tool execution, and workflow step |
| **CLI** | `deepcrew run workflow.yaml` — declarative workflow execution |

---

## Features

- **100+ LLM providers** via [LiteLLM](https://github.com/BerriAI/litellm) — OpenAI, Anthropic, Gemini, Bedrock, Azure, Ollama, and more
- **Two orchestration modes** — explicit DAG `WorkflowBuilder` or automated `Orchestrator` routing
- **APEX synthesis** — confidence-scored, citation-aware multi-agent result merging
- **Parallel execution** — `asyncio.gather` at every level: tools, agents, workflow nodes
- **MCP tool integration** — stdio (subprocess), SSE (legacy), and streamable-HTTP transports
- **Python function tools** — `@tool` decorator; JSON Schema auto-generated from type hints
- **Skills** — reusable capability bundles exposed to the LLM as tools
- **Memory providers** — pluggable short-term and persistent context stores
- **Retry & fallback** — per-agent exponential backoff + model fallback chains
- **OpenTelemetry** — optional OTel span emission for every LLM call and tool execution
- **SSE streaming** — compatible with FastAPI `StreamingResponse`
- **CLI** — `deepcrew run workflow.yaml` for declarative workflow files

---

## Quick Start

```python
import asyncio
from deepcrew import Agent, run_agent, tool

@tool
def get_weather(city: str) -> dict:
    "Get current weather for a city."
    return {"city": city, "temp": 22, "condition": "sunny"}

async def main():
    agent = Agent(
        name="assistant",
        model="openai/gpt-4o-mini",
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
    )
    result = await run_agent(agent, [{"role": "user", "content": "Weather in Tokyo?"}])
    print(result.text)

asyncio.run(main())
```

---

## Orchestration Modes

### Workflow Builder — explicit DAG

Define agents and their dependencies. Independent nodes run in parallel automatically.

```python
from deepcrew import Agent, WorkflowBuilder

researcher = Agent("researcher", model="openai/gpt-4o-mini",
                   system_prompt="Research the topic thoroughly.")
critic     = Agent("critic",     model="anthropic/claude-haiku-4-5-20251001",
                   system_prompt="Find gaps and weaknesses in the research.")
writer     = Agent("writer",     model="openai/gpt-4o",
                   system_prompt="Write a polished report.")

workflow = (
    WorkflowBuilder()
    .add_agent("research", researcher, task="{input}")
    .add_agent("critique", critic,     task="Critique this research:\n{research}")
    .add_agent("report",   writer,     task="Write a report using:\n{research}\n\nCritique:\n{critique}")
    .then("research", "critique")
    .then("research", "report")
    .then("critique", "report")
)

result = await workflow.run("The future of renewable energy")
print(result.final_output.text)
```

### Orchestrator — automated AI routing with APEX

The router LLM decides which agents to run; APEX synthesizes the results.

```python
from deepcrew import Agent, Orchestrator, ApexConfig

agents = [
    Agent("researcher", model="openai/gpt-4o-mini",  system_prompt="Research specialist."),
    Agent("analyst",    model="gemini/gemini-2.0-flash", system_prompt="Data analyst."),
    Agent("writer",     model="anthropic/claude-haiku-4-5-20251001", system_prompt="Content writer."),
]

orch = Orchestrator(
    agents=agents,
    router_model="openai/gpt-4o-mini",
    apex_model="openai/gpt-4o",
    apex_config=ApexConfig(cite_sources=True, confidence_threshold=0.8),
)

result = await orch.run("State of quantum computing in 2026")
print(result.final_text)
```

---

## Feature Guide (v0.2.0 – v0.3.0)

### APEX Synthesizer

APEX replaces the plain synthesizer with confidence scoring and source citation.

```python
from deepcrew import APEXSynthesizer, ApexConfig, AgentResult

apex = APEXSynthesizer(
    model="openai/gpt-4o",
    config=ApexConfig(
        cite_sources=True,          # adds [source: agent_name] markers
        confidence_threshold=0.8,
        allow_tools=False,
    ),
)

result = await apex.synthesize("What is quantum computing?", agent_results)
print(result.text)         # synthesis with optional citations
print(result.confidence)   # float 0.0–1.0
```

### Agent Spawning

Any agent can dynamically spawn sub-agents mid-loop with intelligent tool allocation.

```python
from deepcrew import Agent, Orchestrator, tool

@tool
def search_web(query: str) -> str: ...

@tool
def read_file(path: str) -> str: ...

orch = Orchestrator(
    agents=[master_agent],
    global_tools=[search_web, read_file],  # pool available for sub-agents
    enable_spawn=True,                      # inject spawn_agent tool into every agent
)

# The master_agent can now call spawn_agent(task="...", model="...") mid-loop
```

#### Nested spawning (bounded)

A spawned sub-agent can itself spawn further sub-agents — useful when a delegated
sub-task is still too large to handle directly — but this is strictly **depth**-bounded,
never sibling/fan-out-bounded: each level can still spawn as many sub-agents as it
wants, but nesting depth is capped by `max_spawn_depth`. Below that hard cap, an
optional `spawn_complexity_check` (a `Verifier`) can gate whether decomposing further
is actually warranted, so agents don't nest "just because they can."

```python
from deepcrew import Agent, Orchestrator, Verifier, tool

orch = Orchestrator(
    agents=[master_agent],
    global_tools=[search_web, read_file],
    enable_spawn=True,
    max_spawn_depth=3,                      # up to 3 levels of nested delegation
    spawn_complexity_check=Verifier(),       # optional: skip nesting for simple sub-tasks
)
```

When a sub-agent tries to spawn beyond `max_spawn_depth`, it simply has no
`spawn_agent` tool to call — there's nothing to invoke, so it completes the task
directly. A defense-in-depth check inside the tool itself also returns "Maximum
nesting depth reached; complete this task directly without further delegation." as a
plain string result for any caller that bypasses the normal attach logic — never an
exception.

### Looping

Outer iteration loop for search-refine patterns.

```python
from deepcrew import Agent, LoopConfig

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=4,
        convergence_fn=lambda r: len(r.text) > 500,
        refine_prompt="Your answer is incomplete. Search more and expand it.",
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
print(f"Converged in {result.loop_iterations} iterations")
```

#### Verifier-driven refinement

A `Verifier` grades each iteration's answer against the original query and returns
**structured** feedback — a score, specific issues, and a concrete suggestion — instead
of a boolean. When set, the loop uses this feedback both to decide convergence and to
build a targeted refinement prompt, instead of the static `refine_prompt` string.

```python
from deepcrew import Agent, LoopConfig, Verifier, VerifierConfig

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=4,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
```

`convergence_fn` and `verifier` can be combined — the loop stops as soon as either one
is satisfied. Pass `VerifierConfig(evaluate_fn=my_custom_grader)` to fully replace the
built-in LLM-graded rubric with your own scoring function.

#### Adaptive iteration budget

With `adaptive=True`, the loop tracks the verifier score across iterations and stops
early once improvement plateaus, instead of always running `max_iterations`. This is
still bounded by `max_iterations` — adaptive can only shorten the loop, never lengthen
it. Requires `verifier` to be set (there's no score to track otherwise).

```python
agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=8,
        verifier=Verifier(VerifierConfig(threshold=0.9)),
        adaptive=True,
        min_improvement=0.02,   # minimum score delta to still count as "improving"
        plateau_patience=2,     # stop after this many non-improving iterations in a row
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
```

#### Branching (self-consistency)

With `branches > 1`, each iteration runs that many parallel candidate continuations
instead of one, and picks the best via `verifier` score (or merges them with
`APEXSynthesizer` when no verifier is configured). This multiplies LLM call volume by
`branches` per iteration — use it deliberately, typically with a low `max_iterations`.

```python
agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=3,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
        branches=3,  # 3x the LLM calls per iteration, in exchange for picking the best
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
```

See also: [Self-evolving skills](#self-evolving-skills-voyager-inspired) —
`LoopConfig.auto_extract_skill` distills a converged loop run into a reusable `Skill`
for other agents to retrieve later.

#### Procedural memory (evolving playbook)

`ProceduralMemory` is an opt-in, durable "the system learns from its own past runs"
store, inspired by ACE (Agentic Context Engineering). It's built on top of any
`MemoryProvider` and accumulates a structured playbook of "helpful"/"avoid" strategy
bullets across runs — read on every run (looped or single-shot) and curated
(incrementally merged, not rewritten) whenever a loop with a `verifier` converges.

```python
from deepcrew import (
    Agent, LoopConfig, Verifier, VerifierConfig,
    FileMemoryProvider, ProceduralMemory,
)

playbook = ProceduralMemory(FileMemoryProvider("playbook.json"), max_entries=30)

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    procedural_memory=playbook,   # read on every run, even single-shot
    loop_config=LoopConfig(
        max_iterations=4,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
        procedural_memory=playbook,  # curated after a converged loop
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
# Run the same agent again later (even a new process, with FileMemoryProvider) and
# it will already know what worked and what to avoid for this task.
```

Curation requires a `verifier` — without one, `procedural_memory` on `LoopConfig` is a
no-op (there's no `VerifierFeedback` to grade the run against).

### Skills

Higher-level capability bundles. Built-ins included:

```python
from deepcrew import Agent, run_agent
from deepcrew import WebSearchSkill, SummarizeSkill, CodeExecutionSkill

agent = Agent(
    name="assistant",
    model="openai/gpt-4o",
    skills=[
        WebSearchSkill(),
        SummarizeSkill(model="openai/gpt-4o-mini"),
        CodeExecutionSkill(timeout=10.0),
    ],
)

result = await run_agent(agent, [{"role": "user", "content": "Search for Python async best practices and summarize them."}])
```

Custom skills with the `@skill` decorator:

```python
from deepcrew import skill

@skill(name="translate", description="Translate text to another language")
async def translate(text: str, target_language: str) -> str:
    # your implementation
    return translated_text
```

#### Self-evolving skills (Voyager-inspired)

With `LoopConfig.auto_extract_skill=True`, a loop run that genuinely converges (via
`convergence_fn` or `verifier`) with a quality signal at or above
`skill_confidence_threshold` is automatically distilled into a reusable, replayable
`Skill` and registered in `SkillRegistry` — so a *different* agent, later, can retrieve
and reuse the proven approach instead of re-deriving it from scratch. This never
triggers on plain `max_iterations` exhaustion without real convergence, and is fully
opt-in (off by default, no surprise side effects).

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

### Memory Providers

Auto-injected context across turns and agent runs.

```python
from deepcrew import Agent, run_agent, InMemoryProvider, FileMemoryProvider

# Short-term (in-process)
agent = Agent(
    name="bot",
    model="openai/gpt-4o-mini",
    memory=InMemoryProvider(),
)

# Persistent (JSON file)
agent = Agent(
    name="bot",
    model="openai/gpt-4o-mini",
    memory=FileMemoryProvider("~/.deepcrew/memory.json"),
)
```

### Retry & Fallback

```python
from deepcrew import Agent, RetryPolicy, FallbackChain

agent = Agent(
    name="resilient",
    model="openai/gpt-4o",
    retry_policy=RetryPolicy(max_retries=3, backoff_seconds=1.0, exponential=True),
    fallback_chain=FallbackChain(models=[
        "anthropic/claude-haiku-4-5-20251001",
        "gemini/gemini-2.0-flash",
    ]),
)
```

### Observability (OpenTelemetry)

```bash
pip install "deepcrew-ai[otel]"
```

```python
from deepcrew import Agent, run_agent, ObservabilityConfig

obs = ObservabilityConfig(
    otel_endpoint="http://localhost:4317",
    service_name="my-ai-app",
)

result = await run_agent(agent, messages, observability=obs)
# Emits spans: agent.run, llm.call, tool.call
```

### CLI

```bash
# Run a declarative workflow
deepcrew run workflow.yaml --input "The future of AI"

# List agents in a config
deepcrew agents list --config workflow.yaml

# Version
deepcrew --version
```

`workflow.yaml` example:

```yaml
agents:
  - name: researcher
    model: openai/gpt-4o-mini
    system_prompt: Research the topic.
    tools: [web_search]
  - name: writer
    model: openai/gpt-4o
    system_prompt: Write clearly.

workflow:
  - step: research
    agent: researcher
    task: "{input}"
  - step: report
    agent: writer
    task: "Write about:\n{research}"
    depends_on: [research]
```

---

## Composing v0.3.0 Features

The six self-improving-loop additions aren't just usable in isolation — they compose.
A single agent can be configured with a verifier-driven, adaptive, branching,
skill-distilling, procedural-memory-backed loop, orchestrated by a coordinator that
allows bounded nested delegation:

```python
from deepcrew import (
    Agent, Orchestrator, LoopConfig, Verifier, VerifierConfig,
    FileMemoryProvider, ProceduralMemory,
)

verifier = Verifier(VerifierConfig(threshold=0.85))
playbook = ProceduralMemory(FileMemoryProvider("researcher_playbook.json"))

researcher = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    procedural_memory=playbook,          # read on every run
    loop_config=LoopConfig(
        max_iterations=6,
        verifier=verifier,                # structured critique drives refinement
        procedural_memory=playbook,        # curated after convergence
        adaptive=True,                     # stop early once improvement plateaus
        min_improvement=0.02,
        plateau_patience=2,
        branches=2,                        # 2 parallel candidates per iteration
        auto_extract_skill=True,           # distill a reusable Skill on success
        skill_confidence_threshold=0.85,
    ),
)

orch = Orchestrator(
    agents=[researcher],
    router_model="openai/gpt-4o-mini",
    global_tools=[search_web, read_file],
    enable_spawn=True,
    max_spawn_depth=2,                    # bounded recursive delegation
    spawn_complexity_check=verifier,       # reuse the same verifier as a complexity gate
)

result = await orch.run("Research and summarize the current state of fusion energy")
print(result.final_text)
```

Every feature here defaults off — this example opts into all six deliberately to show
how they fit together, not as a recommended baseline configuration.

---

## MCP Tools

```python
from deepcrew.mcp import StdioMCP, HTTPMCP, SSEMCP, MCPManager

# Stdio (subprocess)
async with StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as mcp:
    agent = Agent("file_agent", model="openai/gpt-4o", mcps=[mcp])

# HTTP (modern)
async with HTTPMCP("https://my-mcp.example.com/mcp",
                   headers={"Authorization": "Bearer sk-..."}) as mcp:
    agent = Agent("agent", model="openai/gpt-4o", mcps=[mcp])

# Multiple servers
async with MCPManager([
    StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]),
    HTTPMCP("https://search-mcp.example.com/mcp"),
]) as manager:
    agent = Agent("agent", model="openai/gpt-4o", mcps=[manager])
```

---

## Provider Examples

```python
# OpenAI
Agent("a", model="openai/gpt-4o")
Agent("a", model="openai/gpt-4o-mini")

# Anthropic
Agent("a", model="anthropic/claude-opus-4-8")
Agent("a", model="anthropic/claude-haiku-4-5-20251001")

# Google
Agent("a", model="gemini/gemini-2.0-flash")
Agent("a", model="gemini/gemini-2.5-pro")

# AWS Bedrock
Agent("a", model="bedrock/anthropic.claude-opus-4-8-20250514-v1:0")

# Azure OpenAI
Agent("a", model="azure/gpt-4o", extra_params={"api_base": "https://..."})

# Local via Ollama
Agent("a", model="ollama/llama3.2")
Agent("a", model="ollama/qwen2.5-coder")
```

---

## Streaming with FastAPI

The `fastapi` extra (`pip install deepcrew-ai[fastapi]`) gives you a streaming endpoint in one call:

```python
from fastapi import FastAPI
from deepcrew import Agent, Orchestrator, ApexConfig, StreamPolicy
from deepcrew.integrations.fastapi import create_stream_router

orch = Orchestrator(
    agents=[Agent("assistant", model="openai/gpt-4o")],
    apex_config=ApexConfig(cite_sources=True),
)

app = FastAPI()
app.include_router(create_stream_router(orch, policy=StreamPolicy.chat()))
# POST /chat streams Server-Sent Events; POST /chat/complete returns the final result as JSON.
```

Or wire it up manually against any `.stream()`-producing object:

```python
from fastapi.responses import StreamingResponse

@app.post("/chat-manual")
async def chat(query: str):
    async def event_stream():
        async for event in orch.stream(query, policy=StreamPolicy.chat()):
            yield event.to_sse()
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## Stream Events

All events have `event` (EventType), `agent_id` (str), and `data` (dict).

| Event | Data keys | Emitted by |
|---|---|---|
| `agent_start` | `model` | Agent loop begins |
| `text_delta` | `chunk` | Each streamed text token |
| `tool_call` | `tool`, `args` | Before tool execution |
| `tool_result` | `tool`, `result` | After tool execution |
| `tool_denied` | `tool` | An `AgentHooks.approve_tool` hook denied the call |
| `thinking_delta` | `chunk` | Each streamed reasoning-content token (if the model exposes one) |
| `agent_done` | `input_tokens`, `output_tokens` | Agent finished |
| `apex_start` | `agents` | APEX synthesis begins |
| `apex_done` | `confidence` | APEX synthesis complete |
| `loop_iteration` | `iteration`, `converged` | Each outer loop iteration |
| `verifier_scored` | `iteration`, `score`, `issues` | Verifier grades an iteration |
| `playbook_updated` | `task_tag`, `entry_count` | Procedural memory curated |
| `branch_selected` | `branch_count`, `winning_index`, `winning_score` | Best/merged branch chosen |
| `skill_extracted` | `skill_name`, `score` | A converged run distilled into a Skill |
| `spawn_agent` | `task`, `requested_tools`, `depth` | Sub-agent spawned |
| `memory_retrieve` | `count` | Memories injected into context |
| `memory_store` | `key` | Tool result stored to memory |
| `retry_attempt` | `attempt`, `model`, `delay` | Before a retry |
| `fallback_triggered` | `from_model`, `to_model` | Model switch |
| `step_start` | `node` | Workflow node begins |
| `step_done` | `node` | Workflow node finished |
| `error` | `message` | Any exception |
| `done` | `final_text` (Orchestrator/WorkflowBuilder only) | The whole run/pipeline finished |
| `done` | `final_text` | Entire run complete |

---

## Installation

```bash
# Core
pip install deepcrew-ai

# With OpenTelemetry
pip install "deepcrew-ai[otel]"

# With the FastAPI streaming integration
pip install "deepcrew-ai[fastapi]"

# With the Redis memory provider
pip install "deepcrew-ai[redis]"

# With dev dependencies
pip install "deepcrew-ai[dev]"
```

Requires Python 3.11+.

---

## Documentation

Full documentation at **[deepcrew-ai.aayushjoshi.dev](https://deepcrew-ai.aayushjoshi.dev)**

- [Getting Started](https://deepcrew-ai.aayushjoshi.dev)
- [Features](https://deepcrew-ai.aayushjoshi.dev/features.html) — one page per feature
- [Examples Library](https://deepcrew-ai.aayushjoshi.dev/examples.html)
- [Migration Guide (CrewAI / Google ADK)](https://deepcrew-ai.aayushjoshi.dev/guides/migration.html)
- [llms.txt](https://deepcrew-ai.aayushjoshi.dev/llms.txt) — machine-readable docs index

---

## License

MIT
