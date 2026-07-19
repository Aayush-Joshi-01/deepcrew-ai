# Migrating to deepcrew-ai

A concept map for teams coming from CrewAI or Google's Agent Development Kit (ADK), plus what
deepcrew adds once you're here.

## From CrewAI

| CrewAI concept | deepcrew equivalent | Notes |
|---|---|---|
| `Agent(role=, goal=, backstory=)` | `Agent(name=, system_prompt=)` | deepcrew doesn't split role/goal/backstory into separate fields — fold them into one `system_prompt`. |
| `Crew(agents=[...], process=Process.sequential)` | `WorkflowBuilder().add_agent(...).then(...)` | Explicit DAG edges instead of an implicit list order. |
| `Crew(process=Process.hierarchical)` | `Orchestrator(agents=[...])` | An LLM router picks single-agent or fan-out parallel execution; no manager agent to configure. |
| `@tool` / `BaseTool` subclass | `@tool` decorator on a plain function | Schema is auto-generated from type hints — no separate schema class. |
| `Agent(..., output_pydantic=Model)` | `Agent(..., response_model=Model)` | Validated result lands on `AgentResult.parsed`; one automatic repair attempt on invalid JSON. |
| `Task(human_input=True)` | `AgentHooks(approve_tool=...)` | deepcrew's human-in-the-loop is per-tool-call, not per-task; return `False` from `approve_tool` to deny. |
| Crew `verbose=True` | `StreamPolicy.verbose()` | Streaming, not console printing — see [Streaming guide](streaming.html). |

**Before (CrewAI):**
```python
from crewai import Agent, Crew, Task, Process

researcher = Agent(role="Researcher", goal="Find facts", backstory="...")
crew = Crew(agents=[researcher], tasks=[Task(description="...", agent=researcher)],
            process=Process.sequential)
result = crew.kickoff()
```

**After (deepcrew):**
```python
from deepcrew import Agent, run_agent

researcher = Agent(name="researcher", model="openai/gpt-4o", system_prompt="You find facts.")
result = await run_agent(researcher, [{"role": "user", "content": "..."}])
```

**Before (CrewAI, structured output):**
```python
researcher = Agent(role="Researcher", goal="...", output_pydantic=Report)
```

**After (deepcrew):**
```python
researcher = Agent(name="researcher", model="openai/gpt-4o", response_model=Report)
result = await run_agent(researcher, messages)
result.parsed  # a validated Report instance
```

## From Google ADK

| ADK concept | deepcrew equivalent | Notes |
|---|---|---|
| `LlmAgent(model=, instruction=)` | `Agent(model=, system_prompt=)` | Same shape; `model` is a LiteLLM string (`"openai/gpt-4o"`, `"anthropic/claude-opus-4-8"`, ...) rather than an ADK model object. |
| ADK `FunctionTool` | `@tool`-decorated function or `Skill` | Simple callables become tools; multi-step capabilities become `Skill` subclasses. |
| ADK callbacks (`before_tool_callback`, etc.) | `AgentHooks` + the `StreamEvent` queue | Hooks *intercept* (can deny a tool call); events only *observe*. Use `StreamPolicy` to control what a UI sees. |
| ADK `Session` / state | `MemoryProvider` (`InMemoryProvider`, `FileMemoryProvider`, `RedisMemoryProvider`) | Pluggable backend; attach via `Agent(memory=...)`. |
| ADK `SequentialAgent` / `ParallelAgent` | `WorkflowBuilder` | Explicit `.then()` edges; independent nodes at the same DAG level run in parallel automatically. |
| ADK `LoopAgent` | `LoopConfig` + `run_agent_loop` | deepcrew's loop is verifier-driven (a critic scores each iteration) rather than a fixed iteration count, with optional adaptive early-stop and self-consistency branching. |

**Before (ADK):**
```python
from google.adk.agents import LlmAgent

agent = LlmAgent(model="gemini-2.0-flash", name="assistant", instruction="You are helpful.")
```

**After (deepcrew):**
```python
from deepcrew import Agent, run_agent

agent = Agent(name="assistant", model="gemini/gemini-2.0-flash", system_prompt="You are helpful.")
result = await run_agent(agent, [{"role": "user", "content": "Hello!"}])
```

## What deepcrew adds

- **True token streaming with selectable visibility** — every agent, tool call, memory op, retry,
  and verifier score is a `StreamEvent`. `StreamPolicy.chat()` / `.standard()` / `.verbose()` (or a
  custom include/exclude set) control what a given UI actually sees, without changing execution.
- **Self-improving loop** — `LoopConfig` with a `Verifier` critiques each iteration and drives
  refinement, with optional adaptive early-stopping and self-consistency branching across parallel
  candidates.
- **Bounded recursive spawning** — agents can dynamically spawn sub-agents mid-run via a
  `spawn_agent` meta-tool, capped by a hard `max_spawn_depth` so delegation can't recurse forever.
- **Skill distillation** — a converged, high-confidence loop result can be distilled into a
  replayable `Skill` and registered for reuse, Voyager-style.
- **Multimodal input** — `image()` / `pdf()` / `user_message()` attach images and documents as
  standard OpenAI-format content blocks, forwarded by LiteLLM to whichever provider you're using.

See the [Streaming guide](streaming.html) and [Loop guide](guides/loop.html) for details.
