# Changelog

All notable changes to `deepcrew-ai` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-06-29

### Added

#### APEX Synthesizer
- `APEXSynthesizer` — replaces the plain "synthesizer" with an intelligent synthesis engine that produces a confidence score (0.0–1.0) on every result
- `ApexConfig` — fine-grained control: `confidence_threshold`, `cite_sources` (adds `[source: agent_name]` inline markers), `allow_tools` (APEX can call tools mid-synthesis), `system_prompt` override
- `ApexCitation` — dataclass representing a fact attributed to a specific source agent
- `Orchestrator` gains `apex_model` and `apex_config` parameters; `synthesizer_model` kept as a backward-compatible alias
- `AgentResult.confidence` — new optional float field populated by APEX
- `EventType.APEX_START` and `APEX_DONE` — bracket the synthesis step in the event stream

#### Intelligent Agent Spawning
- `ToolAllocator` — asks the router LLM to select the most relevant subset of tools from a global pool for a given task
- `SpawnRequest` — dataclass describing a dynamic sub-agent spawn (task, tools, model, system_prompt, max_turns)
- `spawn_agent()` — dynamically creates and runs a sub-agent with intelligently allocated tools; emits `SPAWN_AGENT` event
- `make_spawn_tool()` — returns a `ToolDef` named `spawn_agent` that can be injected into any running agent so it can spawn children mid-loop (Claude Code-style)
- `Orchestrator` gains `global_tools` (pool available for allocation) and `enable_spawn` (inject spawn meta-tool into every agent) parameters
- When `global_tools` is set, the router assigns relevant tool subsets per agent (intelligent allocation)
- `EventType.SPAWN_AGENT`

#### Looping Methodology
- `LoopConfig` — controls the outer iteration loop: `max_iterations`, `convergence_fn`, `stop_condition`, `refine_prompt`
- `run_agent_loop()` — outer loop that calls `run_agent()` repeatedly, appending a refinement prompt when the result does not satisfy `convergence_fn`
- `search_loop()` — convenience wrapper for iterative search/refine patterns; loops until `result.confidence >= threshold`
- `Agent.loop_config` — new optional field; `run_agent()` auto-delegates to `run_agent_loop()` transparently
- `LoopState` — tracks current iteration and accumulated results
- `LoopConvergedError` — raised by `stop_condition`; carries `.result` for clean early-exit handling
- `AgentResult.loop_iterations` — records how many outer iterations ran
- `EventType.LOOP_ITERATION`

#### Skills
- `Skill` — abstract base class for higher-level reusable capability bundles; surfaces to the LLM identically to tools via `to_tool_def()`
- `@skill` decorator — wraps an async function into a `FunctionSkill` instance (mirrors `@tool`)
- `FunctionSkill` — concrete `Skill` subclass created by the decorator
- `SkillRegistry` — class-level registry for named skill lookup (`register`, `get`, `list_all`, `clear`)
- `Agent.skills` — new list field; `get_tool_defs()` automatically appends `skill.to_tool_def()` for each skill
- **Built-in skills**:
  - `WebSearchSkill` — DuckDuckGo Instant Answer API search
  - `SummarizeSkill` — LLM-backed text summarization (configurable model)
  - `CodeExecutionSkill` — sandboxed Python execution in a subprocess (10 s timeout)
- `SkillError` exception

#### Memory Providers
- `MemoryProvider` — abstract base class: `store(key, value)`, `retrieve(key)`, `search(query, top_k)`, `clear()`
- `InMemoryProvider` — dict-based short-term store; no extra dependencies
- `FileMemoryProvider(path)` — JSON-backed persistent store; atomic write-then-rename to prevent corruption
- `Agent.memory` — new optional field; runner auto-injects relevant memories before LLM calls and stores tool results after
- `EventType.MEMORY_STORE` and `MEMORY_RETRIEVE` emitted at each injection/storage point
- `DeepCrewMemoryError` exception

#### Retry & Fallback Policies
- `RetryPolicy` — per-agent retry configuration: `max_retries`, `backoff_seconds`, `retry_on`, `exponential`
- `FallbackChain` — ordered list of model strings to try when all retries for the current model fail
- `with_retry_and_fallback()` — internal utility; wraps an LLM call factory `(model) → Coroutine` with full retry and model-switching logic
- `Agent.retry_policy` and `Agent.fallback_chain` — new optional fields; runner wraps every `litellm.acompletion` call when either is set
- `EventType.RETRY_ATTEMPT` and `FALLBACK_TRIGGERED`

#### Observability (OpenTelemetry)
- `ObservabilityConfig` — `otel_endpoint`, `service_name`, `enabled`, `export_format` (`"grpc"` or `"http"`)
- `get_tracer(config)` — lazy-imports `opentelemetry-sdk`; raises a helpful `ImportError` if the optional extra is not installed
- Span context managers: `agent_span`, `llm_span`, `tool_span`, `workflow_step_span` — all return `nullcontext()` when `observability=None` (zero overhead)
- `run_agent()` gains `observability` parameter; wraps agent run, each LLM call, and each tool execution in OTel spans with standard attributes (`agent.id`, `llm.model`, `tool.name`, token counts)
- `WorkflowBuilder` gains `observability` parameter; wraps each DAG node in a `workflow.step` span
- Optional install: `pip install deepcrew-ai[otel]`

#### CLI
- `deepcrew run workflow.yaml` — load and execute a declarative YAML workflow file
- `deepcrew agents list --config workflow.yaml` — print a formatted table of agents from a config
- `deepcrew --version` — print library version
- `WorkflowYAML`, `AgentYAML`, `WorkflowStepYAML` — Pydantic v2 models for YAML validation
- `pyyaml>=6.0` added as a core dependency (lightweight)

#### New Event Types (v0.2.0)
`RETRY_ATTEMPT`, `FALLBACK_TRIGGERED`, `MEMORY_STORE`, `MEMORY_RETRIEVE`, `LOOP_ITERATION`, `SPAWN_AGENT`, `APEX_START`, `APEX_DONE`

#### New Exceptions (v0.2.0)
`LoopConvergedError` (carries `.result`), `SkillError`, `DeepCrewMemoryError`

#### Optional Dependency Groups
- `deepcrew-ai[otel]` — `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`
- `deepcrew-ai[redis]` — `redis>=5.0.0` (scaffolded for future `RedisMemoryProvider`)

### Changed
- `Orchestrator._synthesize()` now delegates to `APEXSynthesizer`; `synthesizer_model` kept as a backward-compatible alias for `apex_model`
- `run_agent()` gains optional `observability` parameter (default `None`; no behavior change when omitted)
- `WorkflowBuilder.__init__` gains optional `observability` parameter (default `None`)
- `Agent` gains five new optional fields: `retry_policy`, `fallback_chain`, `memory`, `loop_config`, `skills` — all default to `None`/`[]`, fully backward compatible
- `AgentResult` gains `confidence: float | None` and `loop_iterations: int` fields (both default-safe)
- GitHub/PyPI URLs updated to `github.com/Aayush-Joshi-01/deepcrew-ai`

### Fixed
- `pyproject.toml` version bump to `0.2.0`
- `deepcrew.__version__` updated to `"0.2.0"`

---

## [0.1.0] — 2026-06-21

### Added

#### Core
- `Agent` dataclass — define name, model (litellm string), system prompt, MCP clients, Python function tools, temperature, max tokens, and arbitrary `extra_params` forwarded to litellm
- `run_agent()` — async agentic loop: LLM → parallel tool execution → LLM, repeating until no tool calls or `max_turns` exceeded; streams `StreamEvent` objects to an optional `asyncio.Queue`
- `MaxTurnsError` raised when an agent exhausts its turn budget

#### Orchestrator (automated mode)
- `Orchestrator` — three-stage pipeline: router LLM decides single-agent vs. parallel fan-out, agents run via `asyncio.gather`, synthesizer LLM merges results
- `Orchestrator.run()` — returns `OrchestratorResult` with final text, per-agent results, and token counts
- `Orchestrator.stream()` — async generator yielding `StreamEvent` objects in real time
- Built-in router and synthesizer system prompts; both overridable
- `max_parallel_agents` cap to bound fan-out

#### Workflow Builder (explicit DAG mode)
- `WorkflowBuilder` — fluent API to declare agent nodes and dependency edges
- `.add_agent(name, agent, task)` — add a node with a string template (`{input}`, `{node_name}`) or callable task
- `.then(predecessor, successor)` — declare ordering constraint
- `.run()` / `.stream()` — execute the DAG; independent nodes at each topological level run in parallel via `asyncio.gather`
- `WorkflowResult` with per-node `AgentResult` outputs and aggregate token counts
- `WorkflowError` on cycle detection or unknown node references

#### Python Function Tools
- `@tool` decorator — marks any callable as an agent tool; optional `name` and `description` overrides
- `fn_to_tool_def()` — converts any Python function to a `ToolDef` with JSON Schema generated from type hints
- Google-style docstring parameter descriptions extracted automatically

#### MCP Integration
- `MCPClient` abstract base with async context manager support
- `StdioMCP` — subprocess JSON-RPC over stdin/stdout
- `HTTPMCP` — modern streamable-HTTP with session management and exponential-backoff retries
- `SSEMCP` — legacy SSE transport
- `MCPManager` — aggregates multiple MCP clients with parallel discovery and auto-routing

#### Streaming & Types
- `StreamEvent`, `EventType`, `AgentResult`, `WorkflowResult`, `OrchestratorResult`, `ToolDef`
- `queue_to_stream()`, `make_done_event()`, `make_error_event()`

#### Package
- hatchling build; Python ≥ 3.11
- 37 unit tests; all passing on Python 3.13
- 4 runnable examples

[0.2.0]: https://github.com/Aayush-Joshi-01/deepcrew-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Aayush-Joshi-01/deepcrew-ai/releases/tag/v0.1.0
