# Changelog

All notable changes to `deepcrew-ai` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- `fn_to_tool_def()` — converts any Python function to a `ToolDef` with JSON Schema generated from type hints (`str`, `int`, `float`, `bool`, `list[X]`, `dict[str, X]`, `Optional[X]`, `Union`, `Literal`)
- Google-style docstring parameter descriptions extracted automatically

#### MCP Integration
- `MCPClient` abstract base class with `connect()`, `list_tools()`, `call_tool()`, `close()`, and async context manager support
- `StdioMCP` — spawns a subprocess and communicates via JSON-RPC over stdin/stdout; supports MCP `2024-11-05` initialize handshake
- `HTTPMCP` — modern streamable-HTTP MCP transport with `Mcp-Session-Id` session management, automatic session re-initialization on `400/401`, and exponential-backoff retries
- `SSEMCP` — legacy SSE transport; connects to `/sse`, receives the `endpoint` event, posts JSON-RPC to the resolved URL
- `MCPManager` — aggregates multiple MCP clients; discovers tools in parallel and maintains a routing table for `call_tool()` dispatch; implements `MCPClient` interface so it can be passed directly to `Agent.mcps`

#### Streaming
- `StreamEvent` dataclass with `event` (`EventType` enum), `agent_id`, and `data` dict
- `StreamEvent.to_sse()` — produces a ready-to-stream Server-Sent Event string
- `queue_to_stream()` — drains an `asyncio.Queue[StreamEvent | None]` as an async generator
- `EventType` values: `agent_start`, `text_delta`, `thinking_delta`, `tool_call`, `tool_result`, `agent_done`, `step_start`, `step_done`, `error`, `done`

#### Types & Exceptions
- `AgentResult`, `WorkflowResult`, `OrchestratorResult`, `ToolDef`
- `DeepCrewError`, `MCPError`, `ToolError`, `MaxTurnsError`, `RouterError`, `WorkflowError`

#### Package
- `pyproject.toml` with hatchling build backend; Python ≥ 3.11 required
- Dependencies: `litellm>=1.63.0`, `httpx>=0.27.0`, `httpx-sse>=0.4.0`, `pydantic>=2.0.0`, `anyio>=4.4.0`, `typing-extensions>=4.9.0`
- 37 unit tests (pytest-asyncio); all passing on Python 3.13
- 4 runnable examples: `simple_agent.py`, `workflow_example.py`, `automated_example.py`, `mcp_example.py`

[0.1.0]: https://github.com/deepcrew-ai/deepcrew-ai/releases/tag/v0.1.0
