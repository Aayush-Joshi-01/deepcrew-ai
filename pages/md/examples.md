# deepcrew-ai Examples

Runnable, self-contained scripts live in [`examples/`](https://github.com/Aayush-Joshi-01/deepcrew-ai/tree/main/examples)
in the repository. Each file's module docstring lists the required environment variables (API
keys) before you run it.

## Core

- `simple_agent.py` — a single agent with an `@tool`-decorated function.
- `workflow_example.py` — a 4-node DAG (`WorkflowBuilder`) with streaming and non-streaming runs.
- `automated_example.py` — `Orchestrator` with router + APEX synthesis, streaming events.
- `mcp_example.py` — all three MCP transports (Stdio/HTTP/SSE) plus `MCPManager`.

## Self-improving loop / spawning

- `self_improving_research.py` — `LoopConfig` with a `Verifier`, adaptive early-stop, and
  procedural memory.
- `consensus_code_review.py` — self-consistency branching plus skill distillation.
- `autonomous_task_planning.py` — bounded nested agent spawning via `Orchestrator(enable_spawn=True)`.

## v0.4.0

- `multimodal_agent.py` — attaching an image and a PDF to a query with `image()`/`pdf()`.
- `structured_output.py` — `response_model` validated against a pydantic schema.
- `human_in_the_loop.py` — `AgentHooks.approve_tool` denying a specific tool call.
- `fastapi_streaming.py` — `create_stream_router` wired into a FastAPI app, chat vs. verbose
  `StreamPolicy`.

Provider quick reference: swap the `model=` string on any `Agent` —
`"openai/gpt-4o"`, `"anthropic/claude-opus-4-8"`, `"gemini/gemini-2.0-flash"`,
`"bedrock/anthropic.claude-3-sonnet"`, `"azure/<deployment>"`, `"groq/llama-3.1-70b"`,
`"ollama/llama3.2"` (local, no API key needed) — LiteLLM handles the rest.

Full walkthroughs with expected output: [examples.html](examples.html).
