# deepcrew-ai Features

Every feature has its own self-contained page under `docs/guides/`, each with a description, a
working code example, and a "Copy prompt" button that copies a ready-to-paste AI implementation
prompt for that specific feature.

## Orchestration

- [APEX Synthesizer](guides/apex.md) — confidence-scored, citation-aware multi-agent result synthesis.
- [Agent Spawning](guides/spawning.md) — agents dynamically spawn sub-agents mid-loop, with bounded nested delegation.

## Self-Improving Loop

- [Looping](guides/looping.md) — outer iteration loop for search-refine patterns, with self-consistency branching.
- [Verifier](guides/verifier.md) — structured, LLM-graded critique drives targeted refinement, with an adaptive compute budget.
- [Procedural Memory](guides/procedural-memory.md) — an ACE-inspired evolving playbook of strategies accumulated across runs.

## Input & Streaming

- [Multimodal Input](guides/multimodal.md) — attach images and PDFs to any message with `image()`/`pdf()`.
- [StreamPolicy](guides/streampolicy.md) — choose exactly which event types a consumer sees.
- [FastAPI Integration](guides/fastapi.md) — one call turns any Agent/Orchestrator/WorkflowBuilder into an SSE endpoint.

## Agent Behavior

- [Structured Output](guides/structured-output.md) — `response_model` validates the final answer against a pydantic schema.
- [Human-in-the-Loop Hooks](guides/hooks.md) — `AgentHooks.approve_tool` can deny individual tool calls before they run.

## Memory & Extensibility

- [Memory Providers](guides/memory.md) — pluggable context stores auto-injected into each LLM call.
- [Redis Memory Provider](guides/redis-memory.md) — a persistent, shared `MemoryProvider` backed by Redis.
- [Skills](guides/skills.md) — reusable capability bundles, self-evolving from converged loop runs.

## Infrastructure

- [Retry & Fallback](guides/retry.md) — per-agent exponential backoff with model fallback chains.
- [Observability](guides/observability.md) — OpenTelemetry spans for every LLM call, tool execution, and workflow step.
- [CLI](guides/cli.md) — `deepcrew run workflow.yaml` for declarative workflow execution.

See also the [migration guide](guides/migration.md) if you're coming from CrewAI or Google ADK.
