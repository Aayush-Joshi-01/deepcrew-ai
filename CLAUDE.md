# CLAUDE.md

Guidance for coding agents (Claude Code and others) working in this repository.

## What this is

`deepcrew-ai` (import name `deepcrew`) is an async, LiteLLM-backed multi-agent library. Its
distinguishing features are a self-improving refinement loop (verifier-scored, adaptive,
self-consistency branching, skill distillation), bounded recursive agent spawning, APEX
(Assembly & Precision Executor) multi-agent synthesis, and true token streaming with selectable
visibility (`StreamPolicy`).

## Architecture map

- `src/deepcrew/agent.py` — `Agent` dataclass: model, system prompt, tools, memory, hooks,
  `response_model`, retry/fallback config.
- `src/deepcrew/runner.py` — `run_agent()`: the core agentic loop (stream → buffer tool calls →
  execute in parallel → repeat). Delegates to `loop.py` when `Agent.loop_config` is set.
- `src/deepcrew/orchestrator.py` — `Orchestrator`: NIA (Navigator & Intelligence Allocator), the
  router LLM, picks single-agent or parallel fan-out, then `apex.py`'s `APEXSynthesizer` (APEX,
  Assembly & Precision Executor) merges parallel results.
- `src/deepcrew/loop.py` — `run_agent_loop()`: outer refinement loop driven by `LoopConfig` —
  verifier scoring (`verifier.py`), adaptive early-stop, self-consistency branching, and
  Voyager-style skill distillation into `skills/registry.py`.
- `src/deepcrew/spawner.py` — bounded recursive sub-agent spawning (`spawn_agent` meta-tool),
  hard-capped by `max_spawn_depth`.
- `src/deepcrew/procedural_memory.py` — durable, evolving playbook layered on any `MemoryProvider`.
  Optional `reflect_model` swaps the heuristic `curate()` for one LLM call.
- `src/deepcrew/episodic_memory.py` — `EpisodicMemory` / `Episode`: a ring-buffer log of past
  runs layered on any `MemoryProvider`, with dependency-free similarity+recency `recall()`.
  `Agent.episodic_memory` drives an automatic recall-before / record-after cycle in
  `runner.py` (`_run_agent_turns` recalls; the public `run_agent` and `loop.py` record — one
  episode per loop iteration via `record_agent_episode`).
- `src/deepcrew/memory/` — `MemoryProvider` ABC + `InMemoryProvider`, `FileMemoryProvider`,
  `RedisMemoryProvider` (lazy-imported; needs the `redis` extra).
- `src/deepcrew/content.py` — multimodal input: `image()`, `pdf()`, `user_message()` build OpenAI
  content blocks; `extract_text()` is the canonical way to pull plain text out of a message whose
  `content` may be a string or a block list.
- `src/deepcrew/stream.py` — `StreamEvent`/`EventType`, `StreamPolicy` (chat/standard/verbose
  presets), `filter_stream()`. Filtering is view-only — it never changes what actually executes.
- `src/deepcrew/logging_config.py` — general-purpose `logging` layer: `LoggingConfig`,
  `configure_logging()` (named profiles dev/prod/verbose/quiet, env-var toggles
  `DEEPCREW_LOG*`), and `get_logger()` (a `StructuredAdapter` — kwargs become structured
  fields; public API for user components). Colour via the optional `rich` (`[log]` extra),
  plain ANSI fallback otherwise. Off by default (NullHandler only). Independent of OTel and
  of the `StreamEvent` queue.
- `src/deepcrew/hooks.py` — `AgentHooks`: human-in-the-loop interception (`approve_tool` can deny
  a tool call outright). Distinct from the observe-only event stream.
- `src/deepcrew/workflow.py` — `WorkflowBuilder`: explicit DAG of agents, Kahn's-algorithm level
  scheduling so independent nodes run in parallel.
- `src/deepcrew/mcp/` — MCP client transports (stdio, SSE, streamable HTTP) + `MCPManager`.
- `src/deepcrew/integrations/fastapi.py` — optional (`fastapi` extra) drop-in SSE router.
- `src/deepcrew/cli/` — `deepcrew run <workflow.yaml>` / `deepcrew agents list`.

## Commands

```
pip install -e .[dev,otel,fastapi,log]   # editable install with dev/otel/fastapi/log extras
pytest                                # run the test suite
pytest --cov=deepcrew --cov-report=xml
ruff check .                          # lint
ruff format .                         # format
mypy src/                             # type check
python -m build                       # build sdist + wheel
```

CI (`.github/workflows/ci.yml`) runs lint (ruff + mypy + a version-consistency check) on Python
3.12 and the test suite across 3.11/3.12/3.13.

## Conventions

- Async-first throughout; parallel fan-out via `asyncio.gather(..., return_exceptions=True)`.
- Dataclasses for config objects; `from __future__ import annotations` in every module.
- Custom exceptions all inherit from `DeepCrewError` (`exceptions.py`).
- Every consequential backend action emits a `StreamEvent` — if you add a new one, also decide
  whether it belongs in `StreamPolicy`'s `standard` preset (`stream.py`).
- Consequential actions should also emit a structured log via `get_logger(...)` (from
  `logging_config.py`), passing fields as kwargs (not f-strings) and an `event="..."` slug
  whose value matches the corresponding `EventType`. Keep `INFO` for lifecycle milestones,
  `DEBUG` for per-turn/per-tool detail, `TRACE` for token-level. Never use `print`.
- **The default `pytest` run never calls live LLM APIs.** It mocks `litellm.acompletion` via
  `unittest.mock.patch`/`AsyncMock`, with a hand-built `MagicMock` chunk for streaming responses
  (shared factories in `tests/conftest.py`). Opt-in live tests live in `tests/live/`
  (`pytest --live`, needs `GEMINI_API_KEY`); a content-hash ledger (`tests/live/LEDGER.*`)
  skips a feature whose watched source files are unchanged since its last live pass.
  If you add a field to a streamed chunk's `delta` that code reads with `getattr(...)` (like
  `reasoning_content`), you must also set it explicitly in every test's chunk-builder helper —
  otherwise `MagicMock`'s default truthy attribute breaks assertions across the whole suite.
- Optional dependencies (`fastapi`, `redis`, `opentelemetry`) are imported lazily inside the
  functions/classes that need them, and exposed from `deepcrew/__init__.py` via `__getattr__` —
  never at module import time. A bare `pip install deepcrew-ai` must always import cleanly.
- Version is duplicated in `pyproject.toml` (`[project].version`) and
  `src/deepcrew/__init__.py` (`__version__`) — keep them in sync; CI checks this.
- The GitHub Pages site is hand-written static HTML under `pages/` (no site generator, kept
  separate from repository documentation like this file) with markdown twins under `pages/md/`
  for the "copy page as Markdown" feature — update both together. Each feature has its own page
  under `pages/guides/` (no version split) with a "Copy prompt" button that copies a ready-to-use
  AI implementation prompt.

## Where to look first

- Adding a feature that touches the agentic loop → start in `runner.py`.
- Adding a feature that touches multi-agent coordination → start in `orchestrator.py` or
  `workflow.py`.
- Adding a memory type → mirror `procedural_memory.py` / `episodic_memory.py`: a class over a
  `MemoryProvider` with indexed keys + a `__meta__` key, wired into `runner.py` alongside the
  other memory injections and (optionally) `loop.py`.
- Verifying a feature works against a real model → add a `tests/live/` test with a
  `pytest.mark.live(feature=..., watch=[...])` marker; run `pytest --live`.
- Adding a new stream event → `types.py` (enum member) + the emission site + `stream.py`'s preset
  sets if it should show up under `StreamPolicy.standard()`.
- Anything about how implementers should *use* the library → see
  `.claude/skills/deepcrew/SKILL.md` and `pages/md/migration.md`.
