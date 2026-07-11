from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .tools import fn_to_tool_def
from .types import ToolDef

if TYPE_CHECKING:
    from .loop import LoopConfig
    from .mcp.base import MCPClient
    from .memory.base import MemoryProvider
    from .procedural_memory import ProceduralMemory
    from .retry import FallbackChain, RetryPolicy
    from .skills.base import Skill


@dataclass
class Agent:
    """
    Defines a single AI agent: model, system prompt, and its tools.

    Parameters
    ----------
    name:
        Unique identifier used in logs and result objects.
    model:
        LiteLLM model string, e.g. ``"openai/gpt-4o"``,
        ``"anthropic/claude-opus-4-8"``, ``"gemini/gemini-2.0-flash"``.
    system_prompt:
        The agent's system/instruction prompt.
    mcps:
        MCP clients (StdioMCP, SSEMCP, or HTTPMCP) whose tools this agent
        can use.
    tools:
        Plain Python callables decorated with @tool (or undecorated).
        Schemas are auto-generated from type hints.
    max_turns:
        Maximum LLM→tool→LLM cycles before raising MaxTurnsError.
    temperature:
        Sampling temperature forwarded to the LLM. None uses provider default.
    max_tokens:
        Maximum output tokens. None uses provider default.
    extra_params:
        Any additional kwargs forwarded verbatim to ``litellm.acompletion``.
    retry_policy:
        Retry configuration for transient LLM failures.
    fallback_chain:
        Ordered list of fallback model strings if all retries fail.
    memory:
        Pluggable memory provider for context injection and storage.
    loop_config:
        Outer iteration loop configuration (refinement loop over full runs).
    skills:
        Higher-level reusable capability bundles exposed to the LLM as tools.
    procedural_memory:
        Optional durable, evolving playbook (see ``ProceduralMemory``). When
        set, accumulated strategies are read and injected into context on
        every run of this agent (single-shot or looped). Writing/curating
        new strategies only happens via a loop whose ``LoopConfig`` also has
        ``procedural_memory`` configured (typically the same instance).
    """

    name: str
    model: str
    system_prompt: str = ""
    mcps: list[MCPClient] = field(default_factory=list)
    tools: list[Callable[..., Any]] = field(default_factory=list)
    max_turns: int = 10
    temperature: float | None = None
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    retry_policy: RetryPolicy | None = field(default=None)
    fallback_chain: FallbackChain | None = field(default=None)
    memory: MemoryProvider | None = field(default=None)
    loop_config: LoopConfig | None = field(default=None)
    skills: list[Skill] = field(default_factory=list)
    procedural_memory: ProceduralMemory | None = field(default=None)

    async def get_tool_defs(self) -> list[ToolDef]:
        """Discover and merge tools from all attached MCP servers, Python functions, and skills."""
        defs: list[ToolDef] = []

        if self.mcps:
            results = await asyncio.gather(
                *[mcp.list_tools() for mcp in self.mcps],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, list):
                    defs.extend(r)

        for fn in self.tools:
            defs.append(fn_to_tool_def(fn))

        for sk in self.skills:
            defs.append(sk.to_tool_def())

        return defs
