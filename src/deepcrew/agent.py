from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from .tools import fn_to_tool_def
from .types import ToolDef

if TYPE_CHECKING:
    from .mcp.base import MCPClient


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

    async def get_tool_defs(self) -> list[ToolDef]:
        """Discover and merge tools from all attached MCP servers + Python functions."""
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

        return defs
