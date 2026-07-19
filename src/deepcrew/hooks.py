"""
Lifecycle hooks for human-in-the-loop control over a single Agent run.

Hooks *intercept and can alter* execution (``approve_tool`` returning False
blocks a tool call outright). This is different from the observe-only
:class:`~deepcrew.StreamEvent` queue and :class:`~deepcrew.StreamPolicy`,
which only ever describe what happened -- they never change it.

Example
-------
::

    async def approve(tool_name: str, args: dict) -> bool:
        return tool_name != "delete_file"  # block one specific tool

    agent = Agent(
        name="assistant",
        model="openai/gpt-4o",
        tools=[delete_file, read_file],
        hooks=AgentHooks(approve_tool=approve),
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentHooks:
    """
    Per-agent lifecycle callbacks. Any hook left as ``None`` is a no-op.

    A hook that raises is caught and logged (WARNING) -- it never crashes the
    run. ``approve_tool`` is the exception: if it returns ``False`` (or
    raises), the tool call is denied and never executed.
    """

    on_agent_start: Callable[[], Awaitable[None]] | None = None
    on_tool_start: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    on_tool_end: Callable[[str, str], Awaitable[None]] | None = None
    approve_tool: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None
