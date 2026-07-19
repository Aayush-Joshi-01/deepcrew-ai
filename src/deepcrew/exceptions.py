from __future__ import annotations

from typing import Any


class DeepCrewError(Exception):
    """Base exception for all deepcrew errors."""


class MCPError(DeepCrewError):
    """Raised when an MCP server operation fails."""


class ToolError(DeepCrewError):
    """Raised when a tool call fails or a tool is not found."""


class MaxTurnsError(DeepCrewError):
    """Raised when an agent exceeds its max_turns limit without finishing."""


class RouterError(DeepCrewError):
    """Raised when the orchestrator router fails to parse a routing decision."""


class WorkflowError(DeepCrewError):
    """Raised for invalid workflow configurations (e.g. cycles in the DAG)."""


class LoopConvergedError(DeepCrewError):
    """Raised when a loop's stop_condition returns True (successful early exit)."""

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class SkillError(DeepCrewError):
    """Raised when a skill's execution fails."""


class DeepCrewMemoryError(DeepCrewError):
    """Raised when a memory provider operation fails."""


class ContentError(DeepCrewError):
    """Raised for invalid, unreadable, oversized, or unsupported multimodal content."""


class OutputParseError(DeepCrewError):
    """Raised when an agent's ``response_model`` output could not be parsed
    into the target schema, even after one automatic repair attempt."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text
