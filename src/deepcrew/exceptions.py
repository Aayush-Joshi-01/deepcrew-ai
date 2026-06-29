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

    def __init__(self, message: str, result: "Any" = None) -> None:  # noqa: F821
        super().__init__(message)
        self.result = result


class SkillError(DeepCrewError):
    """Raised when a skill's execution fails."""


class DeepCrewMemoryError(DeepCrewError):
    """Raised when a memory provider operation fails."""
