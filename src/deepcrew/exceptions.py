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
