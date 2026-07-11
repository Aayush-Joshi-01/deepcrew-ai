from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    AGENT_START = "agent_start"
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    AGENT_DONE = "agent_done"
    STEP_START = "step_start"
    STEP_DONE = "step_done"
    ERROR = "error"
    DONE = "done"
    # v0.2.0
    RETRY_ATTEMPT = "retry_attempt"
    FALLBACK_TRIGGERED = "fallback_triggered"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    LOOP_ITERATION = "loop_iteration"
    SPAWN_AGENT = "spawn_agent"
    APEX_START = "apex_start"
    APEX_DONE = "apex_done"
    # v0.3.0
    VERIFIER_SCORED = "verifier_scored"


@dataclass
class StreamEvent:
    """A single event emitted during agent execution."""

    event: EventType
    data: dict[str, Any]
    agent_id: str = ""

    def to_sse(self) -> str:
        """Encode as a Server-Sent Event string, ready to stream over HTTP."""
        payload = {"agent_id": self.agent_id, **self.data}
        return f"event: {self.event.value}\ndata: {json.dumps(payload)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event.value, "agent_id": self.agent_id, **self.data}


@dataclass
class ToolDef:
    """Unified tool definition used by the runner regardless of tool source."""

    name: str
    description: str
    parameters: dict[str, Any]
    _callable: Callable[..., Any] | None = field(default=None, repr=False)
    _mcp_client: Any | None = field(default=None, repr=False)


@dataclass
class AgentResult:
    """Result produced by a single agent after its agentic loop completes."""

    agent_id: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    confidence: float | None = None
    loop_iterations: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class WorkflowResult:
    """Result produced by a completed workflow run."""

    outputs: dict[str, AgentResult]
    final_output: AgentResult | None
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class OrchestratorResult:
    """Result produced by an automated orchestrator run."""

    final_text: str
    agent_results: list[AgentResult] = field(default_factory=list)
    router_result: AgentResult | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
