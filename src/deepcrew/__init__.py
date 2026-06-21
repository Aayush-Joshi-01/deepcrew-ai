"""
deepcrew-ai — Multi-agent spawning library.

Quick start::

    from deepcrew import Agent, run_agent, Orchestrator, WorkflowBuilder, tool
    from deepcrew.mcp import StdioMCP, SSEMCP, HTTPMCP, MCPManager

Single agent::

    agent = Agent(name="assistant", model="openai/gpt-4o",
                  system_prompt="You are helpful.")
    result = await run_agent(agent, [{"role": "user", "content": "Hello!"}])
    print(result.text)

Automated orchestration (router → parallel agents → synthesizer)::

    orch = Orchestrator(
        agents=[researcher, analyst, writer],
        router_model="openai/gpt-4o-mini",
        synthesizer_model="openai/gpt-4o",
    )
    result = await orch.run("Explain quantum entanglement")

DAG workflow::

    workflow = (
        WorkflowBuilder()
        .add_agent("research", researcher, task="{input}")
        .add_agent("write", writer, task="Write about:\\n{research}")
        .then("research", "write")
    )
    result = await workflow.run("Quantum computing trends")
"""

from .agent import Agent
from .exceptions import (
    DeepCrewError,
    MCPError,
    MaxTurnsError,
    RouterError,
    ToolError,
    WorkflowError,
)
from .orchestrator import Orchestrator
from .runner import run_agent
from .stream import make_done_event, make_error_event, queue_to_stream
from .tools import fn_to_tool_def, tool
from .types import (
    AgentResult,
    EventType,
    OrchestratorResult,
    StreamEvent,
    ToolDef,
    WorkflowResult,
)
from .workflow import WorkflowBuilder

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "run_agent",
    "Orchestrator",
    "WorkflowBuilder",
    # Tools
    "tool",
    "fn_to_tool_def",
    # Types
    "StreamEvent",
    "EventType",
    "AgentResult",
    "WorkflowResult",
    "OrchestratorResult",
    "ToolDef",
    # Streaming
    "queue_to_stream",
    "make_done_event",
    "make_error_event",
    # Exceptions
    "DeepCrewError",
    "MCPError",
    "ToolError",
    "MaxTurnsError",
    "RouterError",
    "WorkflowError",
]
