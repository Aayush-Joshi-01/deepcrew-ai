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

Automated orchestration with APEX synthesis::

    orch = Orchestrator(
        agents=[researcher, analyst, writer],
        router_model="openai/gpt-4o-mini",
        apex_model="openai/gpt-4o",
        apex_config=ApexConfig(cite_sources=True),
        enable_spawn=True,
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
from .apex import APEXSynthesizer, ApexCitation, ApexConfig
from .exceptions import (
    DeepCrewError,
    DeepCrewMemoryError,
    LoopConvergedError,
    MCPError,
    MaxTurnsError,
    RouterError,
    SkillError,
    ToolError,
    WorkflowError,
)
from .loop import LoopConfig, LoopState, run_agent_loop, search_loop
from .memory import FileMemoryProvider, InMemoryProvider, MemoryProvider
from .observability import ObservabilityConfig
from .orchestrator import Orchestrator
from .retry import FallbackChain, RetryPolicy
from .runner import run_agent
from .skills import FunctionSkill, Skill, SkillRegistry, skill
from .skills.builtin import CodeExecutionSkill, SummarizeSkill, WebSearchSkill
from .spawner import SpawnRequest, ToolAllocator, make_spawn_tool, spawn_agent
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

__version__ = "0.2.0"

__all__ = [
    # Core
    "Agent",
    "run_agent",
    "Orchestrator",
    "WorkflowBuilder",
    # APEX
    "APEXSynthesizer",
    "ApexConfig",
    "ApexCitation",
    # Loop / Search
    "LoopConfig",
    "LoopState",
    "run_agent_loop",
    "search_loop",
    # Memory
    "MemoryProvider",
    "InMemoryProvider",
    "FileMemoryProvider",
    # Retry / Fallback
    "RetryPolicy",
    "FallbackChain",
    # Observability
    "ObservabilityConfig",
    # Skills
    "Skill",
    "skill",
    "FunctionSkill",
    "SkillRegistry",
    "WebSearchSkill",
    "SummarizeSkill",
    "CodeExecutionSkill",
    # Spawner
    "SpawnRequest",
    "ToolAllocator",
    "spawn_agent",
    "make_spawn_tool",
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
    "LoopConvergedError",
    "SkillError",
    "DeepCrewMemoryError",
]
