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

import logging
from typing import Any

from .agent import Agent
from .apex import ApexCitation, ApexConfig, APEXSynthesizer
from .content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    TextPart,
    describe_attachments,
    extract_text,
    image,
    pdf,
    user_message,
)
from .exceptions import (
    ContentError,
    DeepCrewError,
    DeepCrewMemoryError,
    LoopConvergedError,
    MaxTurnsError,
    MCPError,
    OutputParseError,
    RouterError,
    SkillError,
    ToolError,
    WorkflowError,
)
from .hooks import AgentHooks
from .loop import LoopConfig, LoopState, run_agent_loop, search_loop
from .memory import FileMemoryProvider, InMemoryProvider, MemoryProvider
from .observability import ObservabilityConfig
from .orchestrator import Orchestrator
from .procedural_memory import PlaybookEntry, ProceduralMemory
from .retry import FallbackChain, RetryPolicy
from .runner import run_agent
from .skills import FunctionSkill, Skill, SkillRegistry, skill
from .skills.builtin import CodeExecutionSkill, SummarizeSkill, WebSearchSkill
from .spawner import SpawnRequest, ToolAllocator, make_spawn_tool, spawn_agent
from .stream import StreamPolicy, filter_stream, make_done_event, make_error_event, queue_to_stream
from .tools import fn_to_tool_def, tool
from .types import (
    AgentResult,
    EventType,
    OrchestratorResult,
    StreamEvent,
    ToolDef,
    WorkflowResult,
)
from .verifier import Verifier, VerifierConfig, VerifierFeedback
from .workflow import WorkflowBuilder

logging.getLogger("deepcrew").addHandler(logging.NullHandler())

__version__ = "0.4.0"

__all__ = [
    # APEX
    "APEXSynthesizer",
    # Core
    "Agent",
    "AgentHooks",
    "AgentResult",
    "ApexCitation",
    "ApexConfig",
    "CodeExecutionSkill",
    # Exceptions
    "ContentError",
    # Multimodal content
    "ContentPart",
    "DeepCrewError",
    "DeepCrewMemoryError",
    "DocumentPart",
    "EventType",
    "FallbackChain",
    "FileMemoryProvider",
    "FunctionSkill",
    "ImagePart",
    "InMemoryProvider",
    # Loop / Search
    "LoopConfig",
    "LoopConvergedError",
    "LoopState",
    "MCPError",
    "MaxTurnsError",
    # Memory
    "MemoryProvider",
    # Observability
    "ObservabilityConfig",
    "Orchestrator",
    "OrchestratorResult",
    "OutputParseError",
    "PlaybookEntry",
    "ProceduralMemory",
    # Integrations / optional-dependency providers (lazy)
    "RedisMemoryProvider",
    # Retry / Fallback
    "RetryPolicy",
    "RouterError",
    # Skills
    "Skill",
    "SkillError",
    "SkillRegistry",
    # Spawner
    "SpawnRequest",
    # Types
    "StreamEvent",
    # Streaming
    "StreamPolicy",
    "SummarizeSkill",
    "TextPart",
    "ToolAllocator",
    "ToolDef",
    "ToolError",
    # Verifier
    "Verifier",
    "VerifierConfig",
    "VerifierFeedback",
    "WebSearchSkill",
    "WorkflowBuilder",
    "WorkflowError",
    "WorkflowResult",
    "create_stream_router",
    "describe_attachments",
    "extract_text",
    "filter_stream",
    "fn_to_tool_def",
    "image",
    "make_done_event",
    "make_error_event",
    "make_spawn_tool",
    "pdf",
    "queue_to_stream",
    "run_agent",
    "run_agent_loop",
    "search_loop",
    "skill",
    "spawn_agent",
    # Tools
    "tool",
    "user_message",
]


def __getattr__(name: str) -> Any:
    """Lazily expose optional-dependency integrations without importing them eagerly."""
    if name == "create_stream_router":
        from .integrations.fastapi import create_stream_router

        return create_stream_router
    if name == "RedisMemoryProvider":
        from .memory.redis_provider import RedisMemoryProvider

        return RedisMemoryProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
