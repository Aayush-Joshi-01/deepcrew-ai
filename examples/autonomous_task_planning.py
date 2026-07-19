"""
Autonomous task planning via bounded nested agent spawning.

With enable_spawn=True, every agent the Orchestrator runs gets a spawn_agent
meta-tool it can call mid-loop to delegate a sub-task to a fresh agent.
max_spawn_depth caps how deep that delegation chain can go.

Requires: OPENAI_API_KEY environment variable.
Run: python examples/autonomous_task_planning.py
"""

import asyncio

from deepcrew import Agent, Orchestrator, fn_to_tool_def, tool
from deepcrew.types import EventType


@tool
def search_docs(query: str) -> str:
    """Search internal documentation for a topic (mock implementation)."""
    return f"Docs for {query!r}: (placeholder search results)"


@tool
def draft_section(topic: str) -> str:
    """Draft a short section of a report on the given topic (mock implementation)."""
    return f"Draft section on {topic!r}: (placeholder draft content)"


async def main():
    planner = Agent(
        name="planner",
        model="openai/gpt-4o",
        system_prompt=(
            "You are a planning agent. Break a large task into sub-tasks and use "
            "spawn_agent to delegate each sub-task to a fresh agent with the right tools."
        ),
    )

    orch = Orchestrator(
        agents=[planner],
        router_model="openai/gpt-4o-mini",
        global_tools=[fn_to_tool_def(search_docs), fn_to_tool_def(draft_section)],
        enable_spawn=True,
        max_spawn_depth=2,
    )

    query = "Research and draft a two-section report on vector database indexing strategies."

    async for event in orch.stream(query):
        if event.event == EventType.SPAWN_AGENT:
            print(f"[spawn] task={event.data.get('task')!r} depth={event.data.get('depth')}")
        elif event.event == EventType.TEXT_DELTA:
            print(event.data["chunk"], end="", flush=True)
        elif event.event == EventType.DONE:
            print("\n\n=== Final ===")
            print(event.data.get("final_text", ""))


if __name__ == "__main__":
    asyncio.run(main())
