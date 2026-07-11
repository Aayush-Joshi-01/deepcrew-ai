"""
Automated orchestrator example.

The router LLM decides at runtime whether to use one agent or fan out to
multiple agents in parallel. The synthesizer merges parallel results.

Requires: OPENAI_API_KEY
Run: python examples/automated_example.py
"""

import asyncio

from deepcrew import Agent, Orchestrator
from deepcrew.types import EventType


async def main():
    agents = [
        Agent(
            name="researcher",
            model="openai/gpt-4o-mini",
            system_prompt=(
                "You are a research specialist. Gather comprehensive facts, "
                "statistics, and background information on any given topic."
            ),
        ),
        Agent(
            name="analyst",
            model="openai/gpt-4o-mini",
            system_prompt=(
                "You are a critical analyst. Evaluate trends, implications, "
                "risks, and opportunities related to any given topic."
            ),
        ),
        Agent(
            name="writer",
            model="openai/gpt-4o-mini",
            system_prompt=(
                "You are a professional writer. When given a task, produce "
                "clear, engaging, well-structured written content."
            ),
        ),
    ]

    orch = Orchestrator(
        agents=agents,
        router_model="openai/gpt-4o-mini",
        synthesizer_model="openai/gpt-4o",
        max_parallel_agents=3,
    )

    query = "What are the key challenges and opportunities of AI in healthcare?"
    print(f"Query: {query!r}\n")

    # --- Streaming ---
    print("=== Streaming ===")
    active_agents = set()
    async for event in orch.stream(query):
        if event.event == EventType.AGENT_START:
            active_agents.add(event.agent_id)
            print(f"[+] Agent started: {event.agent_id} ({event.data.get('model')})")
        elif event.event == EventType.TEXT_DELTA:
            print(event.data["chunk"], end="", flush=True)
        elif event.event == EventType.AGENT_DONE:
            print(
                f"\n[-] Agent done: {event.agent_id} "
                f"({event.data['input_tokens']}in/{event.data['output_tokens']}out)"
            )
        elif event.event == EventType.DONE:
            print("\n\n=== Final Answer ===")
            print(event.data.get("final_text", ""))
        elif event.event == EventType.ERROR:
            print(f"\n[ERROR] {event.data.get('message')}")

    print(f"\nAgents used: {active_agents}")

    # --- Non-streaming ---
    print("\n\n=== Non-streaming ===")
    result = await orch.run("Give me a one-sentence summary of quantum computing.")
    print(f"Answer: {result.final_text}")
    print(f"Total tokens: {result.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
