"""
Episodic memory: the agent remembers specific past runs and recalls the
relevant ones before answering a new, related question.

Set ``Agent.episodic_memory`` and the record/recall cycle is automatic — every
run recalls relevant past episodes into context beforehand and records its own
outcome afterwards.

Requires: GEMINI_API_KEY environment variable.
Run: python examples/episodic_memory.py
"""

import asyncio

from deepcrew import Agent, EpisodicMemory, run_agent
from deepcrew.memory import InMemoryProvider

MODEL = "gemini/gemini-3.5-flash-lite"


async def main():
    episodic = EpisodicMemory(InMemoryProvider(), namespace="demo", recall_k=2)
    agent = Agent(
        name="assistant",
        model=MODEL,
        system_prompt="Answer in two or three sentences.",
        episodic_memory=episodic,
        max_turns=2,
    )

    first = "In one sentence, what is a bloom filter?"
    print(f"Q1: {first}")
    r1 = await run_agent(agent, [{"role": "user", "content": first}])
    print(f"A1: {r1.text}\n")

    second = "What are the trade-offs of using a bloom filter?"
    print(f"Q2: {second}")
    recalled = await episodic.recall(second)
    print(f"(recalled {len(recalled)} past episode(s) before answering)")
    for ep in recalled:
        print(f"  - [{ep.outcome}] {ep.query}")
    r2 = await run_agent(agent, [{"role": "user", "content": second}])
    print(f"A2: {r2.text}")
    print(f"\nepisodes now stored: {len(await episodic.load())}")


if __name__ == "__main__":
    asyncio.run(main())
