"""
Self-improving research agent.

A Verifier grades each iteration's answer (score + specific issues + a concrete
suggestion) and drives targeted refinement. adaptive=True stops early once
the score plateaus, instead of always running max_iterations.

Requires: OPENAI_API_KEY environment variable.
Run: python examples/self_improving_research.py
"""

import asyncio

from deepcrew import Agent, LoopConfig, Verifier, VerifierConfig, run_agent
from deepcrew.types import EventType


async def main():
    agent = Agent(
        name="researcher",
        model="openai/gpt-4o",
        system_prompt="You are a thorough research assistant. Cite specifics, not generalities.",
        loop_config=LoopConfig(
            max_iterations=5,
            verifier=Verifier(
                VerifierConfig(
                    threshold=0.85,
                    rubric="Answer must include at least two concrete, named examples.",
                )
            ),
            adaptive=True,
            min_improvement=0.02,
            plateau_patience=2,
        ),
    )

    messages = [
        {"role": "user", "content": "What are the main approaches to reducing LLM hallucination?"}
    ]

    queue: asyncio.Queue = asyncio.Queue()
    result = await run_agent(agent, messages, queue=queue)

    while not queue.empty():
        event = queue.get_nowait()
        if event is None:
            continue
        if event.event == EventType.VERIFIER_SCORED:
            print(f"[verifier] score={event.data['score']:.2f} issues={event.data.get('issues')}")
        elif event.event == EventType.LOOP_ITERATION:
            print(f"[loop] iteration {event.data.get('iteration')}")

    print(f"\nIterations: {result.loop_iterations}")
    print(f"\nFinal answer:\n{result.text}")


if __name__ == "__main__":
    asyncio.run(main())
