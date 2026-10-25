"""
Procedural memory: a self-improving loop distils reusable strategies (a
"playbook") from verifier feedback, then a later run reads that playbook back.

The create step happens inside ``run_agent_loop`` when the ``LoopConfig`` carries
both a ``Verifier`` and a ``ProceduralMemory``; the utilize step happens on any
later run whose ``Agent.procedural_memory`` points at the same store and tag.

Requires: GEMINI_API_KEY environment variable.
Run: python examples/procedural_memory.py
"""

import asyncio

from deepcrew import Agent, LoopConfig, ProceduralMemory, Verifier, VerifierConfig, run_agent
from deepcrew.memory import InMemoryProvider

MODEL = "gemini/gemini-3.5-flash-lite"
TAG = "explainer"


async def main():
    playbook = ProceduralMemory(InMemoryProvider())

    # --- create: a loop that must satisfy a strict rubric curates strategies
    learner = Agent(
        name="explainer",
        model=MODEL,
        system_prompt="Explain the concept in a short paragraph.",
        max_turns=2,
        loop_config=LoopConfig(
            max_iterations=3,
            task_tag=TAG,
            procedural_memory=playbook,
            adaptive=True,
            verifier=Verifier(
                VerifierConfig(
                    threshold=0.9,
                    rubric="Must include a concrete real-world analogy and one specific number.",
                )
            ),
        ),
    )
    await run_agent(learner, [{"role": "user", "content": "Explain how vaccines work."}])

    entries = await playbook.load(TAG)
    print(f"Playbook after the loop ({len(entries)} strategies):")
    for e in entries:
        print(f"  - ({e.kind}) {e.content}")

    # --- utilize: a fresh single-shot agent reads the same playbook
    writer = Agent(
        name="explainer",
        model=MODEL,
        system_prompt="Explain the concept in a short paragraph.",
        procedural_memory=playbook,
        procedural_task_tag=TAG,
        max_turns=2,
    )
    result = await run_agent(writer, [{"role": "user", "content": "Explain how antibiotics work."}])
    print(f"\nNext answer (guided by the playbook):\n{result.text}")


if __name__ == "__main__":
    asyncio.run(main())
