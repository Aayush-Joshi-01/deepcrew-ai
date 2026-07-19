"""
Consensus code review via self-consistency branching + skill distillation.

branches=3 runs three parallel candidate reviews per iteration; the best is
picked by verifier score (or merged via APEX if no verifier). A run that
converges with high confidence is distilled into a reusable Skill.

Requires: OPENAI_API_KEY environment variable.
Run: python examples/consensus_code_review.py
"""

import asyncio

from deepcrew import Agent, LoopConfig, Verifier, VerifierConfig, run_agent
from deepcrew.skills import SkillRegistry
from deepcrew.types import EventType

DIFF = """\
def divide(a, b):
    return a / b
"""


async def main():
    agent = Agent(
        name="code_reviewer",
        model="openai/gpt-4o",
        system_prompt=(
            "You are a senior code reviewer. Point out concrete bugs and edge cases, "
            "not style nits."
        ),
        loop_config=LoopConfig(
            max_iterations=2,
            verifier=Verifier(VerifierConfig(threshold=0.8)),
            branches=3,
            auto_extract_skill=True,
            skill_confidence_threshold=0.85,
        ),
    )

    messages = [{"role": "user", "content": f"Review this diff:\n\n{DIFF}"}]

    queue: asyncio.Queue = asyncio.Queue()
    result = await run_agent(agent, messages, queue=queue)

    while not queue.empty():
        event = queue.get_nowait()
        if event and event.event == EventType.BRANCH_SELECTED:
            print(f"[branches] winning_index={event.data.get('winning_index')}")
        elif event and event.event == EventType.SKILL_EXTRACTED:
            print(f"[skill] distilled: {event.data.get('skill_name')}")

    print(f"\nReview:\n{result.text}")
    print(f"\nRegistered skills: {[s.name for s in SkillRegistry.list_all()]}")


if __name__ == "__main__":
    asyncio.run(main())
