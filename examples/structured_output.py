"""
Structured output: validate an agent's final answer against a pydantic schema.

One automatic repair attempt is made if the first response isn't valid JSON
matching the schema; a second failure raises OutputParseError with the raw
text attached.

Requires: OPENAI_API_KEY environment variable.
Run: python examples/structured_output.py
"""

import asyncio

from pydantic import BaseModel

from deepcrew import Agent, run_agent


class ReviewVerdict(BaseModel):
    approved: bool
    reason: str
    risk_level: str  # "low" | "medium" | "high"


async def main():
    agent = Agent(
        name="reviewer",
        model="openai/gpt-4o",
        system_prompt="You are a strict but fair code reviewer.",
        response_model=ReviewVerdict,
    )

    diff = "def divide(a, b):\n    return a / b\n"
    messages = [{"role": "user", "content": f"Review this diff and give a verdict:\n\n{diff}"}]

    result = await run_agent(agent, messages)

    verdict: ReviewVerdict = result.parsed
    print(f"Approved: {verdict.approved}")
    print(f"Risk level: {verdict.risk_level}")
    print(f"Reason: {verdict.reason}")


if __name__ == "__main__":
    asyncio.run(main())
