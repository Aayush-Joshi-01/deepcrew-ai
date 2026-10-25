"""Broader live coverage of the headline features."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from deepcrew import (
    Agent,
    ApexConfig,
    LoopConfig,
    Orchestrator,
    Verifier,
    VerifierConfig,
    run_agent,
)

pytestmark = pytest.mark.live(
    feature="core_features",
    watch=[
        "src/deepcrew/orchestrator.py",
        "src/deepcrew/apex.py",
        "src/deepcrew/spawner.py",
        "src/deepcrew/loop.py",
        "src/deepcrew/runner.py",
    ],
)


@pytest.mark.asyncio
async def test_orchestrator_fanout_and_apex(live_model: str):
    researcher = Agent(name="researcher", model=live_model, system_prompt="Give facts.")
    critic = Agent(name="critic", model=live_model, system_prompt="Give counterpoints.")
    orch = Orchestrator(
        agents=[researcher, critic],
        router_model=live_model,
        apex_model=live_model,
        apex_config=ApexConfig(cite_sources=False),
    )
    result = await orch.run("Briefly: are electric cars greener than petrol cars?")
    assert result.final_text.strip()


@pytest.mark.asyncio
async def test_structured_output_parses(live_model: str):
    class Verdict(BaseModel):
        approved: bool
        reason: str

    agent = Agent(name="reviewer", model=live_model, response_model=Verdict, max_turns=2)
    result = await run_agent(
        agent, [{"role": "user", "content": "Approve merging a one-line typo fix? Give a reason."}]
    )
    assert isinstance(result.parsed, Verdict)
    assert result.parsed.reason.strip()


@pytest.mark.asyncio
async def test_loop_with_verifier_runs(live_model: str):
    agent = Agent(
        name="researcher",
        model=live_model,
        system_prompt="Answer with concrete, named examples.",
        max_turns=2,
        loop_config=LoopConfig(
            max_iterations=3,
            adaptive=True,
            verifier=Verifier(
                VerifierConfig(threshold=0.8, rubric="At least two concrete named examples.")
            ),
        ),
    )
    result = await run_agent(
        agent, [{"role": "user", "content": "Name approaches to reducing LLM hallucination."}]
    )
    assert result.loop_iterations >= 1
    assert result.text.strip()
