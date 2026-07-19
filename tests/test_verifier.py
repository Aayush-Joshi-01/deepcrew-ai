from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.types import AgentResult
from deepcrew.verifier import Verifier, VerifierConfig, VerifierFeedback


def _make_litellm_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_evaluate_parses_well_formed_json():
    verifier = Verifier(VerifierConfig(threshold=0.8))
    response = _make_litellm_response(json.dumps({"score": 0.9, "issues": [], "suggestion": ""}))
    result = AgentResult(agent_id="a", text="A great answer.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        feedback = await verifier.evaluate("What is X?", result, default_model="openai/gpt-4o-mini")

    assert isinstance(feedback, VerifierFeedback)
    assert feedback.score == 0.9
    assert feedback.converged is True
    assert feedback.issues == []


@pytest.mark.asyncio
async def test_evaluate_below_threshold_not_converged():
    verifier = Verifier(VerifierConfig(threshold=0.8))
    response = _make_litellm_response(
        json.dumps({"score": 0.4, "issues": ["missing citations"], "suggestion": "add sources"})
    )
    result = AgentResult(agent_id="a", text="A weak answer.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        feedback = await verifier.evaluate("What is X?", result, default_model="openai/gpt-4o-mini")

    assert feedback.score == 0.4
    assert feedback.converged is False
    assert feedback.issues == ["missing citations"]
    assert feedback.suggestion == "add sources"


@pytest.mark.asyncio
async def test_evaluate_malformed_json_falls_back_gracefully():
    verifier = Verifier(VerifierConfig(threshold=0.8))
    response = _make_litellm_response("not json at all @@##")
    result = AgentResult(agent_id="a", text="An answer.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        feedback = await verifier.evaluate("What is X?", result, default_model="openai/gpt-4o-mini")

    assert feedback.score == 0.0
    assert feedback.converged is False


@pytest.mark.asyncio
async def test_evaluate_uses_evaluate_fn_override():
    async def fake_evaluate(query: str, result: AgentResult) -> VerifierFeedback:
        return VerifierFeedback(score=1.0, converged=True)

    verifier = Verifier(VerifierConfig(evaluate_fn=fake_evaluate))
    result = AgentResult(agent_id="a", text="An answer.")

    with patch(
        "litellm.acompletion", new=AsyncMock(side_effect=AssertionError("should not be called"))
    ):
        feedback = await verifier.evaluate("What is X?", result, default_model="openai/gpt-4o-mini")

    assert feedback.score == 1.0
    assert feedback.converged is True


@pytest.mark.asyncio
async def test_assess_complexity_defaults_permissive_on_parse_failure():
    verifier = Verifier(VerifierConfig())
    response = _make_litellm_response("garbage")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        needs_decomposition = await verifier.assess_complexity(
            "Do a big task", default_model="openai/gpt-4o-mini"
        )

    assert needs_decomposition is True


@pytest.mark.asyncio
async def test_assess_complexity_respects_parsed_value():
    verifier = Verifier(VerifierConfig())
    response = _make_litellm_response(json.dumps({"needs_decomposition": False}))

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        needs_decomposition = await verifier.assess_complexity(
            "Do a small task", default_model="openai/gpt-4o-mini"
        )

    assert needs_decomposition is False
