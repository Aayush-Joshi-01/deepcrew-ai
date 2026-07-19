from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.apex import ApexConfig, APEXSynthesizer
from deepcrew.types import AgentResult, EventType


def _make_nonstream_response(text: str, in_tok: int = 10, out_tok: int = 5):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = in_tok
    usage.completion_tokens = out_tok
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


@pytest.mark.asyncio
async def test_synthesize_zero_results():
    synth = APEXSynthesizer("openai/gpt-4o")
    resp = _make_nonstream_response("Final answer.\nCONFIDENCE: 0.9")

    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await synth.synthesize("query", [])

    assert result.agent_id == "apex"
    assert result.text == "Final answer."
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_synthesize_one_result():
    synth = APEXSynthesizer("openai/gpt-4o")
    resp = _make_nonstream_response("Merged.\nCONFIDENCE: 0.75")

    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await synth.synthesize("q", [AgentResult(agent_id="researcher", text="facts")])

    assert result.text == "Merged."
    assert result.confidence == 0.75


@pytest.mark.asyncio
async def test_synthesize_n_results():
    synth = APEXSynthesizer("openai/gpt-4o")
    resp = _make_nonstream_response("Combined.\nCONFIDENCE: 0.6")

    results = [
        AgentResult(agent_id="researcher", text="facts"),
        AgentResult(agent_id="writer", text="prose"),
        AgentResult(agent_id="critic", text="feedback"),
    ]

    mock_completion = AsyncMock(return_value=resp)
    with patch("litellm.acompletion", new=mock_completion):
        result = await synth.synthesize("q", results)

    assert result.text == "Combined."
    sent_prompt = mock_completion.call_args.kwargs["messages"][-1]["content"]
    assert "researcher" in sent_prompt
    assert "writer" in sent_prompt
    assert "critic" in sent_prompt


@pytest.mark.asyncio
async def test_confidence_valid_float_parsed():
    synth = APEXSynthesizer("openai/gpt-4o")
    resp = _make_nonstream_response("Answer text.\nCONFIDENCE: 0.42")

    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await synth.synthesize("q", [])

    assert result.confidence == 0.42


@pytest.mark.asyncio
async def test_confidence_garbage_defaults_to_point_eight():
    synth = APEXSynthesizer("openai/gpt-4o")
    resp = _make_nonstream_response("Answer with no confidence line at all.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await synth.synthesize("q", [])

    assert result.confidence == 0.8


@pytest.mark.asyncio
async def test_confidence_out_of_range_is_clamped():
    synth = APEXSynthesizer("openai/gpt-4o")
    resp = _make_nonstream_response("Answer.\nCONFIDENCE: 5.0")

    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        result = await synth.synthesize("q", [])

    assert result.confidence == 1.0


def test_build_citations_extracts_inline_sources():
    synth = APEXSynthesizer("openai/gpt-4o")
    text = "The sky is blue [source: researcher]. Bananas are yellow [source: writer]."

    citations = synth.build_citations([], text)

    assert [c.agent_id for c in citations] == ["researcher", "writer"]
    assert all(c.confidence == 0.9 for c in citations)


def test_build_citations_no_matches_returns_empty():
    synth = APEXSynthesizer("openai/gpt-4o")
    citations = synth.build_citations([], "No citations here.")
    assert citations == []


@pytest.mark.asyncio
async def test_synthesize_emits_apex_start_and_done_events():
    synth = APEXSynthesizer("openai/gpt-4o", ApexConfig())
    resp = _make_nonstream_response("Answer.\nCONFIDENCE: 0.5")
    queue: asyncio.Queue = asyncio.Queue()

    with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
        await synth.synthesize("q", [AgentResult(agent_id="a", text="x")], queue=queue)

    events = []
    while not queue.empty():
        events.append(await queue.get())

    types = [e.event for e in events]
    assert EventType.APEX_START in types
    assert EventType.APEX_DONE in types
    start_event = next(e for e in events if e.event == EventType.APEX_START)
    assert start_event.data["agents"] == 1
