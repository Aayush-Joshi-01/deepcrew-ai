from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from deepcrew.agent import Agent
from deepcrew.exceptions import OutputParseError
from deepcrew.runner import run_agent


class Answer(BaseModel):
    value: int


def _make_chunk(content: str | None = None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = []
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _make_stream(*chunks):
    async def _gen():
        for c in chunks:
            yield c

    return _gen()


def _make_nonstream_response(text: str):
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_valid_json_parsed_on_first_try():
    agent = Agent(name="t", model="openai/gpt-4o", response_model=Answer)
    chunks = [_make_chunk('{"value": 42}'), _make_chunk()]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        result = await run_agent(agent, [{"role": "user", "content": "answer?"}])

    assert result.parsed == Answer(value=42)


@pytest.mark.asyncio
async def test_fenced_json_is_stripped_before_parsing():
    agent = Agent(name="t", model="openai/gpt-4o", response_model=Answer)
    chunks = [_make_chunk('```json\n{"value": 7}\n```'), _make_chunk()]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        result = await run_agent(agent, [{"role": "user", "content": "answer?"}])

    assert result.parsed == Answer(value=7)


@pytest.mark.asyncio
async def test_invalid_json_repaired_on_second_call():
    agent = Agent(name="t", model="openai/gpt-4o", response_model=Answer)

    async def fake_completion(**kwargs):
        if kwargs.get("stream") is False:
            return _make_nonstream_response('{"value": 99}')
        return _make_stream(_make_chunk("not json at all"), _make_chunk())

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)):
        result = await run_agent(agent, [{"role": "user", "content": "answer?"}])

    assert result.parsed == Answer(value=99)


@pytest.mark.asyncio
async def test_invalid_json_twice_raises_output_parse_error_with_raw_text():
    agent = Agent(name="t", model="openai/gpt-4o", response_model=Answer)

    async def fake_completion(**kwargs):
        if kwargs.get("stream") is False:
            return _make_nonstream_response("still not json")
        return _make_stream(_make_chunk("not json at all"), _make_chunk())

    with (
        patch("litellm.acompletion", new=AsyncMock(side_effect=fake_completion)),
        pytest.raises(OutputParseError) as exc_info,
    ):
        await run_agent(agent, [{"role": "user", "content": "answer?"}])

    assert exc_info.value.raw_text == "still not json"


@pytest.mark.asyncio
async def test_no_response_model_leaves_parsed_none():
    agent = Agent(name="t", model="openai/gpt-4o")
    chunks = [_make_chunk("just plain text"), _make_chunk()]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        result = await run_agent(agent, [{"role": "user", "content": "hi"}])

    assert result.parsed is None


@pytest.mark.asyncio
async def test_response_model_injects_schema_instruction():
    agent = Agent(name="t", model="openai/gpt-4o", response_model=Answer)
    chunks = [_make_chunk('{"value": 1}'), _make_chunk()]

    mock_completion = AsyncMock(return_value=_make_stream(*chunks))
    with patch("litellm.acompletion", new=mock_completion):
        await run_agent(agent, [{"role": "user", "content": "answer?"}])

    sent_messages = mock_completion.call_args.kwargs["messages"]
    schema_messages = [
        m
        for m in sent_messages
        if m["role"] == "system" and "Respond ONLY with JSON matching this schema" in m["content"]
    ]
    assert len(schema_messages) == 1
    assert "value" in schema_messages[0]["content"]
