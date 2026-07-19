from __future__ import annotations

import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from deepcrew.agent import Agent  # noqa: E402
from deepcrew.integrations.fastapi import create_stream_router  # noqa: E402
from deepcrew.stream import StreamPolicy  # noqa: E402


def _make_chunk(content: str | None = None, tool_calls=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
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


def _client_for_agent(agent: Agent, **router_kwargs) -> TestClient:
    app = FastAPI()
    app.include_router(create_stream_router(agent, **router_kwargs))
    return TestClient(app)


def test_import_error_message_when_fastapi_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.delitem(sys.modules, "deepcrew.integrations.fastapi", raising=False)
    try:
        with pytest.raises(ImportError, match=r"pip install deepcrew-ai\[fastapi\]"):
            importlib.import_module("deepcrew.integrations.fastapi")
    finally:
        monkeypatch.delitem(sys.modules, "deepcrew.integrations.fastapi", raising=False)


def test_chat_endpoint_streams_sse_with_default_chat_policy():
    agent = Agent(name="bot", model="openai/gpt-4o")
    client = _client_for_agent(agent)

    chunks = [_make_chunk("Hello "), _make_chunk("world!"), _make_chunk()]
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))),
        client.stream("POST", "/chat", json={"query": "hi"}) as response,
    ):
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: text_delta" in body
    assert "event: tool_call" not in body
    assert "event: done" in body


def test_chat_endpoint_verbose_policy_shows_agent_start():
    agent = Agent(name="bot", model="openai/gpt-4o")
    client = _client_for_agent(agent, policy=StreamPolicy.verbose())

    chunks = [_make_chunk("Hi"), _make_chunk()]
    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))),
        client.stream("POST", "/chat", json={"query": "hi"}) as response,
    ):
        body = "".join(response.iter_text())

    assert "event: agent_start" in body


def test_chat_endpoint_invalid_image_returns_422():
    agent = Agent(name="bot", model="openai/gpt-4o")
    client = _client_for_agent(agent)

    response = client.post("/chat", json={"query": "hi", "images": ["not-a-valid-source"]})
    assert response.status_code == 422


def test_complete_endpoint_returns_final_text():
    agent = Agent(name="bot", model="openai/gpt-4o")
    client = _client_for_agent(agent)

    chunks = [_make_chunk("The answer is 42."), _make_chunk()]
    with patch("litellm.acompletion", new=AsyncMock(return_value=_make_stream(*chunks))):
        response = client.post("/chat/complete", json={"query": "what is the answer?"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "The answer is 42."
    assert body["agent_id"] == "bot"
