from unittest.mock import MagicMock

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


# --- shared litellm mock helpers ------------------------------------------
# These mirror the hand-built chunk/response builders copy-pasted across many
# test modules. New tests should depend on these fixtures instead of re-copying.


@pytest.fixture
def make_stream_chunk():
    """Factory: a minimal streamed-completion chunk with the given text delta."""

    def _make(text: str | None = None, *, tool_calls=None, usage=None, reasoning=None):
        delta = MagicMock()
        delta.content = text
        delta.tool_calls = tool_calls or []
        delta.reasoning_content = reasoning
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = usage
        return chunk

    return _make


@pytest.fixture
def make_stream():
    """Factory: an already-invoked async generator yielding the given chunks."""

    def _make(*chunks):
        async def _gen():
            for c in chunks:
                yield c

        return _gen()

    return _make


@pytest.fixture
def make_litellm_response():
    """Factory: a non-streaming litellm response object with the given content."""

    def _make(text: str):
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    return _make
