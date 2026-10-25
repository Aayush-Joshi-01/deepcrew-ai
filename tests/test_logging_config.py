from __future__ import annotations

import json
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import deepcrew.logging_config as lc
from deepcrew import configure_logging, get_logger
from deepcrew.agent import Agent
from deepcrew.logging_config import TRACE, ColorFormatter, JsonFormatter, LoggingConfig
from deepcrew.runner import run_agent


@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch):
    """Each test starts from a pristine, unconfigured ``deepcrew`` logger."""
    for key in [k for k in list(os.environ) if k.startswith("DEEPCREW_LOG")]:
        monkeypatch.delenv(key, raising=False)
    root = logging.getLogger("deepcrew")
    saved = root.handlers[:]
    saved_level, saved_propagate = root.level, root.propagate
    root.handlers = [logging.NullHandler()]
    lc._configured = False
    yield
    root.handlers = saved
    root.setLevel(saved_level)
    root.propagate = saved_propagate
    lc._configured = False


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, lc._HANDLER_TAG, False)]


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=lc.TRACE)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture():
    """Capture records from the ``deepcrew`` tree without relying on propagation."""
    handler = _CaptureHandler()
    root = logging.getLogger("deepcrew")
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)


def test_configure_logging_is_idempotent():
    configure_logging("dev")
    configure_logging("dev")
    assert len(_managed_handlers(logging.getLogger("deepcrew"))) == 1


def test_profile_dev_sets_debug_text():
    configure_logging("dev")
    root = logging.getLogger("deepcrew")
    assert root.level == logging.DEBUG
    handler = _managed_handlers(root)[0]
    assert not isinstance(handler.formatter, JsonFormatter)


def test_profile_prod_sets_json_info():
    configure_logging("prod")
    root = logging.getLogger("deepcrew")
    assert root.level == logging.INFO
    assert isinstance(_managed_handlers(root)[0].formatter, JsonFormatter)


def test_profile_verbose_enables_trace():
    configure_logging("verbose")
    assert logging.getLogger("deepcrew").level == TRACE


def test_env_var_overrides_explicit_argument(monkeypatch):
    monkeypatch.setenv("DEEPCREW_LOG_LEVEL", "ERROR")
    configure_logging(LoggingConfig(level="INFO"))
    assert logging.getLogger("deepcrew").level == logging.ERROR


def test_json_formatter_emits_valid_json_with_extras():
    rec = logging.LogRecord("deepcrew.runner", logging.INFO, __file__, 1, "agent start", (), None)
    rec.agent_id = "researcher"
    rec.turn = 3
    line = JsonFormatter(timestamps=True).format(rec)
    payload = json.loads(line)
    assert payload["msg"] == "agent start"
    assert payload["agent_id"] == "researcher"
    assert payload["turn"] == 3
    assert payload["level"] == "INFO"


def test_color_formatter_ansi_toggle():
    rec = logging.LogRecord("deepcrew.x", logging.WARNING, __file__, 1, "hi", (), None)
    plain = ColorFormatter(color=False, timestamps=False).format(rec)
    colored = ColorFormatter(color=True, timestamps=False).format(rec)
    assert "\033[" not in plain
    assert "\033[" in colored


def test_get_logger_routes_kwargs_to_extra(capture):
    configure_logging(level="DEBUG")
    get_logger("mycomponent").info("price lookup", sku="ABC", duration_ms=12.4)
    rec = capture.records[-1]
    assert rec.sku == "ABC"
    assert rec.duration_ms == 12.4


def test_bind_carries_context(capture):
    configure_logging(level="DEBUG")
    get_logger("bound").bind(agent_id="a1").info("step", turn=1)
    rec = capture.records[-1]
    assert rec.agent_id == "a1"
    assert rec.turn == 1


def test_silent_by_default():
    root = logging.getLogger("deepcrew")
    assert _managed_handlers(root) == []
    get_logger("quiet").info("should not be configured")
    assert lc._configured is False


def test_env_var_triggers_autoconfigure(monkeypatch):
    monkeypatch.setenv("DEEPCREW_LOG_PROFILE", "prod")
    get_logger("auto").info("hi")
    assert lc._configured is True
    assert isinstance(_managed_handlers(logging.getLogger("deepcrew"))[0].formatter, JsonFormatter)


def test_rich_handler_used_when_available_and_colored():
    pytest.importorskip("rich")
    configure_logging(LoggingConfig(profile="dev", color=True))
    handler = _managed_handlers(logging.getLogger("deepcrew"))[0]
    assert type(handler).__name__ == "RichHandler"


# --- integration -------------------------------------------------------


def _make_stream_chunk(text: str | None = None):
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = []
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


async def _fake_stream(*chunks):
    for c in chunks:
        yield c


async def test_run_agent_emits_lifecycle_logs(capture):
    configure_logging(level="DEBUG", color=False)
    agent = Agent(name="researcher", model="openai/gpt-4o")

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_fake_stream(_make_stream_chunk("done"))),
    ):
        await run_agent(agent, [{"role": "user", "content": "hi"}])

    messages = {r.getMessage() for r in capture.records}
    assert "agent start" in messages
    assert "agent done" in messages
    assert any(getattr(r, "agent_id", None) == "researcher" for r in capture.records)
