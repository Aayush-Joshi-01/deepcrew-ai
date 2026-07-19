from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from deepcrew.exceptions import SkillError
from deepcrew.skills.builtin import CodeExecutionSkill, SummarizeSkill, WebSearchSkill

# --- to_tool_def() schemas ---------------------------------------------------


def test_web_search_to_tool_def_schema():
    td = WebSearchSkill().to_tool_def()
    assert td.name == "web_search"
    assert td.parameters["required"] == ["query"]
    assert "query" in td.parameters["properties"]


def test_summarize_to_tool_def_schema():
    td = SummarizeSkill().to_tool_def()
    assert td.name == "summarize"
    assert td.parameters["required"] == ["text"]


def test_code_exec_to_tool_def_schema():
    td = CodeExecutionSkill().to_tool_def()
    assert td.name == "code_exec"
    assert td.parameters["required"] == ["code"]


# --- WebSearchSkill -----------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_returns_abstract_and_topics():
    skill = WebSearchSkill()
    mock_json = {
        "AbstractText": "DuckDuckGo is a search engine.",
        "RelatedTopics": [{"Text": "Topic one"}, {"Text": "Topic two"}],
    }

    with respx.mock:
        respx.get("https://api.duckduckgo.com/").mock(
            return_value=httpx.Response(200, json=mock_json)
        )
        result = await skill.execute(query="duckduckgo", max_results=5)

    assert "DuckDuckGo is a search engine." in result
    assert "Topic one" in result
    assert "Topic two" in result


@pytest.mark.asyncio
async def test_web_search_respects_max_results():
    skill = WebSearchSkill()
    mock_json = {
        "AbstractText": "",
        "RelatedTopics": [{"Text": f"Topic {i}"} for i in range(5)],
    }

    with respx.mock:
        respx.get("https://api.duckduckgo.com/").mock(
            return_value=httpx.Response(200, json=mock_json)
        )
        result = await skill.execute(query="x", max_results=2)

    assert "Topic 0" in result
    assert "Topic 1" in result
    assert "Topic 2" not in result


@pytest.mark.asyncio
async def test_web_search_no_results_message():
    skill = WebSearchSkill()
    with respx.mock:
        respx.get("https://api.duckduckgo.com/").mock(
            return_value=httpx.Response(200, json={"AbstractText": "", "RelatedTopics": []})
        )
        result = await skill.execute(query="nonexistentqueryxyz")

    assert "No results found" in result


@pytest.mark.asyncio
async def test_web_search_http_error_returns_message_not_raise():
    skill = WebSearchSkill()
    with respx.mock:
        respx.get("https://api.duckduckgo.com/").mock(return_value=httpx.Response(500))
        result = await skill.execute(query="x")

    assert "Error searching" in result


@pytest.mark.asyncio
async def test_web_search_httpx_not_installed_returns_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    skill = WebSearchSkill()
    result = await skill.execute(query="x")
    assert "httpx not installed" in result


# --- SummarizeSkill -----------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_calls_litellm_and_returns_content():
    skill = SummarizeSkill(model="openai/gpt-4o-mini")

    msg = MagicMock()
    msg.content = "A short summary."
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]

    mock_completion = AsyncMock(return_value=resp)
    with patch("litellm.acompletion", new=mock_completion):
        result = await skill.execute(text="A very long piece of text.", max_words=50)

    assert result == "A short summary."
    sent_prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
    assert "50 words" in sent_prompt
    assert "A very long piece of text." in sent_prompt
    assert mock_completion.call_args.kwargs["model"] == "openai/gpt-4o-mini"


# --- CodeExecutionSkill -----------------------------------------------------


@pytest.mark.asyncio
async def test_code_exec_runs_trivial_snippet():
    skill = CodeExecutionSkill()
    result = await skill.execute(code="print('hello from subprocess')")
    assert "hello from subprocess" in result


@pytest.mark.asyncio
async def test_code_exec_captures_stderr():
    skill = CodeExecutionSkill()
    result = await skill.execute(code="import sys; sys.stderr.write('oops')")
    assert "[stderr]" in result
    assert "oops" in result


@pytest.mark.asyncio
async def test_code_exec_unsupported_language_raises():
    skill = CodeExecutionSkill()
    with pytest.raises(SkillError, match="only supports 'python'"):
        await skill.execute(code="echo hi", language="bash")


@pytest.mark.asyncio
async def test_code_exec_timeout_raises_skill_error():
    skill = CodeExecutionSkill(timeout=0.2)
    with pytest.raises(SkillError, match="timed out"):
        await skill.execute(code="import time; time.sleep(5)")
