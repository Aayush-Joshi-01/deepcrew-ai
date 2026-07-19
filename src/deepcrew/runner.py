from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import ValidationError

from .content import extract_text
from .exceptions import MaxTurnsError, OutputParseError, ToolError
from .types import AgentResult, EventType, StreamEvent, ToolDef

if TYPE_CHECKING:
    from .agent import Agent
    from .hooks import AgentHooks
    from .observability import ObservabilityConfig

litellm.drop_params = True

logger = logging.getLogger(__name__)


async def run_agent(
    agent: Agent,
    messages: list[dict[str, Any]],
    *,
    tool_defs: list[ToolDef] | None = None,
    queue: asyncio.Queue[StreamEvent | None] | None = None,
    agent_id: str | None = None,
    observability: ObservabilityConfig | None = None,
) -> AgentResult:
    """
    Run a single agent's agentic loop to completion.

    The loop calls the LLM, buffers streamed tool calls, executes them in
    parallel, appends results to the message history, and repeats until the
    model produces a response with no tool calls or ``max_turns`` is reached.

    If ``agent.loop_config`` is set, delegates to ``run_agent_loop`` for outer
    iteration before entering the inner tool-call loop.

    Parameters
    ----------
    agent:
        The Agent whose model and config to use.
    messages:
        Conversation history in OpenAI message format (role/content dicts).
        Do **not** include the system prompt here — it is taken from
        ``agent.system_prompt``.
    tool_defs:
        Pre-fetched tool definitions. If None, they are fetched via
        ``agent.get_tool_defs()``.
    queue:
        If provided, StreamEvent objects are put into this queue as they occur,
        enabling real-time streaming to callers.
    agent_id:
        Identifier used in emitted events. Defaults to ``agent.name``.
    observability:
        Optional OpenTelemetry observability configuration.

    Returns
    -------
    AgentResult
        The agent's final text, all tool calls made, and token counts.

    Raises
    ------
    MaxTurnsError
        If the agent does not finish within ``agent.max_turns`` turns.
    """
    # Delegate to outer loop if loop_config is set
    if agent.loop_config is not None:
        from .loop import run_agent_loop

        return await run_agent_loop(
            agent, messages, tool_defs=tool_defs, queue=queue, agent_id=agent_id
        )

    return await _run_agent_turns(
        agent,
        messages,
        tool_defs=tool_defs,
        queue=queue,
        agent_id=agent_id,
        observability=observability,
    )


async def _run_agent_turns(
    agent: Agent,
    messages: list[dict[str, Any]],
    *,
    tool_defs: list[ToolDef] | None = None,
    queue: asyncio.Queue[StreamEvent | None] | None = None,
    agent_id: str | None = None,
    observability: ObservabilityConfig | None = None,
) -> AgentResult:
    """
    The actual per-turn agentic loop (LLM call -> buffer tool calls -> execute
    -> repeat) for a single, non-looped run.

    This is intentionally separate from :func:`run_agent` and does **not**
    re-check ``agent.loop_config``: :func:`run_agent_loop` calls this function
    directly (not the public ``run_agent``) precisely so that an agent with
    ``loop_config`` set can be driven turn-by-turn from inside the outer loop
    without re-triggering ``run_agent``'s own delegation check on every
    iteration, which would otherwise recurse back into ``run_agent_loop``
    indefinitely.
    """
    agent_id = agent_id or agent.name
    if tool_defs is None:
        tool_defs = await agent.get_tool_defs()

    litellm_tools = [_tool_def_to_litellm(td) for td in tool_defs]
    tool_map = {td.name: td for td in tool_defs}

    history: list[dict[str, Any]] = []
    if agent.system_prompt:
        history.append({"role": "system", "content": agent.system_prompt})

    if agent.response_model is not None:
        schema_json = json.dumps(agent.response_model.model_json_schema())
        history.append(
            {
                "role": "system",
                "content": f"Respond ONLY with JSON matching this schema: {schema_json}",
            }
        )

    # Memory injection: prepend relevant memories before the conversation
    if agent.memory:
        query_text = " ".join(extract_text(m.get("content")) for m in messages[-3:])
        memories = await agent.memory.search(query_text)
        if memories:
            if queue:
                await queue.put(
                    StreamEvent(
                        EventType.MEMORY_RETRIEVE,
                        {"count": len(memories)},
                        agent_id,
                    )
                )
            memory_block = "\n".join(f"[Memory] {k}: {v}" for k, v in memories)
            history.append({"role": "system", "content": f"Relevant memories:\n{memory_block}"})

    # Procedural memory injection: read-only, works even for non-looped agents.
    # Writing/curating new strategies only happens via run_agent_loop.
    if agent.procedural_memory:
        playbook_entries = await agent.procedural_memory.load(agent_id)
        playbook_block = agent.procedural_memory.render(playbook_entries)
        if playbook_block:
            history.append({"role": "system", "content": playbook_block})

    history.extend(messages)

    total_in = 0
    total_out = 0
    all_tool_calls: list[dict[str, Any]] = []
    final_text = ""

    obs = observability
    from .observability import agent_span, llm_span, tool_span

    if agent.hooks and agent.hooks.on_agent_start is not None:
        try:
            await agent.hooks.on_agent_start()
        except Exception:
            logger.warning("agent=%s hook on_agent_start raised", agent_id, exc_info=True)

    if queue:
        await queue.put(StreamEvent(EventType.AGENT_START, {"model": agent.model}, agent_id))

    with agent_span(obs, agent.model, agent_id):
        for _turn in range(agent.max_turns):
            logger.debug(
                "agent=%s turn=%d/%d model=%s", agent_id, _turn + 1, agent.max_turns, agent.model
            )
            kwargs: dict[str, Any] = {
                "model": agent.model,
                "messages": history,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if litellm_tools:
                kwargs["tools"] = litellm_tools
                kwargs["tool_choice"] = "auto"
            if agent.temperature is not None:
                kwargs["temperature"] = agent.temperature
            if agent.max_tokens is not None:
                kwargs["max_tokens"] = agent.max_tokens
            kwargs.update(agent.extra_params)

            with llm_span(obs, agent.model, agent_id):
                if agent.retry_policy or agent.fallback_chain:
                    from .retry import with_retry_and_fallback

                    def _call_model(model: str, _kw: dict[str, Any] = kwargs) -> Any:
                        return litellm.acompletion(**{**_kw, "model": model})

                    response = await with_retry_and_fallback(_call_model, agent, queue, agent_id)
                else:
                    response = await litellm.acompletion(**kwargs)

            text_parts: list[str] = []
            tc_buffers: dict[int, dict[str, str]] = {}

            async for chunk in response:
                choice = chunk.choices[0] if chunk.choices else None
                if choice:
                    delta = choice.delta
                    if delta.content:
                        text_parts.append(delta.content)
                        if queue:
                            await queue.put(
                                StreamEvent(
                                    EventType.TEXT_DELTA,
                                    {"chunk": delta.content},
                                    agent_id,
                                )
                            )
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning and queue:
                        await queue.put(
                            StreamEvent(
                                EventType.THINKING_DELTA,
                                {"chunk": reasoning},
                                agent_id,
                            )
                        )
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            buf = tc_buffers.setdefault(
                                tc.index, {"id": "", "name": "", "args": ""}
                            )
                            if tc.id:
                                buf["id"] = tc.id
                            if tc.function and tc.function.name:
                                buf["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                buf["args"] += tc.function.arguments
                if hasattr(chunk, "usage") and chunk.usage:
                    total_in += getattr(chunk.usage, "prompt_tokens", 0) or 0
                    total_out += getattr(chunk.usage, "completion_tokens", 0) or 0

            turn_text = "".join(text_parts)
            final_text = turn_text

            if not tc_buffers:
                break

            parsed_tcs: list[dict[str, Any]] = []
            for buf in tc_buffers.values():
                try:
                    args = json.loads(buf["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                parsed_tcs.append({"id": buf["id"], "name": buf["name"], "args": args})
                all_tool_calls.append({"tool": buf["name"], "args": args, "agent_id": agent_id})
                if queue:
                    await queue.put(
                        StreamEvent(
                            EventType.TOOL_CALL,
                            {"tool": buf["name"], "args": args},
                            agent_id,
                        )
                    )

            history.append(
                {
                    "role": "assistant",
                    "content": turn_text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in parsed_tcs
                    ],
                }
            )

            logger.debug("agent=%s executing tools=%s", agent_id, [tc["name"] for tc in parsed_tcs])
            with tool_span(obs, ",".join(tc["name"] for tc in parsed_tcs), agent_id):
                tool_results = await asyncio.gather(
                    *[_run_tool(tc, tool_map, agent.hooks) for tc in parsed_tcs],
                    return_exceptions=True,
                )

            for tc, result in zip(parsed_tcs, tool_results, strict=False):
                if isinstance(result, _ToolDenied):
                    content = "Tool call denied by user."
                    if queue:
                        await queue.put(
                            StreamEvent(EventType.TOOL_DENIED, {"tool": tc["name"]}, agent_id)
                        )
                elif isinstance(result, BaseException):
                    content = f"Error executing tool '{tc['name']}': {result}"
                    logger.warning(
                        "agent=%s tool=%s raised %s: %s",
                        agent_id,
                        tc["name"],
                        type(result).__name__,
                        result,
                    )
                    if queue:
                        await queue.put(
                            StreamEvent(
                                EventType.TOOL_RESULT,
                                {"tool": tc["name"], "result": content},
                                agent_id,
                            )
                        )
                else:
                    content = result
                    if queue:
                        await queue.put(
                            StreamEvent(
                                EventType.TOOL_RESULT,
                                {"tool": tc["name"], "result": content},
                                agent_id,
                            )
                        )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": content,
                    }
                )

                # Store notable tool results in memory
                if agent.memory and not isinstance(result, BaseException):
                    mem_key = f"tool:{tc['name']}:{agent_id}"
                    await agent.memory.store(mem_key, str(result)[:500])
                    if queue:
                        await queue.put(
                            StreamEvent(
                                EventType.MEMORY_STORE,
                                {"key": mem_key},
                                agent_id,
                            )
                        )
        else:
            logger.error("agent=%s exhausted max_turns=%d", agent_id, agent.max_turns)
            raise MaxTurnsError(
                f"Agent {agent_id!r} reached max_turns={agent.max_turns} without finishing"
            )

    parsed_output: Any = None
    if agent.response_model is not None:
        parsed_output = await _parse_structured_output(agent, final_text, history)

    agent_result = AgentResult(
        agent_id=agent_id,
        text=final_text,
        tool_calls=all_tool_calls,
        input_tokens=total_in,
        output_tokens=total_out,
        model=agent.model,
        parsed=parsed_output,
    )
    if queue:
        await queue.put(
            StreamEvent(
                EventType.AGENT_DONE,
                {"input_tokens": total_in, "output_tokens": total_out},
                agent_id,
            )
        )
    return agent_result


class _ToolDenied(Exception):
    """Internal marker: an ``AgentHooks.approve_tool`` hook denied this call."""


async def _run_tool(
    tc: dict[str, Any],
    tool_map: dict[str, ToolDef],
    hooks: AgentHooks | None,
) -> str:
    """Run a tool call, honoring approve_tool/on_tool_start/on_tool_end hooks."""
    if hooks and hooks.approve_tool is not None:
        try:
            approved = await hooks.approve_tool(tc["name"], tc["args"])
        except Exception:
            logger.warning(
                "approve_tool hook raised for tool=%s; denying call", tc["name"], exc_info=True
            )
            approved = False
        if not approved:
            raise _ToolDenied()

    if hooks and hooks.on_tool_start is not None:
        try:
            await hooks.on_tool_start(tc["name"], tc["args"])
        except Exception:
            logger.warning("on_tool_start hook raised for tool=%s", tc["name"], exc_info=True)

    result = await _execute_tool(tc, tool_map)

    if hooks and hooks.on_tool_end is not None:
        try:
            await hooks.on_tool_end(tc["name"], result)
        except Exception:
            logger.warning("on_tool_end hook raised for tool=%s", tc["name"], exc_info=True)

    return result


def _strip_json_fences(text: str) -> str:
    """Strip a leading/trailing ```json ... ``` (or plain ```) code fence, if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


async def _parse_structured_output(
    agent: Agent,
    text: str,
    history: list[dict[str, Any]],
) -> Any:
    """
    Parse ``text`` against ``agent.response_model``, with one repair attempt.

    On the first parse failure, re-prompts the model once with the error
    message and asks for corrected JSON. If that also fails, raises
    ``OutputParseError`` carrying the last raw response text.
    """
    model_cls = agent.response_model
    assert model_cls is not None
    try:
        return model_cls.model_validate_json(_strip_json_fences(text))
    except (ValidationError, json.JSONDecodeError) as exc:
        repair_messages = [
            *history,
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    f"That response was not valid JSON matching the required schema. "
                    f"Error: {exc}. Respond ONLY with corrected JSON matching the schema."
                ),
            },
        ]
        resp = await litellm.acompletion(model=agent.model, messages=repair_messages, stream=False)
        repaired_text = resp.choices[0].message.content or ""
        try:
            return model_cls.model_validate_json(_strip_json_fences(repaired_text))
        except (ValidationError, json.JSONDecodeError) as exc2:
            raise OutputParseError(
                f"Agent output did not match response_model schema after "
                f"one repair attempt: {exc2}",
                raw_text=repaired_text,
            ) from exc2


async def _execute_tool(tc: dict[str, Any], tool_map: dict[str, ToolDef]) -> str:
    td = tool_map.get(tc["name"])
    if td is None:
        raise ToolError(f"Unknown tool: {tc['name']!r}")

    if td._callable is not None:
        result = td._callable(**tc["args"])
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)

    if td._mcp_client is not None:
        result = await td._mcp_client.call_tool(tc["name"], tc["args"])
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)

    raise ToolError(f"Tool {tc['name']!r} has neither a callable nor an MCP client")


def _tool_def_to_litellm(td: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }
