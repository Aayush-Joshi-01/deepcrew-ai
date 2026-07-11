from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import litellm

from .agent import Agent
from .apex import ApexConfig, APEXSynthesizer
from .exceptions import RouterError
from .runner import run_agent
from .spawner import ToolAllocator, make_spawn_tool
from .stream import make_error_event, queue_to_stream
from .types import AgentResult, EventType, OrchestratorResult, StreamEvent, ToolDef
from .verifier import Verifier

litellm.drop_params = True

_ROUTER_SYSTEM = """\
You are a task decomposition specialist.
Given a user query and a list of available agents, decide the best execution plan.

Rules:
- Use "single" when one agent is clearly the best fit or the task is simple.
- Use "parallel" when multiple agents can independently work on different aspects
  of the task simultaneously (research + analysis + writing, for example).
- Never assign the same task to two agents.
- Limit parallel agents to those that are truly helpful for this specific query.

Available agents:
{agent_descriptions}
{tool_section}
Respond with ONLY valid JSON, exactly one of:
  {{"route": "single", "agent": "<name>", "task": "<specific task for the agent>"}}
  {{"route": "parallel", "agents": [{{"name": "<name>", "task": "<specific task>"}}, ...]}}
"""

_ROUTER_TOOL_SECTION = """\

Available global tools (assign relevant ones to each agent when routing parallel):
{tool_descriptions}

For parallel routing, you may optionally include a "tools" array per agent:
  {{"route": "parallel", "agents": [{{"name": "<name>", "task": "...", "tools": ["tool_a"]}}, ...]}}
"""

_SYNTHESIZER_SYSTEM = """\
You are a synthesis specialist.
You receive results from multiple specialized agents who worked in parallel.
Synthesize all their findings into a single, cohesive, well-structured response.
Do NOT mention agents or attribute information to specific agents.
Just produce the best unified answer.
"""


class Orchestrator:
    """
    Automated multi-agent orchestrator with APEX synthesis.

    Runs a three-stage pipeline:

    1. **Router** — a lightweight LLM call that reads the user query and the
       list of available agents, then decides whether to route to a single
       agent or fan out to multiple agents running in parallel. When
       ``global_tools`` are provided, the router also performs intelligent
       tool allocation per agent.
    2. **Execution** — the chosen agent(s) run via :func:`run_agent`, either
       one at a time or concurrently via ``asyncio.gather``. When
       ``enable_spawn=True``, every agent receives a ``spawn_agent`` meta-tool
       so it can dynamically create sub-agents mid-loop.
    3. **APEX** — if multiple agents ran, the APEX synthesizer merges their
       outputs with confidence scoring and optional source citation.

    Parameters
    ----------
    agents:
        The pool of agents the orchestrator can choose from.
    router_model:
        LiteLLM model string used for the routing decision.
    synthesizer_model:
        Backward-compat alias for ``apex_model``.
    apex_model:
        LiteLLM model string for the APEX synthesis step.
        Defaults to ``router_model``.
    apex_config:
        Fine-grained APEX configuration (confidence threshold, citations, etc.).
    router_system_prompt:
        Override the built-in router system prompt.
    synthesizer_system_prompt:
        Override the built-in synthesizer system prompt (used only when
        ``apex_config`` is not set).
    max_parallel_agents:
        Cap on the number of agents the router may spawn in parallel.
    global_tools:
        Pool of ToolDefs available for intelligent allocation to agents.
    enable_spawn:
        When True, inject a ``spawn_agent`` meta-tool into every running agent
        so it can dynamically spawn sub-agents mid-loop.
    max_spawn_depth:
        Hard, never-exceeded ceiling on nesting depth for recursive spawning
        (a spawned sub-agent spawning its own sub-agent, and so on). Only
        relevant when ``enable_spawn=True``. Defaults to 2 (one nested level).
    spawn_complexity_check:
        Optional ``Verifier`` whose ``assess_complexity()`` gates whether a
        newly-spawned sub-agent is worth giving its own nested spawn tool to,
        below the hard ``max_spawn_depth`` cap.
    """

    def __init__(
        self,
        agents: list[Agent],
        router_model: str = "openai/gpt-4o-mini",
        synthesizer_model: str | None = None,
        apex_model: str | None = None,
        apex_config: ApexConfig | None = None,
        router_system_prompt: str | None = None,
        synthesizer_system_prompt: str | None = None,
        max_parallel_agents: int = 5,
        global_tools: list[ToolDef] | None = None,
        enable_spawn: bool = False,
        max_spawn_depth: int = 2,
        spawn_complexity_check: Verifier | None = None,
    ) -> None:
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self.router_model = router_model
        # apex_model takes priority; synthesizer_model kept for backward compat
        self._apex_model = apex_model or synthesizer_model or router_model
        # Keep as a property for backward compat
        self.synthesizer_model = self._apex_model
        self._apex_config = apex_config
        self._router_system = router_system_prompt or _ROUTER_SYSTEM
        self._synth_system = synthesizer_system_prompt or _SYNTHESIZER_SYSTEM
        self.max_parallel_agents = max_parallel_agents
        self._global_tools: list[ToolDef] = global_tools or []
        self._enable_spawn = enable_spawn
        self._max_spawn_depth = max_spawn_depth
        self._spawn_complexity_check = spawn_complexity_check
        self._apex = APEXSynthesizer(self._apex_model, apex_config)

    async def run(self, query: str, context: dict[str, Any] | None = None) -> OrchestratorResult:
        """Run the orchestration pipeline and return the complete result."""
        events: list[StreamEvent] = []
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        task = asyncio.create_task(self._orchestrate(query, context or {}, queue))

        async for event in queue_to_stream(queue, task):
            events.append(event)

        # Reconstruct result from captured events
        agent_results: list[AgentResult] = []
        final_text = ""
        total_in = total_out = 0

        current_agent: dict[str, Any] = {}
        text_parts: dict[str, list[str]] = {}

        for ev in events:
            if ev.event == EventType.AGENT_START:
                current_agent[ev.agent_id] = ev.data.get("model", "")
                text_parts[ev.agent_id] = []
            elif ev.event == EventType.TEXT_DELTA:
                text_parts.setdefault(ev.agent_id, []).append(ev.data.get("chunk", ""))
            elif ev.event == EventType.AGENT_DONE:
                text = "".join(text_parts.get(ev.agent_id, []))
                agent_results.append(
                    AgentResult(
                        agent_id=ev.agent_id,
                        text=text,
                        input_tokens=ev.data.get("input_tokens", 0),
                        output_tokens=ev.data.get("output_tokens", 0),
                        model=current_agent.get(ev.agent_id, ""),
                    )
                )
                total_in += ev.data.get("input_tokens", 0)
                total_out += ev.data.get("output_tokens", 0)
            elif ev.event == EventType.DONE:
                final_text = ev.data.get("final_text", "")

        return OrchestratorResult(
            final_text=final_text,
            agent_results=agent_results,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    def stream(
        self, query: str, context: dict[str, Any] | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream events as the orchestration pipeline runs."""
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        task = asyncio.create_task(self._orchestrate(query, context or {}, queue))
        return queue_to_stream(queue, task)

    async def _orchestrate(
        self,
        query: str,
        context: dict[str, Any],
        queue: asyncio.Queue[StreamEvent | None],
    ) -> None:
        try:
            routing = await self._route(query)
            agent_results: list[AgentResult] = []

            # Build spawn meta-tool if enabled
            spawn_tool: ToolDef | None = None
            if self._enable_spawn and self._global_tools:
                spawn_tool = make_spawn_tool(
                    self._global_tools,
                    queue,
                    self.router_model,
                    "orchestrator",
                    current_depth=0,
                    max_depth=self._max_spawn_depth,
                    complexity_check=self._spawn_complexity_check,
                )

            if routing["route"] == "single":
                agent_name = routing.get("agent", "")
                agent = self._get_agent(agent_name)
                task_override = routing.get("task", query)
                extra_tools = await self._get_allocated_tools(routing, agent_name)
                if spawn_tool:
                    extra_tools = [spawn_tool, *extra_tools]
                result = await run_agent(
                    agent,
                    [{"role": "user", "content": task_override}],
                    tool_defs=(await agent.get_tool_defs() + extra_tools) if extra_tools else None,
                    queue=queue,
                    agent_id=agent_name,
                )
                agent_results.append(result)
                final_text = result.text

            else:
                specs = routing.get("agents", [])[: self.max_parallel_agents]
                parallel_coros = []
                for spec in specs:
                    agent = self._get_agent(spec["name"])
                    extra_tools = await self._get_allocated_tools(spec, spec["name"])
                    if spawn_tool:
                        extra_tools = [spawn_tool, *extra_tools]
                    if extra_tools:
                        base_tools = await agent.get_tool_defs()
                        merged = base_tools + extra_tools
                    else:
                        merged = None
                    parallel_coros.append(
                        run_agent(
                            agent,
                            [{"role": "user", "content": spec.get("task", query)}],
                            tool_defs=merged,
                            queue=queue,
                            agent_id=spec["name"],
                        )
                    )

                parallel_results = await asyncio.gather(
                    *parallel_coros,
                    return_exceptions=True,
                )

                for r in parallel_results:
                    if isinstance(r, BaseException):
                        await queue.put(make_error_event("orchestrator", str(r)))
                    else:
                        agent_results.append(r)

                apex_result = await self._apex.synthesize(query, agent_results, queue)
                final_text = apex_result.text

            await queue.put(StreamEvent(EventType.DONE, {"final_text": final_text}, "orchestrator"))
        except Exception as exc:
            await queue.put(make_error_event("orchestrator", str(exc)))
        finally:
            await queue.put(None)

    async def _route(self, query: str) -> dict[str, Any]:
        agent_descriptions = "\n".join(
            f"- {name}: {agent.system_prompt[:120] or '(no description)'}"
            for name, agent in self.agents.items()
        )

        tool_section = ""
        if self._global_tools:
            tool_descriptions = "\n".join(
                f"- {td.name}: {td.description}" for td in self._global_tools
            )
            tool_section = _ROUTER_TOOL_SECTION.format(tool_descriptions=tool_descriptions)

        system = self._router_system.format(
            agent_descriptions=agent_descriptions,
            tool_section=tool_section,
        )

        resp = await litellm.acompletion(
            model=self.router_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            stream=False,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            raise RouterError(f"Router returned invalid JSON: {raw!r}") from None

    async def _get_allocated_tools(self, spec: dict[str, Any], agent_name: str) -> list[ToolDef]:
        """Return globally-allocated tools for a routing spec entry."""
        if not self._global_tools:
            return []
        requested_names: list[str] = spec.get("tools", [])
        if requested_names:
            name_set = set(requested_names)
            return [td for td in self._global_tools if td.name in name_set]
        # If router didn't assign tools, use ToolAllocator
        task = spec.get("task", "")
        if task:
            allocator = ToolAllocator(self.router_model)
            return await allocator.allocate(task, self._global_tools)
        return []

    async def _synthesize(
        self,
        original_query: str,
        results: list[AgentResult],
        queue: asyncio.Queue[StreamEvent | None],
    ) -> str:
        """Backward-compat wrapper that delegates to APEX."""
        result = await self._apex.synthesize(original_query, results, queue)
        return result.text

    def _get_agent(self, name: str) -> Agent:
        if name not in self.agents:
            available = ", ".join(self.agents)
            raise RouterError(
                f"Router chose agent {name!r} but it is not in the agent pool. "
                f"Available: {available}"
            )
        return self.agents[name]
