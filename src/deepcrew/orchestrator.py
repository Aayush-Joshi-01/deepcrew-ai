from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import litellm

from .agent import Agent
from .exceptions import RouterError
from .runner import run_agent, _tool_def_to_litellm
from .stream import make_done_event, make_error_event, queue_to_stream
from .types import AgentResult, EventType, OrchestratorResult, StreamEvent

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

Respond with ONLY valid JSON, exactly one of:
  {{"route": "single", "agent": "<name>", "task": "<specific task for the agent>"}}
  {{"route": "parallel", "agents": [{{"name": "<name>", "task": "<specific task>"}}, ...]}}
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
    Automated multi-agent orchestrator.

    Runs a three-stage pipeline:

    1. **Router** — a lightweight LLM call that reads the user query and the
       list of available agents, then decides whether to route to a single
       agent or fan out to multiple agents running in parallel.
    2. **Execution** — the chosen agent(s) run via :func:`run_agent`, either
       one at a time or concurrently via ``asyncio.gather``.
    3. **Synthesis** — if multiple agents ran, a synthesizer LLM merges their
       outputs into a single cohesive response.

    Parameters
    ----------
    agents:
        The pool of agents the orchestrator can choose from.
    router_model:
        LiteLLM model string used for the routing decision.
    synthesizer_model:
        LiteLLM model string for the synthesis step. Defaults to
        ``router_model``.
    router_system_prompt:
        Override the built-in router system prompt.
    synthesizer_system_prompt:
        Override the built-in synthesizer system prompt.
    max_parallel_agents:
        Cap on the number of agents the router may spawn in parallel.

    Example
    -------
    ::

        orch = Orchestrator(
            agents=[researcher, analyst, writer],
            router_model="openai/gpt-4o-mini",
            synthesizer_model="openai/gpt-4o",
        )

        # Non-streaming
        result = await orch.run("Summarize the state of quantum computing")

        # Streaming (yields StreamEvent objects)
        async for event in orch.stream("Summarize the state of quantum computing"):
            print(event.to_sse(), end="")
    """

    def __init__(
        self,
        agents: list[Agent],
        router_model: str = "openai/gpt-4o-mini",
        synthesizer_model: str | None = None,
        router_system_prompt: str | None = None,
        synthesizer_system_prompt: str | None = None,
        max_parallel_agents: int = 5,
    ) -> None:
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self.router_model = router_model
        self.synthesizer_model = synthesizer_model or router_model
        self._router_system = router_system_prompt or _ROUTER_SYSTEM
        self._synth_system = synthesizer_system_prompt or _SYNTHESIZER_SYSTEM
        self.max_parallel_agents = max_parallel_agents

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
                agent_results.append(AgentResult(
                    agent_id=ev.agent_id,
                    text=text,
                    input_tokens=ev.data.get("input_tokens", 0),
                    output_tokens=ev.data.get("output_tokens", 0),
                    model=current_agent.get(ev.agent_id, ""),
                ))
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
            user_msg = [{"role": "user", "content": query}]

            if routing["route"] == "single":
                agent_name = routing.get("agent", "")
                agent = self._get_agent(agent_name)
                task_override = routing.get("task", query)
                result = await run_agent(
                    agent,
                    [{"role": "user", "content": task_override}],
                    queue=queue,
                    agent_id=agent_name,
                )
                agent_results.append(result)
                final_text = result.text

            else:
                specs = routing.get("agents", [])[: self.max_parallel_agents]
                parallel_results = await asyncio.gather(
                    *[
                        run_agent(
                            self._get_agent(spec["name"]),
                            [{"role": "user", "content": spec.get("task", query)}],
                            queue=queue,
                            agent_id=spec["name"],
                        )
                        for spec in specs
                    ],
                    return_exceptions=True,
                )

                for r in parallel_results:
                    if isinstance(r, Exception):
                        await queue.put(make_error_event("orchestrator", str(r)))
                    else:
                        agent_results.append(r)

                final_text = await self._synthesize(query, agent_results, queue)

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
        system = self._router_system.format(agent_descriptions=agent_descriptions)

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
            # Try to extract JSON from the response
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            raise RouterError(f"Router returned invalid JSON: {raw!r}")

    async def _synthesize(
        self,
        original_query: str,
        results: list[AgentResult],
        queue: asyncio.Queue[StreamEvent | None],
    ) -> str:
        parts = [f"Original query: {original_query}\n"]
        for r in results:
            parts.append(f"--- {r.agent_id} ---\n{r.text}\n")

        synthesis_prompt = "\n".join(parts)

        synth_agent = Agent(
            name="synthesizer",
            model=self.synthesizer_model,
            system_prompt=self._synth_system,
        )
        result = await run_agent(
            synth_agent,
            [{"role": "user", "content": synthesis_prompt}],
            queue=queue,
            agent_id="synthesizer",
        )
        return result.text

    def _get_agent(self, name: str) -> Agent:
        if name not in self.agents:
            available = ", ".join(self.agents)
            raise RouterError(
                f"Router chose agent {name!r} but it is not in the agent pool. "
                f"Available: {available}"
            )
        return self.agents[name]
