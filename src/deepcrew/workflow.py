from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent
from .exceptions import WorkflowError
from .runner import run_agent
from .stream import make_done_event, make_error_event, queue_to_stream
from .types import AgentResult, EventType, StreamEvent, WorkflowResult


@dataclass
class _WorkflowNode:
    name: str
    agent: Agent
    task_template: str | Callable[[dict[str, Any]], str]


class WorkflowBuilder:
    """
    Build an explicit DAG (directed acyclic graph) of agents.

    Each node is an Agent with a task template. Edges define dependencies:
    a successor node only runs after all its predecessors complete, and
    their outputs are available as template variables.

    Nodes at the same "level" (no pending dependencies between each other)
    run in parallel automatically.

    Example
    -------
    ::

        workflow = (
            WorkflowBuilder()
            .add_agent("research", researcher, task="{input}")
            .add_agent("code", coder, task="{input}")
            .add_agent("report", writer, task=(
                "Combine this research:\\n{research}\\n\\n"
                "And this code:\\n{code}\\n\\n"
                "Into a final report."
            ))
            .then("research", "report")
            .then("code", "report")
        )

        result = await workflow.run("Build a web scraper for news sites")
        print(result.final_output.text)
    """

    def __init__(self) -> None:
        self._nodes: dict[str, _WorkflowNode] = {}
        self._deps: dict[str, set[str]] = {}

    def add_agent(
        self,
        name: str,
        agent: Agent,
        task: str | Callable[[dict[str, Any]], str] = "{input}",
    ) -> WorkflowBuilder:
        """
        Add a named agent node to the workflow.

        Parameters
        ----------
        name:
            Unique node name.
        agent:
            The Agent to run at this node.
        task:
            Task string or callable.

            If a **string**, it is a Python format string where:
            - ``{input}`` is the initial user input passed to ``run()``
            - ``{node_name}`` is the text output of any predecessor node

            If a **callable**, it receives a dict with keys ``"input"`` and
            one key per completed predecessor node (value = their output text).
            It should return the task string.
        """
        if name in self._nodes:
            raise WorkflowError(f"Node {name!r} already exists in this workflow")
        self._nodes[name] = _WorkflowNode(name=name, agent=agent, task_template=task)
        self._deps.setdefault(name, set())
        return self

    def then(self, predecessor: str, successor: str) -> WorkflowBuilder:
        """
        Declare that ``successor`` must run after ``predecessor`` completes.

        The predecessor's output text becomes available as ``{predecessor}``
        in the successor's task template.
        """
        if predecessor not in self._nodes:
            raise WorkflowError(f"Unknown node {predecessor!r}. Add it before creating edges.")
        if successor not in self._nodes:
            raise WorkflowError(f"Unknown node {successor!r}. Add it before creating edges.")
        self._deps.setdefault(successor, set()).add(predecessor)
        return self

    async def run(
        self,
        initial_input: str,
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute the workflow and return the complete result."""
        self._validate()
        levels = _topological_levels(self._nodes, self._deps)
        outputs: dict[str, AgentResult] = {}
        total_in = total_out = 0

        for level in levels:
            tasks = [
                _run_node(self._nodes[name], initial_input, outputs)
                for name in level
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(level, results):
                if isinstance(result, Exception):
                    raise result
                outputs[name] = result
                total_in += result.input_tokens
                total_out += result.output_tokens

        sink = _find_sink(self._nodes, self._deps)
        return WorkflowResult(
            outputs=outputs,
            final_output=outputs.get(sink) if sink else None,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )

    def stream(
        self,
        initial_input: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the workflow, streaming events in real time."""
        self._validate()
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        task = asyncio.create_task(
            self._run_streaming(initial_input, context or {}, queue)
        )
        return queue_to_stream(queue, task)

    async def _run_streaming(
        self,
        initial_input: str,
        context: dict[str, Any],
        queue: asyncio.Queue[StreamEvent | None],
    ) -> None:
        try:
            levels = _topological_levels(self._nodes, self._deps)
            outputs: dict[str, AgentResult] = {}

            for level in levels:
                for name in level:
                    await queue.put(StreamEvent(EventType.STEP_START, {"node": name}, name))

                tasks = [
                    _run_node(self._nodes[name], initial_input, outputs, queue=queue)
                    for name in level
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for name, result in zip(level, results):
                    if isinstance(result, Exception):
                        await queue.put(make_error_event(name, str(result)))
                        await queue.put(None)
                        return
                    outputs[name] = result
                    await queue.put(StreamEvent(EventType.STEP_DONE, {"node": name}, name))

            sink = _find_sink(self._nodes, self._deps)
            final_text = outputs[sink].text if sink and sink in outputs else ""
            await queue.put(StreamEvent(EventType.DONE, {"final_text": final_text}, "workflow"))
        except Exception as exc:
            await queue.put(make_error_event("workflow", str(exc)))
        finally:
            await queue.put(None)

    def _validate(self) -> None:
        if not self._nodes:
            raise WorkflowError("Workflow has no nodes. Add agents with add_agent().")
        if _has_cycle(self._nodes, self._deps):
            raise WorkflowError("Workflow DAG contains a cycle.")


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------

async def _run_node(
    node: _WorkflowNode,
    initial_input: str,
    outputs: dict[str, AgentResult],
    queue: asyncio.Queue[StreamEvent | None] | None = None,
) -> AgentResult:
    task_str = _resolve_task(node.task_template, initial_input, outputs)
    return await run_agent(
        node.agent,
        [{"role": "user", "content": task_str}],
        queue=queue,
        agent_id=node.name,
    )


def _resolve_task(
    template: str | Callable[[dict[str, Any]], str],
    initial_input: str,
    outputs: dict[str, AgentResult],
) -> str:
    ctx: dict[str, Any] = {"input": initial_input}
    ctx.update({k: v.text for k, v in outputs.items()})
    if callable(template):
        return template(ctx)
    try:
        return template.format_map(ctx)
    except KeyError as exc:
        raise WorkflowError(
            f"Task template references {exc} but that node has not run yet. "
            "Add a .then() edge to declare the dependency."
        ) from exc


def _topological_levels(
    nodes: dict[str, Any],
    deps: dict[str, set[str]],
) -> list[list[str]]:
    """Kahn's algorithm — returns nodes grouped into parallel execution levels."""
    in_degree = {name: len(deps.get(name, set())) for name in nodes}
    levels: list[list[str]] = []

    while True:
        ready = [n for n, deg in in_degree.items() if deg == 0]
        if not ready:
            break
        levels.append(sorted(ready))
        for name in ready:
            del in_degree[name]
        # Reduce in-degree for nodes that depended on any completed node
        for succ, succ_deps in deps.items():
            if succ in in_degree:
                remaining = succ_deps & in_degree.keys()
                in_degree[succ] = len(remaining)

    return levels


def _find_sink(
    nodes: dict[str, Any],
    deps: dict[str, set[str]],
) -> str | None:
    """Return the node that no other node depends on (the final output node)."""
    all_predecessors: set[str] = set()
    for preds in deps.values():
        all_predecessors.update(preds)
    sinks = [n for n in nodes if n not in all_predecessors]
    return sinks[-1] if sinks else None


def _has_cycle(
    nodes: dict[str, Any],
    deps: dict[str, set[str]],
) -> bool:
    levels = _topological_levels(nodes, deps)
    visited = {n for level in levels for n in level}
    return len(visited) != len(nodes)
