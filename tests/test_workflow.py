from __future__ import annotations

import pytest

from deepcrew.exceptions import WorkflowError
from deepcrew.workflow import (
    WorkflowBuilder,
    _find_sink,
    _has_cycle,
    _resolve_task,
    _topological_levels,
)
from deepcrew.types import AgentResult


def _make_result(agent_id: str, text: str) -> AgentResult:
    return AgentResult(agent_id=agent_id, text=text, model="test")


def test_topological_levels_linear():
    nodes = {"a": None, "b": None, "c": None}
    deps = {"a": set(), "b": {"a"}, "c": {"b"}}
    levels = _topological_levels(nodes, deps)
    assert levels == [["a"], ["b"], ["c"]]


def test_topological_levels_parallel():
    nodes = {"a": None, "b": None, "c": None}
    deps = {"a": set(), "b": set(), "c": {"a", "b"}}
    levels = _topological_levels(nodes, deps)
    assert levels[0] == ["a", "b"]
    assert levels[1] == ["c"]


def test_topological_levels_single():
    nodes = {"x": None}
    deps = {"x": set()}
    assert _topological_levels(nodes, deps) == [["x"]]


def test_find_sink_linear():
    nodes = {"a": None, "b": None}
    deps = {"a": set(), "b": {"a"}}
    assert _find_sink(nodes, deps) == "b"


def test_has_cycle_detects_cycle():
    nodes = {"a": None, "b": None}
    deps = {"a": {"b"}, "b": {"a"}}
    assert _has_cycle(nodes, deps) is True


def test_has_cycle_no_cycle():
    nodes = {"a": None, "b": None}
    deps = {"a": set(), "b": {"a"}}
    assert _has_cycle(nodes, deps) is False


def test_resolve_task_string_template():
    outputs = {"research": _make_result("research", "quantum facts")}
    result = _resolve_task("Write about: {research}", "original", outputs)
    assert result == "Write about: quantum facts"


def test_resolve_task_callable():
    outputs = {"step1": _make_result("step1", "step1 output")}
    fn = lambda ctx: f"Input was: {ctx['input']}, step1: {ctx['step1']}"
    result = _resolve_task(fn, "my input", outputs)
    assert result == "Input was: my input, step1: step1 output"


def test_workflow_builder_duplicate_node_raises():
    from deepcrew.agent import Agent
    wb = WorkflowBuilder()
    agent = Agent("a", model="openai/gpt-4o")
    wb.add_agent("node", agent)
    with pytest.raises(WorkflowError, match="already exists"):
        wb.add_agent("node", agent)


def test_workflow_builder_unknown_edge_raises():
    from deepcrew.agent import Agent
    wb = WorkflowBuilder()
    agent = Agent("a", model="openai/gpt-4o")
    wb.add_agent("node1", agent)
    with pytest.raises(WorkflowError, match="Unknown node"):
        wb.then("node1", "nonexistent")


def test_workflow_builder_empty_raises():
    wb = WorkflowBuilder()
    with pytest.raises(WorkflowError, match="no nodes"):
        wb._validate()


def test_workflow_builder_cycle_raises():
    from deepcrew.agent import Agent
    wb = WorkflowBuilder()
    a = Agent("a", model="openai/gpt-4o")
    b = Agent("b", model="openai/gpt-4o")
    wb.add_agent("x", a)
    wb.add_agent("y", b)
    wb._deps["x"].add("y")
    wb._deps["y"].add("x")
    with pytest.raises(WorkflowError, match="cycle"):
        wb._validate()
