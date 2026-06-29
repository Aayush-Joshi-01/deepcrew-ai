from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..types import ToolDef


class Skill(ABC):
    """
    A higher-level reusable capability bundle that surfaces to the LLM as a tool.

    Skills differ from atomic @tool functions in that their execute() method can
    contain multi-step logic, call other agents, or run sub-workflows.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Run the skill and return a string result."""

    def to_tool_def(self) -> ToolDef:
        """Expose this skill to the LLM as a ToolDef."""
        skill_self = self

        async def _run(**kwargs: Any) -> str:
            return await skill_self.execute(**kwargs)

        return ToolDef(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            _callable=_run,
        )


def skill(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """
    Decorator that wraps an async function into a Skill instance.

    Usage::

        @skill
        async def my_skill(query: str) -> str:
            return f"Result for {query}"

        @skill(name="custom_name", description="Does X")
        async def my_skill(query: str) -> str:
            ...
    """
    import inspect

    def _make_skill(func: Callable[..., Any]) -> "FunctionSkill":
        skill_name = name or func.__name__
        skill_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        sig = inspect.signature(func)
        props: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in sig.parameters.items():
            ann = param.annotation
            json_type = "string"
            if ann is int:
                json_type = "integer"
            elif ann is float:
                json_type = "number"
            elif ann is bool:
                json_type = "boolean"
            props[pname] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        parameters = {"type": "object", "properties": props}
        if required:
            parameters["required"] = required

        instance = FunctionSkill(
            _fn=func,
            _name=skill_name,
            _description=skill_desc,
            _parameters=parameters,
        )
        return instance

    if fn is not None:
        return _make_skill(fn)
    return _make_skill


class FunctionSkill(Skill):
    """A Skill created from a plain async function via the @skill decorator."""

    def __init__(
        self,
        _fn: Callable[..., Any],
        _name: str,
        _description: str,
        _parameters: dict[str, Any],
    ) -> None:
        self._fn = _fn
        self.name = _name
        self.description = _description
        self.parameters = _parameters

    async def execute(self, **kwargs: Any) -> str:
        import asyncio

        result = self._fn(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result)
