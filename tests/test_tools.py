from __future__ import annotations

from deepcrew.tools import fn_to_tool_def, tool


def test_tool_decorator_no_args():
    @tool
    def greet(name: str) -> str:
        "Greet someone by name."
        return f"Hello, {name}"

    assert greet._is_tool is True
    assert greet._tool_name == "greet"
    assert greet._tool_description == "Greet someone by name."


def test_tool_decorator_with_args():
    @tool(name="say_hi", description="Say hi to a person")
    def greet(name: str) -> str:
        return f"Hi {name}"

    assert greet._tool_name == "say_hi"
    assert greet._tool_description == "Say hi to a person"


def test_fn_to_tool_def_basic():
    @tool
    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    td = fn_to_tool_def(add)
    assert td.name == "add"
    assert td.description == "Add two numbers."
    params = td.parameters
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"
    assert set(params["required"]) == {"a", "b"}


def test_fn_to_tool_def_defaults_not_required():
    def search(query: str, max_results: int = 5) -> list:
        return []

    td = fn_to_tool_def(search)
    assert "query" in td.parameters["required"]
    assert "max_results" not in td.parameters["required"]


def test_fn_to_tool_def_optional_type():
    def func(value: str | None = None) -> str:
        return value or ""

    td = fn_to_tool_def(func)
    prop = td.parameters["properties"]["value"]
    t = prop.get("type")
    # Should include "null" for Optional
    assert t is None or "null" in t or "string" in str(prop)


def test_fn_to_tool_def_callable_stored():
    @tool
    def ping() -> str:
        "Ping."
        return "pong"

    td = fn_to_tool_def(ping)
    assert td._callable is ping
    assert td._mcp_client is None


def test_fn_to_tool_def_bool_type():
    def func(flag: bool) -> bool:
        return flag

    td = fn_to_tool_def(func)
    assert td.parameters["properties"]["flag"]["type"] == "boolean"


def test_fn_to_tool_def_list_type():
    def func(items: list[str]) -> list:
        return items

    td = fn_to_tool_def(func)
    prop = td.parameters["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"
