from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .types import ToolDef


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable:
    """
    Decorator to register a Python function as an agent tool.

    Can be used with or without arguments::

        @tool
        def search_web(query: str, max_results: int = 5) -> str:
            "Search the web and return results."
            ...

        @tool(name="web_search", description="Search the internet")
        def search_web(query: str) -> str:
            ...
    """

    def decorator(f: Callable) -> Callable:
        f._is_tool = True  # type: ignore[attr-defined]
        f._tool_name = name or f.__name__  # type: ignore[attr-defined]
        f._tool_description = description or _first_docline(f)  # type: ignore[attr-defined]
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def fn_to_tool_def(fn: Callable) -> ToolDef:
    """Convert a Python function (optionally decorated with @tool) to a ToolDef.

    Generates a JSON Schema for the function parameters from type hints.
    Supports: str, int, float, bool, list, dict, Optional, Union, Literal, None.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    param_docs = _parse_param_docs(fn)

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "return"):
            continue
        hint = hints.get(param_name, str)
        schema = _type_to_jsonschema(hint)
        doc = param_docs.get(param_name, "")
        if doc:
            schema["description"] = doc
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return ToolDef(
        name=getattr(fn, "_tool_name", fn.__name__),
        description=getattr(fn, "_tool_description", _first_docline(fn)),
        parameters={"type": "object", "properties": properties, "required": required},
        _callable=fn,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_docline(fn: Callable) -> str:
    return (fn.__doc__ or "").strip().split("\n")[0].strip()


def _parse_param_docs(fn: Callable) -> dict[str, str]:
    """Extract parameter descriptions from Google-style or plain docstrings."""
    doc = fn.__doc__ or ""
    result: dict[str, str] = {}
    # Google-style: "    param_name (type): description"
    for m in re.finditer(r"^\s{4,8}(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", doc, re.MULTILINE):
        result[m.group(1)] = m.group(2).strip()
    return result


def _type_to_jsonschema(hint: Any) -> dict[str, Any]:
    """Recursively convert Python type hints to a JSON Schema dict."""
    if hint is type(None):
        return {"type": "null"}

    origin = get_origin(hint)
    args = get_args(hint)

    # Optional[X] → Union[X, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        if len(non_none) == 1:
            schema = _type_to_jsonschema(non_none[0])
            if has_none:
                existing = schema.get("type")
                if existing and isinstance(existing, str):
                    schema["type"] = [existing, "null"]
                elif existing and isinstance(existing, list) and "null" not in existing:
                    schema["type"] = [*existing, "null"]
            return schema
        return {"anyOf": [_type_to_jsonschema(a) for a in args]}

    if origin is Literal:
        return {"enum": list(args)}

    if origin is list:
        item_schema = _type_to_jsonschema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin is dict:
        value_schema = _type_to_jsonschema(args[1]) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": value_schema}

    _PRIMITIVES = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        bytes: "string",
        Any: {},
    }
    if hint in _PRIMITIVES:
        v = _PRIMITIVES[hint]
        return v if isinstance(v, dict) else {"type": v}

    # Fallback for unrecognized types
    return {"type": "string"}
