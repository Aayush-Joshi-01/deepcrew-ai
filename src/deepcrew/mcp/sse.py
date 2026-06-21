from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urljoin

import httpx

from ..exceptions import MCPError
from ..types import ToolDef
from .base import MCPClient, _parse_mcp_tool


class SSEMCP(MCPClient):
    """
    MCP client for the legacy Server-Sent Events transport.

    This transport was used by early MCP servers (pre-2024-11-05). It
    connects to a ``/sse`` endpoint to receive an event that tells it which
    URL to POST requests to.

    Parameters
    ----------
    base_url:
        Base URL of the MCP server, e.g. ``"http://localhost:3000"``.
        The client appends ``/sse`` for the event stream and uses the
        ``endpoint`` event to discover the POST URL.
    headers:
        Extra HTTP headers forwarded with every request.

    Example
    -------
    ::

        async with SSEMCP("http://localhost:3000") as mcp:
            tools = await mcp.list_tools()
    """

    def __init__(self, base_url: str, headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._extra_headers: dict[str, str] = headers or {}
        self._post_url: str | None = None
        self._http: httpx.AsyncClient | None = None
        self._tool_cache: list[ToolDef] | None = None
        self._msg_id = 0

    async def connect(self) -> None:
        self._http = httpx.AsyncClient(timeout=60.0)
        await self._sse_handshake()
        await self._initialize()

    async def _sse_handshake(self) -> None:
        """Subscribe to /sse until we receive the 'endpoint' event."""
        try:
            from httpx_sse import aconnect_sse
        except ImportError as exc:
            raise MCPError(
                "httpx-sse is required for SSEMCP. Install it with: pip install httpx-sse"
            ) from exc

        if self._http is None:
            raise MCPError("SSEMCP not connected")

        sse_url = f"{self.base_url}/sse"
        async with aconnect_sse(
            self._http, "GET", sse_url, headers=self._extra_headers
        ) as event_source:
            async for event in event_source.aiter_sse():
                if event.event == "endpoint":
                    data = event.data.strip()
                    if data.startswith("http"):
                        self._post_url = data
                    else:
                        self._post_url = urljoin(self.base_url, data)
                    return

        raise MCPError("SSE stream closed before receiving 'endpoint' event")

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._http is None or self._post_url is None:
            raise MCPError("SSEMCP is not connected. Call connect() first.")
        resp = await self._http.post(
            self._post_url, json=payload, headers=self._extra_headers
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"result": resp.text}

    async def _initialize(self) -> None:
        self._msg_id += 1
        body = await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._msg_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "deepcrew", "version": "0.1.0"},
                },
            }
        )
        if "error" in body:
            raise MCPError(f"SSEMCP initialize error: {body['error']}")

        # Fire-and-forget notification
        try:
            if self._http and self._post_url:
                await self._http.post(
                    self._post_url,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers=self._extra_headers,
                )
        except Exception:
            pass

    async def list_tools(self) -> list[ToolDef]:
        if self._tool_cache is not None:
            return self._tool_cache
        self._msg_id += 1
        body = await self._post(
            {"jsonrpc": "2.0", "id": self._msg_id, "method": "tools/list", "params": {}}
        )
        if "error" in body:
            raise MCPError(f"tools/list failed: {body['error']}")
        raw_tools: list[dict] = body.get("result", {}).get("tools", [])
        self._tool_cache = [_parse_mcp_tool(t, self) for t in raw_tools]
        return self._tool_cache

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any] | str:
        self._msg_id += 1
        body = await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._msg_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
        )
        if "error" in body:
            raise MCPError(f"Tool {name!r} call failed: {body['error']}")
        result = body.get("result", {})
        if isinstance(result, dict) and "content" in result:
            contents = result["content"]
            if contents and isinstance(contents[0], dict):
                return contents[0].get("text", json.dumps(result))
        return result

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None
