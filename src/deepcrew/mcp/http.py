from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

import httpx

from ..exceptions import MCPError
from ..types import ToolDef
from .base import MCPClient, _parse_mcp_tool

logger = logging.getLogger(__name__)


class HTTPMCP(MCPClient):
    """
    MCP client using the modern streamable-HTTP transport (MCP spec 2024-11-05+).

    Performs an initialize handshake to obtain a ``Mcp-Session-Id`` header,
    then uses that session ID for all subsequent requests. Supports automatic
    session re-initialization on 400/401 responses and exponential-backoff
    retries.

    Parameters
    ----------
    url:
        Full URL of the MCP server endpoint, e.g. ``"https://example.com/mcp"``.
    headers:
        Extra HTTP headers (e.g. ``{"Authorization": "Bearer token"}``).
    timeout:
        Request timeout in seconds.
    retries:
        Number of retry attempts on transient errors.

    Example
    -------
    ::

        async with HTTPMCP("https://my-mcp.example.com/mcp",
                           headers={"Authorization": "Bearer sk-..."}) as mcp:
            tools = await mcp.list_tools()
            result = await mcp.call_tool("search", {"query": "hello"})
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        retries: int = 2,
    ) -> None:
        self.url = url.rstrip("/")
        self._extra_headers: dict[str, str] = headers or {}
        self._timeout = timeout
        self._retries = retries
        self._session_id: str | None = None
        self._http: httpx.AsyncClient | None = None
        self._tool_cache: list[ToolDef] | None = None
        self._msg_id = 0

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self._extra_headers}
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    async def connect(self) -> None:
        logger.info("HTTPMCP connecting: %s", self.url)
        self._http = httpx.AsyncClient(timeout=self._timeout)
        await self._initialize()

    async def _initialize(self) -> None:
        self._msg_id += 1
        resp = await self._raw_post(
            {
                "jsonrpc": "2.0",
                "id": self._msg_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "deepcrew", "version": "0.1.0"},
                },
            },
            skip_session=True,
        )
        self._session_id = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        body = resp.json()
        if "error" in body:
            logger.error("HTTPMCP initialize error: %s", body["error"])
            raise MCPError(f"MCP initialize error: {body['error']}")

        # Send notifications/initialized (fire-and-forget)
        if self._http:
            with contextlib.suppress(Exception):
                await self._http.post(
                    self.url,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers=self._headers(),
                )

    async def _raw_post(
        self, payload: dict[str, Any], skip_session: bool = False
    ) -> httpx.Response:
        if self._http is None:
            raise MCPError("HTTPMCP is not connected. Call connect() first.")
        headers = {"Content-Type": "application/json", **self._extra_headers}
        if not skip_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        resp = await self._http.post(self.url, json=payload, headers=headers)
        return resp

    async def _post(self, payload: dict[str, Any], attempt: int = 0) -> dict[str, Any]:
        try:
            resp = await self._raw_post(payload)
            if resp.status_code in (400, 401):
                # Session expired — re-initialize and retry once
                self._session_id = None
                self._tool_cache = None
                await self._initialize()
                if attempt < self._retries:
                    return await self._post(payload, attempt + 1)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if attempt < self._retries:
                await asyncio.sleep(2**attempt)
                return await self._post(payload, attempt + 1)
            logger.error("HTTPMCP request failed after retries: %s", exc)
            raise MCPError(f"HTTPMCP request failed: {exc}") from exc
        except httpx.RequestError as exc:
            if attempt < self._retries:
                await asyncio.sleep(2**attempt)
                return await self._post(payload, attempt + 1)
            logger.error("HTTPMCP connection error after retries: %s", exc)
            raise MCPError(f"HTTPMCP connection error: {exc}") from exc

    async def list_tools(self) -> list[ToolDef]:
        if self._tool_cache is not None:
            return self._tool_cache
        self._msg_id += 1
        body = await self._post(
            {"jsonrpc": "2.0", "id": self._msg_id, "method": "tools/list", "params": {}}
        )
        if "error" in body:
            logger.error("HTTPMCP tools/list failed: %s", body["error"])
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
            logger.error("HTTPMCP tool %r call failed: %s", name, body["error"])
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
