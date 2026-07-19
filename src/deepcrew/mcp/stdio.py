from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from ..exceptions import MCPError
from ..types import ToolDef
from .base import MCPClient, _parse_mcp_tool

logger = logging.getLogger(__name__)


class StdioMCP(MCPClient):
    """
    MCP client that spawns a subprocess and communicates via JSON-RPC over
    stdin/stdout (the stdio transport defined by the MCP specification).

    Parameters
    ----------
    command:
        Executable to run, e.g. ``"npx"`` or ``"python"``.
    args:
        Command-line arguments passed to the executable.
    env:
        Extra environment variables merged with the current process environment.

    Example
    -------
    ::

        async with StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as mcp:
            tools = await mcp.list_tools()
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.args: list[str] = args or []
        self.env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._msg_id = 0
        self._tool_cache: list[ToolDef] | None = None

    async def connect(self) -> None:
        logger.info("StdioMCP connecting: %s %s", self.command, self.args)
        merged_env = {**os.environ, **(self.env or {})}
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        await self._handshake()

    async def _send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise MCPError("StdioMCP is not connected. Call connect() first.")

        self._msg_id += 1
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._msg_id,
            "method": method,
            "params": params or {},
        }
        line = json.dumps(msg) + "\n"

        async with self._lock:
            self._proc.stdin.write(line.encode())
            await self._proc.stdin.drain()
            raw = await self._proc.stdout.readline()

        if not raw:
            raise MCPError("StdioMCP subprocess closed its stdout unexpectedly")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MCPError(f"StdioMCP returned non-JSON: {raw!r}") from exc

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPError("StdioMCP is not connected.")
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        await self._proc.stdin.drain()

    async def _handshake(self) -> None:
        resp = await self._send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "deepcrew", "version": "0.1.0"},
            },
        )
        if "error" in resp:
            logger.error("StdioMCP initialize failed: %s", resp["error"])
            raise MCPError(f"StdioMCP initialize failed: {resp['error']}")
        await self._notify("notifications/initialized")

    async def list_tools(self) -> list[ToolDef]:
        if self._tool_cache is not None:
            return self._tool_cache
        resp = await self._send("tools/list")
        if "error" in resp:
            logger.error("StdioMCP tools/list failed: %s", resp["error"])
            raise MCPError(f"tools/list failed: {resp['error']}")
        raw_tools: list[dict] = resp.get("result", {}).get("tools", [])
        self._tool_cache = [_parse_mcp_tool(t, self) for t in raw_tools]
        return self._tool_cache

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any] | str:
        resp = await self._send("tools/call", {"name": name, "arguments": args})
        if "error" in resp:
            logger.error("StdioMCP tool %r failed: %s", name, resp["error"])
            raise MCPError(f"Tool {name!r} failed: {resp['error']}")
        result = resp.get("result", {})
        # MCP returns content array; extract text or return raw
        if isinstance(result, dict) and "content" in result:
            contents = result["content"]
            if contents and isinstance(contents[0], dict):
                return contents[0].get("text", json.dumps(result))
        return result

    async def close(self) -> None:
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (TimeoutError, Exception):
                self._proc.kill()
            finally:
                self._proc = None
