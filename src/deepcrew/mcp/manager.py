from __future__ import annotations

import asyncio
from typing import Any

from ..exceptions import MCPError
from ..types import ToolDef
from .base import MCPClient


class MCPManager:
    """
    Aggregates multiple MCP clients into a single interface.

    Discovers tools from all attached clients in parallel and maintains a
    routing table so that tool calls are automatically dispatched to the
    correct server.

    Parameters
    ----------
    clients:
        Any combination of StdioMCP, SSEMCP, and HTTPMCP instances.

    Example
    -------
    ::

        manager = MCPManager([
            StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]),
            HTTPMCP("https://my-search-mcp.example.com/mcp"),
        ])
        await manager.connect_all()
        tools = await manager.discover_tools()

        # Pass manager as a single MCP to an Agent
        agent = Agent("assistant", model="openai/gpt-4o", mcps=[manager])
    """

    def __init__(self, clients: list[MCPClient]) -> None:
        self._clients = clients
        self._routing: dict[str, MCPClient] = {}

    async def connect_all(self) -> None:
        """Connect all clients in parallel."""
        await asyncio.gather(*[c.connect() for c in self._clients], return_exceptions=True)

    async def discover_tools(self) -> list[ToolDef]:
        """
        Fetch tools from all servers in parallel and build the routing table.
        Tools from earlier clients take precedence if names collide.
        """
        all_tools: list[ToolDef] = []
        results = await asyncio.gather(
            *[c.list_tools() for c in self._clients],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, list):
                for td in r:
                    if td.name not in self._routing:
                        self._routing[td.name] = td._mcp_client  # type: ignore[assignment]
                    all_tools.append(td)
        return all_tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any] | str:
        client = self._routing.get(name)
        if not client:
            raise MCPError(
                f"No MCP server owns tool {name!r}. "
                "Did you call discover_tools() before calling call_tool()?"
            )
        return await client.call_tool(name, args)

    async def list_tools(self) -> list[ToolDef]:
        """Implement the MCPClient interface so MCPManager can be passed as an MCP."""
        return await self.discover_tools()

    async def close_all(self) -> None:
        """Close all clients, ignoring individual errors."""
        await asyncio.gather(*[c.close() for c in self._clients], return_exceptions=True)

    async def __aenter__(self) -> MCPManager:
        await self.connect_all()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close_all()
