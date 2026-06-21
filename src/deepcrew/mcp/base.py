from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ToolDef


class MCPClient(ABC):
    """
    Abstract base class for all MCP transport implementations.

    Subclasses implement the three transport types defined by the MCP spec:
    - :class:`StdioMCP` — subprocess JSON-RPC over stdin/stdout
    - :class:`SSEMCP` — legacy HTTP/SSE protocol
    - :class:`HTTPMCP` — modern streamable-HTTP protocol

    All clients support the async context manager protocol::

        async with HTTPMCP("https://example.com/mcp") as mcp:
            tools = await mcp.list_tools()
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection and run the MCP initialize handshake."""

    @abstractmethod
    async def list_tools(self) -> list[ToolDef]:
        """Return all tools exposed by this MCP server."""

    @abstractmethod
    async def call_tool(self, name: str, args: dict) -> dict | str:
        """Invoke a tool and return its result."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection and release resources."""

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def _parse_mcp_tool(raw: dict, client: MCPClient) -> ToolDef:
    """Convert the raw tools/list entry from an MCP server into a ToolDef."""
    schema = raw.get("inputSchema") or raw.get("input_schema") or {}
    return ToolDef(
        name=raw["name"],
        description=raw.get("description", ""),
        parameters=schema,
        _mcp_client=client,
    )
