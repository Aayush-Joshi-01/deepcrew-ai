"""
MCP tools example.

Demonstrates all three MCP transport types: stdio, SSE, and HTTP.

Requires: npx (Node.js), OPENAI_API_KEY
Run: python examples/mcp_example.py
"""

import asyncio

from deepcrew import Agent, run_agent
from deepcrew.mcp import HTTPMCP, MCPManager, SSEMCP, StdioMCP


async def stdio_example():
    """Use the official MCP filesystem server via stdio."""
    print("=== StdioMCP: Filesystem Server ===")
    async with StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as mcp:
        tools = await mcp.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        agent = Agent(
            name="file_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You help users navigate and understand their filesystem.",
            mcps=[mcp],
        )

        result = await run_agent(
            agent,
            [{"role": "user", "content": "List the files in the current directory."}],
        )
        print(f"\nResponse:\n{result.text}")


async def http_example():
    """Use a streamable-HTTP MCP server."""
    print("\n=== HTTPMCP: Remote Server ===")
    # Replace with your MCP server URL
    mcp = HTTPMCP(
        "https://your-mcp-server.example.com/mcp",
        headers={"Authorization": "Bearer your-token"},
    )
    async with mcp:
        tools = await mcp.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")


async def sse_example():
    """Use a legacy SSE MCP server."""
    print("\n=== SSEMCP: Legacy SSE Server ===")
    async with SSEMCP("http://localhost:3000") as mcp:
        tools = await mcp.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")


async def manager_example():
    """Combine multiple MCP servers with MCPManager."""
    print("\n=== MCPManager: Multiple Servers ===")
    fs_mcp = StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."])

    async with MCPManager([fs_mcp]) as manager:
        tools = await manager.discover_tools()
        print(f"Total tools from all servers: {len(tools)}")
        print(f"Tool names: {[t.name for t in tools]}")

        # MCPManager itself implements the MCPClient interface
        agent = Agent(
            name="multi_tool_agent",
            model="openai/gpt-4o-mini",
            system_prompt="You are a helpful assistant with access to multiple tools.",
            mcps=[manager],
        )

        result = await run_agent(
            agent,
            [{"role": "user", "content": "What tools do you have access to?"}],
        )
        print(f"\nResponse:\n{result.text}")


if __name__ == "__main__":
    asyncio.run(stdio_example())
    # Uncomment to test other transports:
    # asyncio.run(http_example())
    # asyncio.run(sse_example())
    # asyncio.run(manager_example())
