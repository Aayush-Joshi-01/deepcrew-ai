from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepcrew.exceptions import MCPError
from deepcrew.mcp.base import _parse_mcp_tool
from deepcrew.mcp.http import HTTPMCP
from deepcrew.mcp.manager import MCPManager


def test_parse_mcp_tool_basic():
    raw = {
        "name": "search",
        "description": "Search the web",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    client = MagicMock()
    td = _parse_mcp_tool(raw, client)
    assert td.name == "search"
    assert td.description == "Search the web"
    assert td._mcp_client is client
    assert td.parameters["type"] == "object"


def test_parse_mcp_tool_no_schema():
    raw = {"name": "ping", "description": "Ping"}
    td = _parse_mcp_tool(raw, MagicMock())
    assert td.parameters == {}


@pytest.mark.asyncio
async def test_http_mcp_list_tools():
    mcp = HTTPMCP("https://example.com/mcp")

    init_resp = MagicMock()
    init_resp.headers = {"mcp-session-id": "sess-123"}
    init_resp.json.return_value = {"result": {"protocolVersion": "2024-11-05"}}
    init_resp.raise_for_status = MagicMock()
    init_resp.status_code = 200

    list_resp = MagicMock()
    list_resp.headers = {}
    list_resp.json.return_value = {
        "result": {
            "tools": [
                {
                    "name": "search",
                    "description": "Search",
                    "inputSchema": {"type": "object", "properties": {}, "required": []},
                },
            ]
        }
    }
    list_resp.raise_for_status = MagicMock()
    list_resp.status_code = 200

    notif_resp = MagicMock()
    notif_resp.raise_for_status = MagicMock()

    call_seq = [init_resp, notif_resp, list_resp]

    async def fake_post(url, **kwargs):
        return call_seq.pop(0)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client):
        await mcp.connect()
        tools = await mcp.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "search"
    assert mcp._session_id == "sess-123"


@pytest.mark.asyncio
async def test_mcp_manager_discover_tools():
    mock_mcp1 = AsyncMock()
    mock_mcp1.list_tools = AsyncMock(
        return_value=[
            MagicMock(name="tool1", _mcp_client=mock_mcp1),
        ]
    )
    mock_mcp1.list_tools.return_value[0].name = "tool1"

    mock_mcp2 = AsyncMock()
    mock_mcp2.list_tools = AsyncMock(
        return_value=[
            MagicMock(name="tool2", _mcp_client=mock_mcp2),
        ]
    )
    mock_mcp2.list_tools.return_value[0].name = "tool2"

    manager = MCPManager([mock_mcp1, mock_mcp2])
    tools = await manager.discover_tools()
    assert len(tools) == 2


@pytest.mark.asyncio
async def test_mcp_manager_call_tool_routing():
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value={"result": "ok"})

    from deepcrew.types import ToolDef

    fake_tool = ToolDef(name="my_tool", description="", parameters={}, _mcp_client=mock_client)

    mock_mcp = AsyncMock()
    mock_mcp.list_tools = AsyncMock(return_value=[fake_tool])

    manager = MCPManager([mock_mcp])
    await manager.discover_tools()
    await manager.call_tool("my_tool", {"x": 1})

    mock_client.call_tool.assert_awaited_once_with("my_tool", {"x": 1})


@pytest.mark.asyncio
async def test_mcp_manager_unknown_tool_raises():
    manager = MCPManager([])
    with pytest.raises(MCPError, match="No MCP server"):
        await manager.call_tool("ghost_tool", {})
