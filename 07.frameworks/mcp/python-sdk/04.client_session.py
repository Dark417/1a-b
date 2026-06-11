"""04 · The official MCP client — `ClientSession` over stdio.

The SDK's client is ``ClientSession`` wrapped around a transport. For local
servers the transport is ``stdio_client(StdioServerParameters(...))``, which
launches the server as a subprocess and wires up its stdin/stdout. ``async with``
both context managers, then ``await session.initialize()`` runs the full
lifecycle for you (``initialize`` + ``notifications/initialized``).

This client connects to the FastMCP server from ``02.fastmcp_server.py --stdio``
and exercises **every primitive**:

  * ``list_tools`` / ``call_tool``
  * ``list_resources`` / ``list_resource_templates`` / ``read_resource``
  * ``list_prompts`` / ``get_prompt``

It runs **offline**: the "remote" service is just another Python process on this
machine, spoken to over pipes. If ``mcp`` is not installed it skips gracefully.

Docs: https://github.com/modelcontextprotocol/python-sdk#writing-mcp-clients
"""

from __future__ import annotations

import asyncio
import os
import sys

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    HAVE_MCP = True
except Exception as exc:  # pragma: no cover
    HAVE_MCP = False
    _IMPORT_ERROR = exc

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "02.fastmcp_server.py")


def _text(result) -> str:
    """Flatten a CallToolResult's content blocks into text."""
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", str(block)))
    return " ".join(parts)


async def run() -> None:
    params = StdioServerParameters(command=sys.executable, args=[SERVER, "--stdio"])

    # stdio_client spawns the server; ClientSession speaks JSON-RPC over it.
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("connected to:", init.serverInfo.name, init.serverInfo.version)
            print("server capabilities:", init.capabilities.model_dump(exclude_none=True))

            print("\n— TOOLS —")
            tools = (await session.list_tools()).tools
            for t in tools:
                print(f"  • {t.name}: {t.description}")
            r = await session.call_tool("add", {"a": 40, "b": 2})
            print("  add(40,2)            ->", _text(r))
            r = await session.call_tool("slugify", {"text": "Hello, MCP World!"})
            print("  slugify('Hello...')  ->", _text(r))
            r = await session.call_tool("batch_add", {"numbers": [1, 2, 3, 4]})
            print("  batch_add([1,2,3,4]) ->", _text(r))

            print("\n— RESOURCES —")
            for res in (await session.list_resources()).resources:
                print(f"  • {res.uri} ({res.name})")
            for tmpl in (await session.list_resource_templates()).resourceTemplates:
                print(f"  • template {tmpl.uriTemplate}")
            content = await session.read_resource("config://server")
            print("  read config://server ->", repr(content.contents[0].text.splitlines()[0]))
            greet = await session.read_resource("greeting://Ada")
            print("  read greeting://Ada  ->", repr(greet.contents[0].text))

            print("\n— PROMPTS —")
            for p in (await session.list_prompts()).prompts:
                print(f"  • {p.name}({', '.join(a.name for a in (p.arguments or []))})")
            got = await session.get_prompt("code_review", {"code": "x=1", "language": "python"})
            msg = got.messages[0]
            print("  get_prompt code_review ->", getattr(msg.content, "text", "")[:60], "...")

    print("\nRound-trip complete: one real MCP client ↔ server, offline.")


def main() -> int:
    if not HAVE_MCP:
        print("[skip] `mcp` not installed — pip install -r requirements.txt")
        print(f"       ({_IMPORT_ERROR})")
        print("       (01.jsonrpc_wire.py shows the same round-trip with stdlib only.)")
        return 0
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
