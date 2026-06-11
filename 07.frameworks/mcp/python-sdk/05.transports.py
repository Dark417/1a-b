"""05 · Transports — stdio vs Streamable HTTP.

MCP's transport layer is orthogonal to its data layer (the JSON-RPC messages are
identical). There are two standard transports:

  * **stdio** — host launches the server as a subprocess and exchanges
    newline-delimited JSON over stdin/stdout. Local, zero network, one client per
    process. This is what Claude Desktop/Code config uses for local servers, and
    what every other file in this folder runs live.

  * **Streamable HTTP** — a single HTTP endpoint (e.g. ``/mcp``). The client
    ``POST``s JSON-RPC; the server replies with either a JSON body *or* a
    **Server-Sent Events** stream (for incremental output and server→client
    messages). Supports OAuth 2.1 / bearer auth, multiple clients, and sessions
    via the ``Mcp-Session-Id`` header. This is the transport for *remote* servers.
    (It superseded the older HTTP+SSE transport from the 2024 spec.)

This file:
  1. prints a side-by-side comparison;
  2. builds the Streamable-HTTP **ASGI app** object from a FastMCP server to show
     the wiring (without binding a port — so it stays offline and exits 0);
  3. runs a **live stdio** round-trip so you see the data layer is unchanged.

Docs: https://modelcontextprotocol.io/specification/2025-06-18/basic/transports
"""

from __future__ import annotations

import asyncio
import os
import sys

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server.fastmcp import FastMCP

    HAVE_MCP = True
except Exception as exc:  # pragma: no cover
    HAVE_MCP = False
    _IMPORT_ERROR = exc

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "02.fastmcp_server.py")


COMPARISON = """\
                stdio                        Streamable HTTP
 location        same machine                local or remote
 process model   1 subprocess / client       1 web service, many clients
 framing         newline-delimited JSON       HTTP POST (+ SSE for streaming)
 server→client   via the single pipe          via the SSE channel
 auth            OS process boundary          OAuth 2.1 / bearer tokens
 sessions        implicit (the process)       Mcp-Session-Id header
 logging         server stderr                server logs / response
 best for        dev, Claude Desktop/Code     shared/remote servers, SaaS
 run command     mcp.run(transport="stdio")   mcp.run(transport="streamable-http")
"""


def show_http_wiring() -> None:
    """Build (don't serve) the Streamable-HTTP ASGI app to show the wiring."""
    mcp = FastMCP(name="http-demo")

    @mcp.tool()
    def ping() -> str:
        """Health check."""
        return "pong"

    try:
        # FastMCP exposes an ASGI app you mount in Starlette/FastAPI and serve
        # with uvicorn. We build the object but never bind a socket, so this is
        # fully offline.
        app = mcp.streamable_http_app()
        print("[http] built ASGI app:", type(app).__name__)
        print("[http] to serve for real:  uvicorn 05.transports:app --port 8000")
        print("[http] then clients connect to  http://127.0.0.1:8000/mcp")
    except Exception as exc:
        print(f"[http] (skipped building ASGI app: {exc})")


async def live_stdio() -> None:
    params = StdioServerParameters(command=sys.executable, args=[SERVER, "--stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            r = await session.call_tool("add", {"a": 2, "b": 3})
            text = " ".join(getattr(b, "text", "") for b in r.content)
            print("[stdio] live add(2,3) ->", text)


def main() -> int:
    print("MCP transports\n")
    print(COMPARISON)

    if not HAVE_MCP:
        print("[skip] `mcp` not installed — comparison above is conceptual.")
        print(f"       ({_IMPORT_ERROR})")
        return 0

    print("Streamable HTTP wiring (built, not served):")
    show_http_wiring()
    print("\nLive stdio round-trip (data layer is identical either transport):")
    asyncio.run(live_stdio())
    print("\nSame JSON-RPC messages; only the bytes' path differs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
