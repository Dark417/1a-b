"""02 · The in-memory client — a real MCP round-trip, zero transport.

FastMCP's standout testing feature: ``Client(server_object)`` connects a real MCP
client straight to a server object **in the same process**. There is no
subprocess and no socket, yet the full lifecycle, schemas, and content blocks are
exercised exactly as over stdio/HTTP. This makes MCP round-trips instant and
perfectly offline — ideal for tests and tutorials.

Here we build a small server and drive *every* primitive through the in-memory
client: ``list_tools``/``call_tool``, ``list_resources``/``read_resource``
(fixed + templated), and ``list_prompts``/``get_prompt``.

Run:  python 02.in_memory_client.py   (exit 0; skips if fastmcp missing)
Docs: https://gofastmcp.com/clients/client  (In-Memory Transport)
"""

from __future__ import annotations

import asyncio
import sys

try:
    from fastmcp import Client, FastMCP

    HAVE = True
except Exception as exc:  # pragma: no cover
    HAVE = False
    _ERR = exc


def build() -> "FastMCP":
    mcp = FastMCP("inmem-demo")

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @mcp.tool
    def upper(text: str) -> str:
        """Uppercase a string."""
        return text.upper()

    @mcp.resource("config://app")
    def config() -> str:
        """Static config."""
        return "version=1.0"

    @mcp.resource("greeting://{name}")
    def greet(name: str) -> str:
        """Templated greeting."""
        return f"Hello, {name}!"

    @mcp.prompt
    def summarize(text: str) -> str:
        """Summarization prompt."""
        return f"Summarize:\n\n{text}"

    return mcp


def _tool_text(result) -> str:
    # FastMCP's CallToolResult: .data is structured; .content holds blocks.
    if getattr(result, "data", None) is not None:
        return str(result.data)
    return " ".join(getattr(b, "text", str(b)) for b in result.content)


async def run() -> None:
    mcp = build()
    async with Client(mcp) as client:  # <- in-memory transport
        print("[client] connected (in-memory)\n")

        print("— TOOLS —")
        for t in await client.list_tools():
            print(f"  • {t.name}: {t.description}")
        print("  add(40,2)      ->", _tool_text(await client.call_tool("add", {"a": 40, "b": 2})))
        print("  upper('mcp')   ->", _tool_text(await client.call_tool("upper", {"text": "mcp"})))

        print("\n— RESOURCES —")
        for r in await client.list_resources():
            print(f"  • {r.uri}")
        cfg = await client.read_resource("config://app")
        print("  read config://app    ->", cfg[0].text)
        g = await client.read_resource("greeting://Ada")
        print("  read greeting://Ada  ->", g[0].text)

        print("\n— PROMPTS —")
        for p in await client.list_prompts():
            print(f"  • {p.name}")
        got = await client.get_prompt("summarize", {"text": "MCP makes tools portable."})
        print("  get_prompt summarize ->", got.messages[0].content.text)

    print("\nReal MCP round-trip, no transport, fully offline.")


def main() -> int:
    if not HAVE:
        print(f"[skip] fastmcp not installed ({_ERR}) — pip install -r requirements.txt")
        return 0
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
