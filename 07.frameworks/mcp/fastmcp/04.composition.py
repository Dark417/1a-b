"""04 · Server composition — build big servers from small ones.

Real systems split a server into focused modules. FastMCP lets you compose them:

  * ``await main.import_server(sub, prefix="math")`` — a **static copy**: the
    sub-server's tools/resources are folded into ``main`` once, renamed with the
    prefix (``math_add`` etc.). Simple and fast.
  * ``main.mount(sub, prefix="text")`` — a **live link**: requests are delegated
    to the mounted server at call time, so later changes to ``sub`` show up.

This is how you keep a large MCP surface maintainable, and how you wrap/reuse
existing servers (including, with the standalone package, *proxying* a remote MCP
server or generating one from an OpenAPI/FastAPI app).

We build two small servers, compose them into one, and call the prefixed tools
through the in-memory client — offline.

Run:  python 04.composition.py   (exit 0; skips if fastmcp missing)
Docs: https://gofastmcp.com/servers/composition
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


def build_math() -> "FastMCP":
    m = FastMCP("math")

    @m.tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    @m.tool
    def mul(a: int, b: int) -> int:
        """Multiply."""
        return a * b

    return m


def build_text() -> "FastMCP":
    t = FastMCP("text")

    @t.tool
    def upper(s: str) -> str:
        """Uppercase."""
        return s.upper()

    @t.tool
    def reverse(s: str) -> str:
        """Reverse."""
        return s[::-1]

    return t


def _text(result) -> str:
    if getattr(result, "data", None) is not None:
        return str(result.data)
    return " ".join(getattr(b, "text", str(b)) for b in result.content)


async def run() -> None:
    main = FastMCP("composed-app")

    # import_server: a static merge (copy). mount: a live delegation.
    await main.import_server(build_math(), prefix="math")
    main.mount(build_text(), prefix="text")

    async with Client(main) as client:
        names = [t.name for t in await client.list_tools()]
        print("composed tool surface:", names, "\n")

        print("math_add(2,3)        ->", _text(await client.call_tool("math_add", {"a": 2, "b": 3})))
        print("math_mul(4,5)        ->", _text(await client.call_tool("math_mul", {"a": 4, "b": 5})))
        print("text_upper('hi')     ->", _text(await client.call_tool("text_upper", {"s": "hi"})))
        print("text_reverse('abc')  ->", _text(await client.call_tool("text_reverse", {"s": "abc"})))

    print("\nimport_server = static copy; mount = live link. Prefixes avoid clashes.")


def main() -> int:
    if not HAVE:
        print(f"[skip] fastmcp not installed ({_ERR}) — pip install -r requirements.txt")
        return 0
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
