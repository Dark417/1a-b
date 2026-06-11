"""01 · FastMCP decorators — tools, resources, prompts.

The whole FastMCP value proposition: write a normal typed Python function,
decorate it, and you have an MCP primitive with a JSON Schema inferred from the
type hints and a description taken from the docstring. This file builds one
server exposing all three primitive kinds, including a *templated* resource and a
*structured* (dict-returning) tool, then introspects it offline.

Run:  python 01.decorators.py   (exit 0; skips gracefully if fastmcp missing)
Docs: https://gofastmcp.com/servers/tools
"""

from __future__ import annotations

import asyncio
import sys

try:
    from fastmcp import FastMCP

    HAVE = True
except Exception as exc:  # pragma: no cover
    HAVE = False
    _ERR = exc


def build() -> "FastMCP":
    mcp = FastMCP("decorators-demo")

    # -- TOOLS: type hints -> inputSchema, docstring -> description ----------
    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @mcp.tool
    def slugify(text: str, sep: str = "-") -> str:
        """Make a URL-safe slug from text."""
        clean = "".join(c.lower() if c.isalnum() else " " for c in text)
        return sep.join(clean.split())

    # A *structured* tool: returning a dict yields a JSON output schema too.
    @mcp.tool
    def stats(numbers: list[float]) -> dict:
        """Return count, sum, and mean of a list of numbers."""
        n = len(numbers)
        s = sum(numbers)
        return {"count": n, "sum": s, "mean": (s / n) if n else 0.0}

    # -- RESOURCES: fixed and templated -------------------------------------
    @mcp.resource("config://app")
    def config() -> str:
        """Static app config."""
        return "theme=dark\nlocale=en"

    @mcp.resource("user://{uid}/profile")
    def profile(uid: str) -> dict:
        """A templated resource: {uid} is filled at read time."""
        return {"id": uid, "name": f"user-{uid}", "active": True}

    # -- PROMPTS ------------------------------------------------------------
    @mcp.prompt
    def code_review(code: str, language: str = "python") -> str:
        """Build a code-review prompt."""
        return f"Review this {language} code:\n\n```{language}\n{code}\n```"

    return mcp


async def introspect(mcp: "FastMCP") -> None:
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    prompts = await mcp.list_prompts()

    print("TOOLS")
    for t in tools:
        params = list((t.parameters or {}).get("properties", {}))
        print(f"  • {t.name}({', '.join(params)}) — {t.description}")
    print("\nRESOURCES (fixed)")
    for r in resources:
        print(f"  • {r.uri}")
    print("\nRESOURCES (templated)")
    for tmpl in templates:
        print(f"  • {tmpl.uri_template}")
    print("\nPROMPTS")
    for p in prompts:
        print(f"  • {p.name}")


def main() -> int:
    if not HAVE:
        print(f"[skip] fastmcp not installed ({_ERR}) — pip install -r requirements.txt")
        return 0
    print("FastMCP decorators — introspection (offline)\n")
    asyncio.run(introspect(build()))
    print("\nDecorate a typed fn, get an MCP primitive. See 02 for a live client.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
