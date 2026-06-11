"""02 · A full FastMCP server — tools + resources + prompts.

``FastMCP`` (shipped *inside* the official ``mcp`` SDK as
``mcp.server.fastmcp.FastMCP``) is the high-level, decorator-based way to build a
server. You write ordinary typed Python functions; FastMCP infers the JSON
Schema from your type hints, turns docstrings into descriptions, and handles the
whole JSON-RPC lifecycle and transports for you.

This file builds a **complete** server exposing all three primitives:

  * **tools**     — actions the model may call (``add``, ``word_count``,
    ``slugify``, plus an async tool that uses ``Context`` for logging/progress);
  * **resources** — read-only context addressed by URI, both a *fixed* resource
    (``config://server``) and a *templated* one (``greeting://{name}``);
  * **prompts**   — reusable templates (``summarize``, ``code_review``).

Run modes (see ``run_server`` at the bottom):
  * ``python 02.fastmcp_server.py``          → **introspection demo** (default):
    builds the server and prints every registered primitive, then exits 0.
    No client, no network — proves the server is well-formed offline.
  * ``python 02.fastmcp_server.py --stdio``  → run as a real **stdio** MCP server
    (this is the mode ``app.py`` and ``04.client_session.py`` launch).
  * ``python 02.fastmcp_server.py --http``   → run over **Streamable HTTP**
    (needs ``uvicorn``; marked optional, see ``05.transports.py``).

Docs: https://github.com/modelcontextprotocol/python-sdk#quickstart
"""

from __future__ import annotations

import asyncio
import sys

try:
    from mcp.server.fastmcp import Context, FastMCP

    HAVE_MCP = True
except Exception as exc:  # pragma: no cover - exercised only when mcp missing
    HAVE_MCP = False
    _IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Build the server (only if the SDK is available).
# ---------------------------------------------------------------------------
def build_server() -> "FastMCP":
    """Construct and return a fully-populated FastMCP server."""
    mcp = FastMCP(
        name="demo-fastmcp-server",
        instructions="A demo server exposing math/text tools, config resources, "
        "and prompt templates. Built for the 07.frameworks MCP tutorial.",
    )

    # -- TOOLS --------------------------------------------------------------
    # Type hints become the inputSchema; the docstring becomes the description.
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two integers and return the sum."""
        return a + b

    @mcp.tool()
    def word_count(text: str) -> int:
        """Count whitespace-separated words in a string."""
        return len(text.split())

    @mcp.tool()
    def slugify(text: str, sep: str = "-") -> str:
        """Turn arbitrary text into a URL-safe slug.

        Args:
            text: the text to slugify.
            sep: the separator between words (default '-').
        """
        keep = [c.lower() if c.isalnum() else " " for c in text]
        return sep.join("".join(keep).split())

    # An async tool that uses Context — the bridge back to the host for logging,
    # progress reporting, and resource reads from *inside* a tool.
    @mcp.tool()
    async def batch_add(numbers: list[int], ctx: Context) -> int:
        """Sum a list of integers, reporting progress as it goes."""
        total = 0
        n = len(numbers)
        for i, x in enumerate(numbers, start=1):
            total += x
            await ctx.info(f"added {x} ({i}/{n})")  # → logged on the host, via stderr
            await ctx.report_progress(progress=i, total=n)
        return total

    # -- RESOURCES ----------------------------------------------------------
    # A fixed resource: a single URI returning some context. App-controlled.
    @mcp.resource("config://server")
    def server_config() -> str:
        """Static server configuration, as text."""
        return "name=demo-fastmcp-server\nversion=0.1.0\nmax_items=100"

    # A *templated* resource: the {name} segment becomes a parameter. The host
    # lists it under resource *templates* and fills the URI to read it.
    @mcp.resource("greeting://{name}")
    def greeting(name: str) -> str:
        """A personalized greeting resource."""
        return f"Hello, {name}! This text came from an MCP resource."

    # -- PROMPTS ------------------------------------------------------------
    # Prompts are user-invoked templates (think slash commands). Returning a
    # plain string yields a single user message; you can also return message
    # lists for multi-turn / few-shot templates.
    @mcp.prompt()
    def summarize(text: str, max_words: int = 50) -> str:
        """Build a summarization prompt for the given text."""
        return f"Summarize the following in at most {max_words} words:\n\n{text}"

    @mcp.prompt()
    def code_review(code: str, language: str = "python") -> str:
        """Build a code-review prompt."""
        return (
            f"You are a senior {language} engineer. Review this code for bugs, "
            f"clarity, and idiomatic style. Be specific.\n\n```{language}\n{code}\n```"
        )

    return mcp


# ---------------------------------------------------------------------------
# Introspection demo (default) — prove the server is well-formed, offline.
# ---------------------------------------------------------------------------
async def _introspect(mcp: "FastMCP") -> None:
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    prompts = await mcp.list_prompts()

    print("Server:", mcp.name)
    print("\nTOOLS")
    for t in tools:
        params = list((t.inputSchema or {}).get("properties", {}))
        print(f"  • {t.name}({', '.join(params)}) — {t.description}")
    print("\nRESOURCES (fixed)")
    for r in resources:
        print(f"  • {r.uri} — {r.description}")
    print("\nRESOURCES (templated)")
    for t in templates:
        print(f"  • {t.uriTemplate} — {t.description}")
    print("\nPROMPTS")
    for p in prompts:
        args = [a.name for a in (p.arguments or [])]
        print(f"  • {p.name}({', '.join(args)}) — {p.description}")

    # Actually invoke one tool through the server machinery to prove it works.
    result = await mcp.call_tool("slugify", {"text": "Hello, MCP World!"})
    print("\ncall_tool slugify('Hello, MCP World!') ->", _text_of(result))


def _text_of(result: object) -> str:
    """call_tool returns (content_blocks, structured) across SDK versions."""
    blocks = result[0] if isinstance(result, tuple) else result
    out = []
    for b in blocks:
        out.append(getattr(b, "text", str(b)))
    return " ".join(out)


def run_server() -> int:
    if not HAVE_MCP:
        print("[skip] `mcp` not installed — pip install -r requirements.txt")
        print(f"       ({_IMPORT_ERROR})")
        print("       (See 01.jsonrpc_wire.py for a stdlib-only round-trip.)")
        return 0

    mcp = build_server()

    if "--stdio" in sys.argv:
        # Real stdio server: blocks, serving JSON-RPC on stdin/stdout.
        mcp.run(transport="stdio")
        return 0
    if "--http" in sys.argv:
        try:
            mcp.settings.host, mcp.settings.port = "127.0.0.1", 8000
            print("[info] serving Streamable HTTP on http://127.0.0.1:8000/mcp")
            mcp.run(transport="streamable-http")
        except Exception as exc:  # uvicorn missing, port busy, etc.
            print(f"[skip] HTTP transport unavailable: {exc}")
        return 0

    # Default: introspection demo (no client needed).
    print("FastMCP server — introspection demo (offline)\n")
    asyncio.run(_introspect(mcp))
    print("\nServer is well-formed. Run with --stdio to serve a real client,")
    print("or see app.py for a full client↔server round-trip.")
    return 0


if __name__ == "__main__":
    sys.exit(run_server())
