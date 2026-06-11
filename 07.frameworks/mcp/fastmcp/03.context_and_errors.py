"""03 · Context & error handling.

Two production concerns FastMCP makes easy:

  * **`Context`** — inject it by type-hinting a parameter ``ctx: Context``. From
    inside a handler you can log back to the host (``ctx.info/debug/warning``),
    report progress (``ctx.report_progress``), and stash per-request state
    (``ctx.set_state``/``get_state``). On stdio this is how you "print" safely —
    logs go through the protocol, not stdout.

  * **Errors** — raising ``ToolError`` returns a clean, *intentional* error to the
    client (the message is meant to be seen by the model/user). Other exceptions
    are also surfaced as tool errors, but ``ToolError`` is the explicit,
    safe-to-expose path. The client sees ``result.is_error == True``.

We drive both through the in-memory client so it runs offline.

Run:  python 03.context_and_errors.py   (exit 0; skips if fastmcp missing)
Docs: https://gofastmcp.com/servers/context  ·  .../tools (error handling)
"""

from __future__ import annotations

import asyncio
import sys

try:
    from fastmcp import Client, Context, FastMCP
    from fastmcp.exceptions import ToolError

    HAVE = True
except Exception as exc:  # pragma: no cover
    HAVE = False
    _ERR = exc


def build() -> "FastMCP":
    mcp = FastMCP("context-demo")

    @mcp.tool
    async def batch_sum(numbers: list[int], ctx: Context) -> int:
        """Sum integers, logging + reporting progress through Context."""
        total = 0
        n = len(numbers)
        for i, x in enumerate(numbers, start=1):
            total += x
            await ctx.info(f"added {x} ({i}/{n})")
            await ctx.report_progress(progress=i, total=n)
        return total

    @mcp.tool
    def safe_divide(a: float, b: float) -> float:
        """Divide a by b; raises a clean ToolError on division by zero."""
        if b == 0:
            raise ToolError("division by zero is undefined")
        return a / b

    return mcp


def _text(result) -> str:
    if getattr(result, "data", None) is not None:
        return str(result.data)
    return " ".join(getattr(b, "text", str(b)) for b in result.content)


async def run() -> None:
    mcp = build()

    # Capture the server's log messages so we can show Context.info in action.
    logs: list[str] = []

    async def on_log(message) -> None:
        logs.append(getattr(message, "data", str(message)))

    async with Client(mcp, log_handler=on_log) as client:
        print("[context] calling batch_sum([1,2,3,4,5]) ...")
        r = await client.call_tool("batch_sum", {"numbers": [1, 2, 3, 4, 5]})
        print("  result:", _text(r))
        print("  server logs received by client:")
        for line in logs:
            print("    -", line)

        print("\n[errors] safe_divide(10, 2):")
        ok = await client.call_tool("safe_divide", {"a": 10, "b": 2})
        print("  ->", _text(ok), "| is_error:", ok.is_error)

        print("[errors] safe_divide(1, 0)  (expect a clean ToolError):")
        try:
            bad = await client.call_tool("safe_divide", {"a": 1, "b": 0})
            print("  ->", _text(bad), "| is_error:", bad.is_error)
        except ToolError as e:
            # Depending on settings the client may raise instead of returning.
            print("  -> raised ToolError:", e)

    print("\nContext = safe logging/progress; ToolError = intentional, visible errors.")


def main() -> int:
    if not HAVE:
        print(f"[skip] fastmcp not installed ({_ERR}) — pip install -r requirements.txt")
        return 0
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
