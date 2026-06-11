"""app.py · End-to-end MCP demo — a host that drives a real server.

This ties the folder together. It plays the role of an **MCP host**: it launches
the FastMCP server (``02.fastmcp_server.py``) as a stdio subprocess, opens a
``ClientSession``, runs the full lifecycle, then **adapts every discovered tool
into the host's own callable registry** — exactly how Claude Code merges MCP
server tools (namespaced ``mcp__<server>__<tool>``) alongside its built-ins.

It then runs a tiny *agentic loop*: a deterministic **MockLLM** "decides" which
MCP tool to call for a couple of tasks, the host routes the call to the server,
and the result comes back. No API key, no network — the MockLLM stands in for a
real model so the whole thing is reproducible offline.

If ``mcp`` is unavailable, it falls back to the stdlib wire client from
``01.jsonrpc_wire.py`` so the round-trip *still* runs.

Run:  python app.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "02.fastmcp_server.py")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    HAVE_MCP = True
except Exception:  # pragma: no cover
    HAVE_MCP = False


# ---------------------------------------------------------------------------
# A deterministic MockLLM: maps a task string to a (tool, args) decision.
# Stands in for a real model's tool-choice so the demo is reproducible offline.
# ---------------------------------------------------------------------------
class MockLLM:
    """Rule-based 'reasoning' over the available tool names. Deterministic."""

    def __init__(self, tool_names: list[str]) -> None:
        self.tool_names = set(tool_names)

    def decide(self, task: str) -> tuple[str, dict[str, Any]] | None:
        t = task.lower()
        m = re.search(r"(-?\d+)\D+(-?\d+)", task)
        if "add" in t and "add" in self.tool_names and m:
            return "add", {"a": int(m.group(1)), "b": int(m.group(2))}
        if ("slug" in t or "url" in t) and "slugify" in self.tool_names:
            payload = task.split(":", 1)[-1].strip() or task
            return "slugify", {"text": payload}
        if ("count" in t or "words" in t) and "word_count" in self.tool_names:
            payload = task.split(":", 1)[-1].strip() or task
            return "word_count", {"text": payload}
        return None


# ---------------------------------------------------------------------------
# Host registry: MCP tools adapted into namespaced callables (sync facade).
# ---------------------------------------------------------------------------
class HostRegistry:
    def __init__(self, prefix: str = "mcp__demo__") -> None:
        self.prefix = prefix
        self.tools: dict[str, Callable[..., str]] = {}
        self.descriptions: dict[str, str] = {}

    def register(self, name: str, func: Callable[..., str], desc: str) -> None:
        self.tools[name] = func
        self.descriptions[name] = desc


async def build_registry(session: "ClientSession") -> HostRegistry:
    reg = HostRegistry()
    tools = (await session.list_tools()).tools
    for spec in tools:
        async def call(arguments, _name=spec.name):
            res = await session.call_tool(_name, arguments)
            return " ".join(getattr(b, "text", str(b)) for b in res.content)

        reg.register(reg.prefix + spec.name, call, spec.description or "")
    return reg


async def run_with_sdk() -> int:
    params = StdioServerParameters(command=sys.executable, args=[SERVER, "--stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"[host] connected to MCP server: {init.serverInfo.name}\n")

            reg = await build_registry(session)
            short_names = [n.replace(reg.prefix, "") for n in reg.tools]
            print("[host] merged MCP tools into registry:")
            for full in reg.tools:
                print(f"         {full}")
            print()

            llm = MockLLM(short_names)
            tasks = [
                "Please add 19 and 23",
                "slugify: My First MCP App!",
                "count the words: model context protocol rocks",
                "what's the weather",  # no matching tool -> graceful
            ]
            for task in tasks:
                print(f"[user] {task}")
                decision = llm.decide(task)
                if decision is None:
                    print("[host] MockLLM: no MCP tool matches; would answer directly.\n")
                    continue
                short, args = decision
                full = reg.prefix + short
                result = await reg.tools[full](args)
                print(f"[host] -> {full}({args}) = {result}\n")

            print("[host] done. A real host does exactly this, with a real model.")
    return 0


# ---------------------------------------------------------------------------
# Fallback: reuse the stdlib wire client from file 01 so we still round-trip.
# ---------------------------------------------------------------------------
def run_fallback() -> int:
    print("[host] `mcp` SDK missing — falling back to the stdlib wire client.\n")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wire", os.path.join(HERE, "01.jsonrpc_wire.py")
    )
    wire = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(wire)

    with wire.WireClient([sys.executable, os.path.join(HERE, "01.jsonrpc_wire.py"), "--serve"]) as c:
        c.request("initialize", {"protocolVersion": wire.PROTOCOL_VERSION,
                                 "capabilities": {}, "clientInfo": {"name": "host", "version": "0"}})
        c.notify("notifications/initialized")
        tools = c.request("tools/list")["tools"]
        print("[host] tools:", [t["name"] for t in tools])
        res = c.request("tools/call", {"name": "add", "arguments": {"a": 19, "b": 23}})
        print("[host] add(19,23) =", res["content"][0]["text"])
    return 0


def main() -> int:
    print("=== MCP end-to-end host demo (offline) ===\n")
    if HAVE_MCP:
        return asyncio.run(run_with_sdk())
    return run_fallback()


if __name__ == "__main__":
    sys.exit(main())
