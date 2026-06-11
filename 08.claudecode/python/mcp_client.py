"""mcp_client.py — a minimal Model Context Protocol (MCP) client.

MCP is the open standard Claude Code uses to connect to external tools/data via
*servers* (``docs/05.mcp.md``). The wire protocol is **JSON-RPC 2.0** over a
transport (stdio for local servers, Streamable HTTP for remote). The lifecycle
is: ``initialize`` (capability negotiation) → ``notifications/initialized`` →
``tools/list`` to discover tools → ``tools/call`` to invoke them.

This module implements a small but *real* stdio JSON-RPC client and ships a
bundled trivial server (``mcp_server_example.py``) so the whole loop runs
offline with only the Python stdlib. The client then exposes discovered MCP
tools as :class:`~tools.Tool` objects so the agent can call them like any other
tool — exactly how an MCP host adapts server tools into its tool registry.

This is an educational implementation of the public MCP spec, not Anthropic's
code. See ``docs/07.references.md``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from tools import Tool, ToolResult


PROTOCOL_VERSION = "2025-06-18"


class StdioMCPClient:
    """Speaks JSON-RPC 2.0 to an MCP server over its stdin/stdout pipes."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.proc: subprocess.Popen[str] | None = None
        self._id = 0
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}

    # -- transport ----------------------------------------------------------

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.stdin and self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    def __enter__(self) -> "StdioMCPClient":
        self.start()
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self.proc and self.proc.stdout
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection")
        return json.loads(line)

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return its ``result`` (or raise on error)."""
        rid = self._next_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        # Skip any notifications until we get the matching response.
        while True:
            msg = self._read()
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg.get("result", {})

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # -- lifecycle ----------------------------------------------------------

    def initialize(self) -> None:
        """Negotiate capabilities, then announce readiness."""
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mini-agent-mcp-client", "version": "0.1.0"},
        })
        self.server_info = result.get("serverInfo", {})
        self.capabilities = result.get("capabilities", {})
        self._notify("notifications/initialized")

    # -- primitives ---------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        # Content is an array of blocks; concatenate text blocks.
        parts = [b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"]
        return "\n".join(parts)


def mcp_tools_as_agent_tools(client: StdioMCPClient, prefix: str = "mcp__example__") -> list[Tool]:
    """Adapt the server's MCP tools into :class:`~tools.Tool` objects.

    Each MCP tool becomes a normal agent tool whose ``func`` calls back into the
    MCP server. Names are namespaced ``mcp__<server>__<tool>`` exactly like
    Claude Code surfaces MCP tools.
    """
    adapted: list[Tool] = []
    for spec in client.list_tools():
        name = spec["name"]

        def make_func(tool_name: str):
            def _call(**kwargs: Any) -> ToolResult:
                return ToolResult(client.call_tool(tool_name, kwargs))
            return _call

        adapted.append(Tool(
            name=prefix + name,
            description=spec.get("description", ""),
            parameters=spec.get("inputSchema", {"type": "object", "properties": {}}),
            func=make_func(name),
            read_only=False,
        ))
    return adapted


# ---------------------------------------------------------------------------
# Self-test: spin up the bundled server, initialize, list, and call a tool.
# ---------------------------------------------------------------------------

def _demo() -> None:
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    server = os.path.join(here, "mcp_server_example.py")
    with StdioMCPClient([sys.executable, server]) as client:
        print("connected to:", client.server_info)
        tools = client.list_tools()
        print("tools:", [t["name"] for t in tools])
        print("add(2,3) =>", client.call_tool("add", {"a": 2, "b": 3}))
        print("upper('hi') =>", client.call_tool("upper", {"text": "hi"}))


if __name__ == "__main__":  # pragma: no cover
    _demo()
