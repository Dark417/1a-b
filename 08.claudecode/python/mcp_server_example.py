"""mcp_server_example.py — a trivial MCP server (stdio, JSON-RPC 2.0).

A bundled, dependency-free example server so ``mcp_client.py`` can exercise a
full MCP loop offline. It implements just enough of the spec to be real:

* ``initialize``                 → returns serverInfo + capabilities
* ``notifications/initialized``  → (ignored; no response for notifications)
* ``tools/list``                 → advertises two tiny tools
* ``tools/call``                 → executes them and returns content blocks

It speaks newline-delimited JSON-RPC over stdin/stdout, matching the client.
Educational implementation of the public MCP spec (``docs/05.mcp.md``).
"""

from __future__ import annotations

import json
import sys
from typing import Any


PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "add",
        "title": "Add",
        "description": "Add two integers and return the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "upper",
        "title": "Uppercase",
        "description": "Uppercase a string.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "example-mcp-server", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params["name"]
        args = params.get("arguments", {})
        return {"content": [{"type": "text", "text": _run_tool(name, args)}]}
    raise ValueError(f"unknown method {method!r}")


def _run_tool(name: str, args: dict[str, Any]) -> str:
    if name == "add":
        return str(int(args["a"]) + int(args["b"]))
    if name == "upper":
        return str(args["text"]).upper()
    raise ValueError(f"unknown tool {name!r}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method", "")
        # Notifications have no id and expect no response.
        if "id" not in msg:
            continue
        try:
            result = handle(method, msg.get("params", {}))
            response = {"jsonrpc": "2.0", "id": msg["id"], "result": result}
        except Exception as e:  # JSON-RPC error object
            response = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {"code": -32603, "message": str(e)},
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
