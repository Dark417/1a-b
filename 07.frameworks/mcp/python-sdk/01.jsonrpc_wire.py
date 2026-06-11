"""01 · The MCP wire protocol, from scratch (stdlib only).

MCP's *data layer* is plain **JSON-RPC 2.0** over a transport. Before reaching
for the SDK, it's worth seeing *exactly* what crosses the pipe. This file
implements both ends with nothing but the standard library:

  * a tiny **stdio MCP server** (a separate Python subprocess) that handles
    ``initialize`` / ``tools/list`` / ``tools/call`` and speaks newline-delimited
    JSON-RPC on stdin/stdout;
  * a tiny **stdio client** that drives the full lifecycle and prints every
    message so you can read the conversation.

There are no third-party deps here on purpose: this is the protocol with the
training wheels off. The later files use the official ``mcp`` SDK, which
implements this same wire format (and much more) for you.

Spec: https://modelcontextprotocol.io/specification/2025-06-18
JSON-RPC: https://www.jsonrpc.org/specification

Run:  python 01.jsonrpc_wire.py
Exit 0 offline, no network, no installs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


# ---------------------------------------------------------------------------
# SERVER MODE — when launched as `python 01.jsonrpc_wire.py --serve`, this
# process becomes an MCP server, reading JSON-RPC requests from stdin and
# writing responses to stdout. ALL logging goes to stderr (stdout is sacred).
# ---------------------------------------------------------------------------

# The toy "tools" this server exposes. Each has a name, description, and a
# JSON-Schema describing its arguments (this is what `tools/list` returns).
SERVER_TOOLS = [
    {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "shout",
        "description": "Uppercase a string.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def _run_tool(name: str, args: dict[str, Any]) -> str:
    if name == "add":
        return str(int(args["a"]) + int(args["b"]))
    if name == "shout":
        return str(args["text"]).upper()
    raise ValueError(f"unknown tool: {name}")


def serve() -> None:
    """Blocking stdio JSON-RPC loop. One JSON object per line."""
    log = lambda *a: print("[server]", *a, file=sys.stderr, flush=True)
    log("started; waiting for initialize")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params", {})

        # Notifications have no `id` and never get a response.
        if mid is None:
            log(f"notification: {method}")
            continue

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "wire-demo-server", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": SERVER_TOOLS}
            elif method == "tools/call":
                text = _run_tool(params["name"], params.get("arguments", {}))
                # A tool result is an array of typed content blocks.
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            else:
                raise ValueError(f"unknown method: {method}")
            reply = {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as exc:  # JSON-RPC error object
            reply = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}

        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    log("stdin closed; exiting")


# ---------------------------------------------------------------------------
# CLIENT MODE — the default. Spawn the server subprocess and talk to it.
# ---------------------------------------------------------------------------

class WireClient:
    """A minimal stdio JSON-RPC client. Mirrors what `ClientSession` does."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.proc: subprocess.Popen[str] | None = None
        self._id = 0

    def __enter__(self) -> "WireClient":
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # let the server's logs show through
            text=True,
            bufsize=1,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        wire = json.dumps(obj)
        print("  → ", wire)
        self.proc.stdin.write(wire + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> dict:
        assert self.proc and self.proc.stdout
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:  # skip notifications until our id comes back
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server closed connection")
            reply = json.loads(line)
            if reply.get("id") == rid:
                print("  ← ", line.strip())
                if "error" in reply:
                    raise RuntimeError(reply["error"]["message"])
                return reply.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})


def main() -> int:
    print("MCP wire protocol — stdlib client ↔ server (offline)\n")
    with WireClient([sys.executable, os.path.abspath(__file__), "--serve"]) as client:
        print("1) initialize (capability negotiation)")
        info = client.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "wire-demo-client", "version": "0.1.0"},
        })
        print("   serverInfo:", info["serverInfo"], "\n")

        print("2) notifications/initialized (no response expected)")
        client.notify("notifications/initialized")
        print()

        print("3) tools/list (discovery)")
        tools = client.request("tools/list")["tools"]
        print("   tools:", [t["name"] for t in tools], "\n")

        print("4) tools/call")
        for name, args in [("add", {"a": 40, "b": 2}), ("shout", {"text": "mcp"})]:
            res = client.request("tools/call", {"name": name, "arguments": args})
            text = "".join(b["text"] for b in res["content"] if b["type"] == "text")
            print(f"   {name}{args} = {text}\n")

    print("Round-trip complete. That is the whole MCP data layer.")
    return 0


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        sys.exit(main())
