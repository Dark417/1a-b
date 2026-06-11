"""03 · The low-level `Server` API — same capabilities, full control.

``FastMCP`` (file 02) is sugar over the SDK's low-level
``mcp.server.lowlevel.Server``. When you need *total* control — dynamic tool
sets, custom validation, unusual lifecycles, or hand-built schemas — you drop to
this layer. You register one handler per protocol method and return SDK types
(``types.Tool``, ``types.TextContent``, …) yourself; there is no schema
inference, so you write the ``inputSchema`` by hand.

This server exposes the same three primitives as file 02 so you can compare the
ergonomics directly:

  * ``@server.list_tools()`` / ``@server.call_tool()``
  * ``@server.list_resources()`` / ``@server.read_resource()``
  * ``@server.list_prompts()`` / ``@server.get_prompt()``

Run modes:
  * ``python 03.lowlevel_server.py``         → introspection demo (default).
  * ``python 03.lowlevel_server.py --stdio`` → run as a real stdio MCP server.

Docs: https://github.com/modelcontextprotocol/python-sdk (Low-Level Server)
"""

from __future__ import annotations

import asyncio
import sys

try:
    import mcp.types as types
    from mcp.server.lowlevel import Server

    HAVE_MCP = True
except Exception as exc:  # pragma: no cover
    HAVE_MCP = False
    _IMPORT_ERROR = exc


def build_server() -> "Server":
    server = Server("demo-lowlevel-server")

    # -- TOOLS --------------------------------------------------------------
    @server.list_tools()
    async def list_tools() -> list["types.Tool"]:
        # You author the JSON Schema yourself — nothing is inferred.
        return [
            types.Tool(
                name="add",
                description="Add two integers and return the sum.",
                inputSchema={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            ),
            types.Tool(
                name="reverse",
                description="Reverse a string.",
                inputSchema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list["types.TextContent"]:
        if name == "add":
            out = str(int(arguments["a"]) + int(arguments["b"]))
        elif name == "reverse":
            out = str(arguments["text"])[::-1]
        else:
            raise ValueError(f"unknown tool: {name}")
        # Return content blocks explicitly — the wire format, in Python.
        return [types.TextContent(type="text", text=out)]

    # -- RESOURCES ----------------------------------------------------------
    @server.list_resources()
    async def list_resources() -> list["types.Resource"]:
        return [
            types.Resource(
                uri="config://server",
                name="server_config",
                description="Static server configuration.",
                mimeType="text/plain",
            )
        ]

    @server.read_resource()
    async def read_resource(uri) -> str:
        if str(uri) == "config://server":
            return "name=demo-lowlevel-server\nversion=0.1.0"
        raise ValueError(f"unknown resource: {uri}")

    # -- PROMPTS ------------------------------------------------------------
    @server.list_prompts()
    async def list_prompts() -> list["types.Prompt"]:
        return [
            types.Prompt(
                name="summarize",
                description="Summarize text in N words.",
                arguments=[
                    types.PromptArgument(name="text", description="text to summarize", required=True),
                    types.PromptArgument(name="max_words", description="word budget", required=False),
                ],
            )
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict | None) -> "types.GetPromptResult":
        arguments = arguments or {}
        if name != "summarize":
            raise ValueError(f"unknown prompt: {name}")
        n = arguments.get("max_words", 50)
        text = arguments.get("text", "")
        return types.GetPromptResult(
            description="A summarization prompt.",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text", text=f"Summarize in {n} words:\n\n{text}"
                    ),
                )
            ],
        )

    return server


# ---------------------------------------------------------------------------
# Introspection demo — call the registered handlers directly (no transport).
# ---------------------------------------------------------------------------
async def _introspect(server: "Server") -> None:
    # The decorators store handlers in server.request_handlers keyed by the
    # *request type*. For a no-transport demo we just call our closures via the
    # handler the SDK registered. Simplest: rebuild + call the inner coroutines
    # through the public list/* request handlers.
    list_tools = server.request_handlers[types.ListToolsRequest]
    call_tool = server.request_handlers[types.CallToolRequest]
    list_res = server.request_handlers[types.ListResourcesRequest]
    list_prompts = server.request_handlers[types.ListPromptsRequest]

    tools_resp = await list_tools(types.ListToolsRequest(method="tools/list"))
    print("TOOLS:", [t.name for t in tools_resp.root.tools])

    res_resp = await list_res(types.ListResourcesRequest(method="resources/list"))
    print("RESOURCES:", [str(r.uri) for r in res_resp.root.resources])

    pr_resp = await list_prompts(types.ListPromptsRequest(method="prompts/list"))
    print("PROMPTS:", [p.name for p in pr_resp.root.prompts])

    call = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="reverse", arguments={"text": "stdio"}),
    )
    call_resp = await call_tool(call)
    text = "".join(getattr(c, "text", "") for c in call_resp.root.content)
    print("call reverse('stdio') ->", text)


def run_server() -> int:
    if not HAVE_MCP:
        print("[skip] `mcp` not installed — pip install -r requirements.txt")
        print(f"       ({_IMPORT_ERROR})")
        return 0

    server = build_server()

    if "--stdio" in sys.argv:
        import mcp.server.stdio

        async def _serve() -> None:
            async with mcp.server.stdio.stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(_serve())
        return 0

    print("Low-level Server — introspection demo (offline)\n")
    asyncio.run(_introspect(server))
    print("\nSame primitives as FastMCP, but every schema/handler is explicit.")
    return 0


if __name__ == "__main__":
    sys.exit(run_server())
