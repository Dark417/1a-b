# MCP — the official Python SDK (`mcp`)

> **Model Context Protocol (MCP)** is an open standard that lets AI applications
> connect to external tools and data through a uniform protocol. A service ships
> an **MCP server**; any **MCP host** (Claude Code, Claude Desktop, an IDE, your
> own agent) connects to it and uses its capabilities. This folder teaches the
> **official Python SDK** — the `mcp` package — from the wire protocol up to a
> full server exposing **tools, resources, and prompts** plus a **client** that
> discovers and calls them over a real stdio round-trip, **offline**.
>
> Spec & docs: <https://modelcontextprotocol.io> ·
> SDK: <https://github.com/modelcontextprotocol/python-sdk> ·
> Spec revision used here: `2025-06-18`.

---

## 1. Why MCP exists — the mental model

Before MCP, every AI app hard-coded an integration for every service: one
bespoke adapter for GitHub, one for Postgres, one for your internal wiki. That is
an **M×N problem** — *M* apps times *N* services. MCP turns it into **M+N**: each
service implements **one** server, each app implements **one** client, and they
interoperate through a shared protocol. This is the same move USB-C made for
hardware, or the Language Server Protocol (LSP) made for editor tooling — MCP is
explicitly modelled on LSP.

```
        Without MCP (M×N)                With MCP (M+N)
   appA ─┬─ githubAdapter            appA ─┐         ┌─ githubServer
   appB ─┼─ githubAdapter                  ├─ MCP ───┤
   appA ─┼─ pgAdapter                appB ─┘         └─ pgServer
   appB ─┴─ pgAdapter
```

## 2. The participants (architecture in words)

MCP is **client–server**, and a *host* coordinates many clients:

* **Host** — the AI application (Claude Code, your agent). It runs the model and
  owns the conversation. It does **not** speak MCP directly; instead it spawns…
* **Client** — a connector object the host creates, **one per server**, holding a
  single stateful connection. The client is the thing that sends JSON-RPC.
* **Server** — a program exposing capabilities. It can run as a local subprocess
  (stdio transport) or as a remote web service (Streamable HTTP transport). The
  protocol is *identical* either way; "local" vs "remote" is only *where it runs*.

```
┌──────────────── MCP Host (your agent / Claude Code) ───────────────┐
│  model loop                                                        │
│    │                                                               │
│    ├── Client 1 ──stdio (subprocess pipes)──► Server A (filesystem)│
│    ├── Client 2 ──stdio───────────────────────► Server B (sqlite)  │
│    └── Client 3 ──Streamable HTTP (+OAuth)────► Server C (remote)   │
└────────────────────────────────────────────────────────────────────┘
```

## 3. Two layers: data + transport

MCP cleanly separates **what** is said from **how** it travels.

* **Data layer — JSON-RPC 2.0.** All messages are JSON-RPC: *requests* (have an
  `id`, expect a response), *responses* (carry `result` or `error` for that
  `id`), and *notifications* (no `id`, fire-and-forget). On top of JSON-RPC,
  MCP defines a **lifecycle** and the **primitives** (tools/resources/prompts).
* **Transport layer — how bytes move.**
  * **stdio** — the host launches the server as a subprocess and exchanges
    newline-delimited JSON over the server's **stdin/stdout**. Zero network, best
    for local tools. **`stderr` is free for logging** — never write protocol
    bytes to stdout that aren't JSON-RPC, or you corrupt the stream.
  * **Streamable HTTP** — a single HTTP endpoint. The client `POST`s JSON-RPC;
    the server replies either with a JSON body or with a **Server-Sent Events**
    stream (for incremental results / server→client messages). Supports
    standard web auth (OAuth 2.1 / bearer tokens). This is the transport for
    *remote* servers. (Conceptual here; we run stdio so it works offline.)

## 4. The lifecycle (the handshake)

MCP is **stateful**. Every connection begins with a capability negotiation:

```
client                                   server
  │  initialize (protocolVersion, caps) ─►│
  │�„── result (serverInfo, capabilities)  │
  │  notifications/initialized ──────────►│   (notification: no id, no reply)
  │                                        │
  │  tools/list ─────────────────────────►│
  │◄── { tools: [...] }                    │
  │  tools/call (name, arguments) ───────►│
  │◄── { content: [ {type:"text",...} ] }  │
```

1. **initialize** — client offers its `protocolVersion` + `capabilities`; server
   answers with `serverInfo` and the capabilities **it** supports (e.g.
   `tools.listChanged`). Version mismatch is negotiated or the connection fails.
2. **notifications/initialized** — client signals "ready". Only now may normal
   requests flow.
3. **discovery + use** — `tools/list` → `tools/call`, `resources/list` →
   `resources/read`, `prompts/list` → `prompts/get`.

A server that declared `listChanged` may push `notifications/tools/list_changed`
at any time, prompting the client to re-list.

## 5. The three server primitives

| Primitive | What it is | Who controls it | Discover / use |
|---|---|---|---|
| **Tools** | Executable functions the **model** chooses to call (search, write file, run query). | *Model*-controlled | `tools/list` → `tools/call` |
| **Resources** | Read-only data the **app** attaches as context (file contents, a DB row, an API doc). Addressed by **URI**. | *App*-controlled | `resources/list` → `resources/read` |
| **Prompts** | Parameterised interaction templates (a `/summarize` command, a few-shot block). | *User*-controlled (e.g. a slash command) | `prompts/list` → `prompts/get` |

The clean split matters: **tools** are for *actions the model decides to take*,
**resources** are for *context the application decides to provide*, and
**prompts** are for *workflows the user decides to invoke*. (The client can also
expose primitives back: **sampling** — ask the host's LLM to complete something;
**elicitation** — ask the user a question; **roots** — declare which folders the
client may access. We mention these; the common case is server→host tools.)

## 6. Two APIs in one SDK

The official SDK gives you **two** ways to write the same thing:

* **`mcp.server.fastmcp.FastMCP`** — the high-level, decorator-based builder.
  `@mcp.tool()`, `@mcp.resource("uri://{x}")`, `@mcp.prompt()` and it infers the
  JSON Schema from your type hints. (This is the *original* FastMCP, now folded
  into the SDK. The separate `fastmcp` 2.x/3.x project is a superset — see the
  `../fastmcp/` tutorial.) Use this for ~99% of servers.
* **`mcp.server.lowlevel.Server`** — the raw API: you register
  `@server.list_tools()` / `@server.call_tool()` handlers and return SDK types
  yourself. More boilerplate, total control (custom validation, dynamic tool
  sets, unusual lifecycles).

On the client side there is one path: **`ClientSession`** over a transport
(`stdio_client(...)` or `streamablehttp_client(...)`).

## 7. Install

```bash
pip install -r requirements.txt        # mcp>=1.9.0
# or the modern, recommended way (Astral uv):
uv add "mcp[cli]"      # the [cli] extra adds the `mcp` dev command + inspector
```

Quick smoke test with the official **Inspector** (a debugging UI, needs Node):

```bash
mcp dev 02.fastmcp_server.py     # opens the MCP Inspector against your server
```

> Everything here also runs with **no install at all**: each file falls back to a
> stdlib-only JSON-RPC implementation if `mcp` is missing, and the client↔server
> round-trip in `app.py` works offline either way.

## 8. The files in this tutorial

| File | Teaches |
|---|---|
| [`01.jsonrpc_wire.py`](01.jsonrpc_wire.py) | The wire protocol from scratch: a stdlib stdio server + client doing `initialize`/`tools/list`/`tools/call`. No deps — see exactly what crosses the pipe. |
| [`02.fastmcp_server.py`](02.fastmcp_server.py) | A full `FastMCP` server exposing **tools + resources + prompts**, with type-hinted schemas, context/logging, and dual stdio/HTTP run modes. |
| [`03.lowlevel_server.py`](03.lowlevel_server.py) | The same capabilities via the **low-level `Server`** API — register handlers, return SDK content types yourself. |
| [`04.client_session.py`](04.client_session.py) | The official **`ClientSession`** client: connect over stdio, run the lifecycle, list & call every primitive. |
| [`05.transports.py`](05.transports.py) | **stdio vs Streamable HTTP** explained and contrasted; builds an HTTP ASGI app (conceptual) and runs the stdio path live. |
| [`app.py`](app.py) | End-to-end: spins up the FastMCP server as a subprocess and drives a **real client↔server round-trip** over stdio, exercising tools, resources, and prompts. |

Each `.py` runs standalone (`python 0X_*.py`, exit 0) and prints what it did.

## 9. Comparison — when to reach for what

| Approach | Best for | Trade-off |
|---|---|---|
| **`FastMCP` (in `mcp`)** | almost every server; fastest to write | a little "magic" (schema inference) |
| **`fastmcp` 2.x/3.x** (separate pkg) | servers needing auth, proxying, OpenAPI→MCP, server composition | extra dependency; superset API |
| **low-level `Server`** | dynamic tool sets, custom validation, odd lifecycles | verbose |
| **hand-rolled JSON-RPC** (file 01) | learning; embedding in a constrained runtime | you maintain the protocol |
| **stdio transport** | local tools, dev, Claude Desktop/Code config | one client per process; same machine |
| **Streamable HTTP** | remote/shared servers, multi-client, auth | run a web service; handle sessions/auth |

**MCP vs. a plain function-calling tool registry:** function calling is
in-process and bespoke per app; MCP is an *out-of-process, language-agnostic,
discoverable* contract — write the server once, every MCP host gets it for free.

## 10. Gotchas (hard-won)

* **stdout is sacred on stdio.** A stray `print()` in your server corrupts the
  JSON-RPC stream. Log to **stderr** (or use `Context.info(...)`). FastMCP does
  this for you; in raw code, `print(..., file=sys.stderr)`.
* **The lifecycle is mandatory.** You must `initialize` *and* send
  `notifications/initialized` before any `tools/call`. `ClientSession` does it in
  `session.initialize()`; if you hand-roll, don't skip step 2.
* **Tool results are content blocks, not strings.** A result is
  `{"content": [{"type":"text","text":"..."}], "isError": false}`. Concatenate
  the text blocks; check `isError`.
* **Schemas come from type hints (FastMCP).** Annotate every parameter or the
  inferred `inputSchema` will be loose and the model will misuse the tool. Add a
  docstring — it becomes the tool `description`.
* **Resources are read-only context, not actions.** If the model needs to *do*
  something, it's a tool. If the app wants to *attach* data, it's a resource.
* **Async everywhere.** The SDK is asyncio-based; servers/clients run inside
  `anyio`/`asyncio`. Don't block the event loop with sync I/O in a handler.
* **Version negotiation can fail.** If client and server share no protocol
  version, `initialize` errors — pin compatible SDK versions.

## 11. References

* MCP spec & concepts — <https://modelcontextprotocol.io/docs/concepts>
* Lifecycle & transports — <https://modelcontextprotocol.io/specification/2025-06-18>
* Python SDK — <https://github.com/modelcontextprotocol/python-sdk>
* JSON-RPC 2.0 — <https://www.jsonrpc.org/specification>
* Repo's own clean-room MCP primer — `../../../08.claudecode/docs/05.mcp.md`
